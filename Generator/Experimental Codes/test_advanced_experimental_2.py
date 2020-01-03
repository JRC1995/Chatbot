# Modified from: https://colab.research.google.com/drive/1-_KjlAV3J1IVDw_9KogjKDCzgFY7Jp7E#scrollTo=RZd4iXX5SqKC

import os
import torch as T
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np
import sys
import pickle
import math

sys.path.append("../")
import Utils.functions as utils

with open('../Sentence_Encoder/Embeddings/ParaNMT/ParaNMT.pkl', 'rb') as fp:
    ParaNMT_embd = pickle.load(fp)


def para_encode(text):

    global ParaNMT_embd

    text, tokenized_text = utils.simple_preprocess(text)

    paraembds = []

    for token in tokenized_text:
        if token in ParaNMT_embd:
            paraembds.append(ParaNMT_embd[token])

    for word in ParaNMT_embd:
        paradim = ParaNMT_embd[word].shape[0]
        break

    paravec = np.zeros((paradim), np.float32)

    if paraembds:
        paravec = np.sum(np.asarray(paraembds), axis=0)/math.sqrt(len(paraembds))

    return paravec

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
weights = T.load('DialoGPT/Parameters/medium_ft.pkl')
weights_reverse = T.load('DialoGPT/Parameters/small_reverse.pkl')
cfg = GPT2Config.from_json_file('DialoGPT/Configs/config.json')
model = GPT2LMHeadModel(cfg)
model_reverse = GPT2LMHeadModel(cfg)

# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)
weights_reverse["lm_head.weight"] = weights_reverse["lm_head.decoder.weight"]
weights_reverse.pop("lm_head.decoder.weight", None)

model.load_state_dict(weights)
model.to('cpu')
model.eval()

model_reverse.load_state_dict(weights_reverse)
model_reverse.to('cpu')
model_reverse.eval()

EOS_token = tokenizer.encode("<|endoftext|>")[0]

print(EOS_token)

conv_history = ["How can I destroy the world?","what kind of question is this?"]

msg_idx = []
for msg in conv_history:
    msg_idx.append(tokenizer.encode(msg))

# adapted from: https://github.com/microsoft/DialoGPT/blob/master/prepro.py
def _make_feature(sents, eos):
    input_ids = [i for s in sents for i in s+[eos]][:-1]
    token_type_ids = []
    for i, s in enumerate(sents):
        if i == 0:
            token_type_ids += [0] * len(s)
            continue

        token_type_ids += [i] * (len(s) + 1)

    input_ids.append(eos)

    return input_ids

input_ids = _make_feature(msg_idx,EOS_token)

print(input_ids)

input_ids = T.tensor(input_ids).long().to('cpu').unsqueeze(0)

def next(logits, top_k=100, top_p=0.9):

    filter_val = -2.0**32

    G, B, V = logits.size()

    top_k = min(top_k, V)  # Safety check

    sorted_logits, sorted_indices = T.sort(logits, descending=True)

    cumulative_probs = T.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_logits = T.where(cumulative_probs>top_p,
                            T.empty_like(sorted_logits).to('cpu').fill_(filter_val),
                            sorted_logits)

    probabilities = F.softmax(sorted_logits, dim=-1)
    probabilities = probabilities.view(G*B,V)

    try:
        sorted_candidate_idx = T.multinomial(probabilities, num_samples=top_k, replacement=False)
    except:
        sorted_candidate_idx = T.multinomial(probabilities, num_samples=top_k, replacement=True)

    sorted_candidate_idx = sorted_candidate_idx.view(G,B,top_k)

    next_candidates = T.gather(sorted_indices,dim=-1,index=sorted_candidate_idx)

    log_probs = T.gather(T.log(probabilities.view(G,B,V)+1e-9),
                         dim=-1,
                         index=sorted_candidate_idx)

    return next_candidates,log_probs


def generate_candidates(hypotheses):

    hypotheses = hypotheses[:,:,1:]
    G,B,S = hypotheses.size()
    EOS_token = tokenizer.encode("<|endoftext|>")[0]

    hypotheses_list = hypotheses.cpu().numpy().tolist()

    candidates = []

    for group in hypotheses_list:
        for beam in group:
            print(beam)
            if beam[-1] == EOS_token:
                candidate = tokenizer.decode(beam[:-1])
                candidates.append(candidate)

    return candidates



def diversity_penalize(hypotheses):


    hypotheses = hypotheses[:,:,1:]
    G,B,S = hypotheses.size()

    hypotheses_list = hypotheses.cpu().numpy().tolist()

    EOS_token = tokenizer.encode("<|endoftext|>")[0]

    group_embds = []
    hypotheses_embds = []
    scores_mask = []

    for group in hypotheses_list:


        hypotheses_embd = []
        group_embd = []
        scores_mask_beams = []

        for beam in group:

            if EOS_token in beam and beam[-1]!=EOS_token:
                scores_mask_beams.append(0.0)
            else:
                scores_mask_beams.append(1.0)

            beam_string = tokenizer.decode(beam)
            beam_string = beam_string.replace("<|endoftext|>", "")

            embd = para_encode(beam_string)
            group_embd.append(embd)
            hypotheses_embd.append(embd)

        group_embd = np.sum(np.asarray(group_embd,np.float32),axis=0)/len(group_embd)

        group_embds.append(group_embd)
        hypotheses_embds.append(hypotheses_embd)
        scores_mask.append(scores_mask_beams)

    group_embds = T.tensor(group_embds).to('cpu').float()
    hypotheses_embds = T.tensor(hypotheses_embds).to('cpu').float()
    scores_mask = T.tensor(scores_mask).to('cpu').float()

    #print(group_embds.size())
    #print(hypotheses_embds.size())

    D = group_embds.size(-1)

    group_embds = group_embds.view(1,1,G,D)
    group_embds = T.repeat_interleave(group_embds,B,dim=1)
    group_embds = T.repeat_interleave(group_embds,G,dim=0)

    hypotheses_embds = hypotheses_embds.view(G,B,1,D)
    hypotheses_embds = T.repeat_interleave(hypotheses_embds,G,dim=2)


    cosine_scores = T.sum(group_embds*hypotheses_embds,dim=-1)/(group_embds.norm(dim=-1)*hypotheses_embds.norm(dim=-1)+1e-8)

    cosine_scores = cosine_scores.view(G,B,G)

    cosine_scores = (T.sum(cosine_scores,dim=2)-1.0)/(G-1.0)

    cosine_scores = cosine_scores.view(G,B)

    return cosine_scores,scores_mask



def advanced_decoding(input_ids,past=None,
                      group=5,beam=20,
                      top_k=100,top_p=0.9,temperature=1.0):


    G = group
    B = beam
    EOS_token = tokenizer.encode("<|endoftext|>")[0]

    scores = [[0]+[-2.0**32]*(B-1)]*G
    scores = T.tensor(scores).to('cpu').view(G,B)
    scores_mask = T.ones_like(scores).to('cpu')

    hypotheses = T.zeros(G,B,1).long().to('cpu')

    i=0

    input_ids = T.repeat_interleave(input_ids,G*B,dim=0)

    with T.no_grad():

        candidates = []

        while len(candidates) < (G*B) and i<20:
            i+=1

            outputs, past_ = model(input_ids=input_ids,
                                   past=past)
            predictions = outputs

            logits = predictions[:, -1, :]
            logits = logits/temperature
            logits = logits.view(G,B,-1)

            next_candidates, log_probs = next(logits, top_k=top_k, top_p=top_p)

            hypotheses = hypotheses.unsqueeze(-2)
            hypotheses = T.repeat_interleave(hypotheses,top_k,dim=-2)
            hypotheses = hypotheses.view(G,B*top_k,-1)

            next_candidates = next_candidates.view(G,B*top_k,1)

            hypotheses = T.cat([hypotheses,next_candidates],dim=-1)

            diversity_score,scores_mask = diversity_penalize(hypotheses)

            scores_mask = scores_mask.view(G,B,top_k)
            diversity_score = diversity_score.view(G,B,top_k)

            scores = scores.unsqueeze(-1)
            scores = T.repeat_interleave(scores,top_k,dim=-1)
            scores = scores + log_probs - 0.8*diversity_score

            scores = T.where(scores_mask==1.0,
                             scores,
                             T.empty_like(scores).to('cpu').fill_(-2.0**32))


            scores = scores.view(G,B*top_k)

            _,indices = T.topk(scores,B,dim=-1)

            indices = indices.view(G,B)

            scores = T.gather(scores,index=indices,dim=-1)



            _,_,S = hypotheses.size()

            indices = indices.unsqueeze(-1)
            indices = T.repeat_interleave(indices,S,dim=-1)

            hypotheses = T.gather(hypotheses,index=indices,dim=-2)

            input_ids = hypotheses[:,:,1:].view(G*B,S-1)

            candidates += generate_candidates(hypotheses)

    """

    hypotheses = hypotheses[:,:,1:]
    hypotheses = hypotheses.view(G*B,-1)
    hypotheses_list = hypotheses.cpu().numpy().tolist()

    candidate_responses = []

    for hypothesis in hypotheses_list:
        candidate_responses.append(tokenizer.decode(hypothesis))

    print(candidate_responses)
    """

    print(candidates)




        # hypotheses = hypotheses[:,:,1:]

advanced_decoding(input_ids)


"""
def p_sample(logits,top_p=0.9):

    sorted_logits, sorted_indices = T.sort(logits, descending=True)

    cumulative_probs = T.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_logits = T.where(cumulative_probs>top_p,
                            T.empty_like(sorted_logits).to('cpu').fill_(T.tensor(-2.0**32)),
                            sorted_logits)

    probabilities = F.softmax(sorted_logits, dim=-1)
    next_word_id = sorted_indices[T.multinomial(probabilities, num_samples=1, replacement=False)]
    return next_word_id

def greedy_decoding(input_ids,past=None,
                     top_p=0.9,temperature=1.0):


    EOS_token = tokenizer.encode("<|endoftext|>")[0]
    #next_token_type = T.tensor(T.max(token_type_ids)).to('cpu').long().view(1,1)

    i=0

    while i<300:
        with T.no_grad():
            outputs, past_ = model(input_ids=input_ids,
                                   past=past)
            predictions = outputs

            logits = predictions[0, -1, :]
            logits = logits/temperature

            next_word_id = p_sample(logits, top_p=top_p)

            input_ids = T.cat([input_ids,next_word_id.view(1,-1)],dim=-1)
            #token_type_ids = T.cat([token_type_ids,next_token_type],dim=-1)

            next_word_id = next_word_id.item()

            #print(next_word_id)

            if next_word_id == EOS_token:
                break
            else:
                next_word = tokenizer.decode(next_word_id)
                print(next_word,end=" ")
                i+=1

greedy_decoding(input_ids)
"""
