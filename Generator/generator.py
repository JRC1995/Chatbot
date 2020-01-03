# Modified from: https://colab.research.google.com/drive/1-_KjlAV3J1IVDw_9KogjKDCzgFY7Jp7E#scrollTo=RZd4iXX5SqKC
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.CRITICAL)

import torch as T
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np

"""
tokenizer = GPT2Tokenizer.from_pretrained('DialoGPT/Configs/')
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
model.to('cuda')
model.eval()

model_reverse.load_state_dict(weights_reverse)
model_reverse.to('cuda')
model_reverse.eval()

conv_history = ["what is the meaning of life?"]
"""



def generate(conv_history, model, tokenizer, past=None, samples=20):

    EOS_token = tokenizer.encode("<|endoftext|>")[0]

    # adapted from: https://github.com/microsoft/DialoGPT/blob/master/prepro.py
    def _make_feature(sents, eos):

        msg_idx = []
        for msg in sents:
            msg_idx.append(tokenizer.encode(msg))
        input_ids = [i for s in msg_idx for i in s+[eos]][:-1]

        input_ids.append(eos)

        if len(input_ids) > 200:
            input_ids = input_ids[-200:]

        return input_ids


    input_ids = _make_feature(conv_history, EOS_token)

    input_ids = T.tensor(input_ids).long().to('cuda').unsqueeze(0)

    #print(input_ids.size())

    def generate_candidates(hypotheses):

        EOS_token = tokenizer.encode("<|endoftext|>")[0]

        hypotheses_list = hypotheses.cpu().numpy().tolist()

        candidates = []
        for beam in hypotheses_list:
            # print(beam)
            if beam[-1] == EOS_token and EOS_token not in beam[:-1]:
                candidate = tokenizer.decode(beam[:-1])
                candidates.append(candidate)

        return candidates

    def p_sample(logits, top_p=0.9):

        sorted_logits, sorted_indices = T.sort(logits, dim=-1, descending=True)

        cumulative_probs = T.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        min_probs,_ = T.min(cumulative_probs,dim=-1)
        min_probs = min_probs.view(-1,1)
        min_probs = T.repeat_interleave(min_probs,logits.size(-1),dim=-1)

        sorted_logits_ = sorted_logits.clone()

        sorted_logits = T.where(cumulative_probs > top_p,
                                T.empty_like(sorted_logits).to('cuda').fill_(T.tensor(-2.0**32)),
                                sorted_logits_)

        sorted_logits = T.where(min_probs > top_p,
                                sorted_logits_,
                                sorted_logits)

        probabilities = F.softmax(sorted_logits, dim=-1)

        next_word_sorted_idx = T.multinomial(probabilities, num_samples=1, replacement=False)

        next_word_sorted_idx = next_word_sorted_idx.view(-1, 1)

        next_word_idx = T.gather(sorted_indices, dim=-1, index=next_word_sorted_idx)
        log_probs = T.gather(probabilities, dim=-1, index=next_word_sorted_idx)
        log_probs = T.log(log_probs+1e-8)

        log_probs = log_probs.view(-1)

        return next_word_idx, log_probs

    def greedy_decoding(input_ids, samples, past=None,
                        top_p=0.9, temperature=1):

        EOS_token = tokenizer.encode("<|endoftext|>")[0]
        #next_token_type = T.tensor(T.max(token_type_ids)).to('cuda').long().view(1,1)

        i = 0

        input_ids = T.repeat_interleave(input_ids, samples, dim=0)
        _, inp_size = input_ids.size()

        candidates = []

        response_ids = input_ids.clone()

        total_log_probs = T.zeros(samples).to('cuda')
        mask = T.ones(samples).to('cuda')

        while len(candidates) < samples and i < 300:

            with T.no_grad():
                outputs, past = model(input_ids=input_ids,
                                      past=past)
                predictions = outputs

                logits = predictions[:, -1, :]
                logits = logits/temperature

                next_word_idx, log_probs = p_sample(logits, top_p=top_p)

                total_log_probs = total_log_probs+(log_probs*mask)

                mask = T.where(next_word_idx.view(-1) == EOS_token,
                               T.zeros(samples).to('cuda'),
                               mask)

                input_ids = next_word_idx  # T.cat([input_ids,next_word_idx],dim=-1)
                #token_type_ids = T.cat([token_type_ids,next_token_type],dim=-1)

                response_ids = T.cat([response_ids, next_word_idx], dim=-1)

                candidates += generate_candidates(response_ids[:, inp_size:])

                i += 1

        return candidates, total_log_probs

    candidates, scores = greedy_decoding(input_ids, samples)

    return candidates, scores


"""
candidates,_ = generate(conv_history,model,tokenizer)

print(candidates)
"""
