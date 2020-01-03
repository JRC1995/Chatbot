from Sentence_Encoder.meta_response_encoder_fast import encode as response_encode
import Utils.functions as utils
import numpy as np
import torch as T
import copy


def random_response(candidates, conversation_history, p=None):
    loop = 5

    if p is None:
        response = random.choice(candidates)
    else:
        response = np.random.choice(candidates, p=p)

    i = 0
    while response in conversation_history:
        if p is None:
            response = random.choice(candidates)
        else:
            response = np.random.choice(candidates, p=p)
        i += 1
        if i > loop:
            break

    return response


def top_candidates(candidates, scores, top=1):
    sorted_score_idx = np.flip(np.argsort(scores), axis=-1)
    candidates = [candidates[i] for i in sorted_score_idx.tolist()]
    scores = [scores[i] for i in sorted_score_idx.tolist()]
    return candidates[0:top], scores[0:top], sorted_score_idx.tolist()


def rank_and_choose(USE_QA_model, ConvRT_model,
                    tokenizer, model_reverse,
                    utterance, query_encoding,
                    candidates,
                    response_context, conversation_history,
                    bias=None, alpha=0.4, beta=0.6):

    if bias is None:
        bias = 0.0
    #print("In Ranking")
    # print(len(candidates))

    EOS_token = tokenizer.encode("<|endoftext|>")[0]
    original_candidates = copy.deepcopy(candidates)

    response_encodings = response_encode(
        candidates, USE_QA_model, ConvRT_model, response_context*len(candidates))
    #rank_scores = np.inner(query_encoding,response_encodings)
    #rank_scores = np.reshape(rank_scores,(-1))
    rank_scores = utils.cosine_similarity_nd(query_encoding, response_encodings)

    # print(rank_scores)
    # print(rank_scores+bias)
    normed_rank_scores = utils.normalize(rank_scores+bias)

    # print(normed_rank_scores)

    # MMI Computation

    last_utterance = utterance

    def _make_feature(sents, eos):

        msg_idx = []
        for msg in sents:
            msg_idx.append(tokenizer.encode(msg))
        input_ids = [i for s in msg_idx for i in s+[eos]][:-1]

        input_ids.append(eos)

        if len(input_ids) > 300:
            input_ids = input_ids[-300:]

        return input_ids

    output_ids = _make_feature([last_utterance], EOS_token)

    with T.no_grad():
        original_output_ids = T.tensor(output_ids).to('cuda').long().unsqueeze(0)
        losses = []
        for candidate in candidates:
            input_ids = _make_feature([candidate], EOS_token)
            input_ids = T.tensor(input_ids).to('cuda').long().unsqueeze(0)
            output_ids_part_1 = T.empty_like(input_ids).to('cuda').fill_(-1).long()
            input_ids = T.cat([input_ids, original_output_ids], dim=-1)
            output_ids = T.cat([output_ids_part_1, original_output_ids], dim=-1)
            loss, _, _ = model_reverse(input_ids, past=None, labels=output_ids)
            losses.append(loss.item())

    losses = np.asarray(losses, np.float32)

    normed_MMI_scores = utils.normalize(1.0-utils.normalize(losses))

    # COMBINATION

    quasi_probabilities = alpha*(normed_rank_scores+bias) + beta*normed_MMI_scores

    candidates, quasi_probabilities, _ = top_candidates(candidates, quasi_probabilities, top=3)

    probabilities = utils.normalize(quasi_probabilities)

    response = random_response(candidates, conversation_history, p=probabilities)

    id = original_candidates.index(response)

    return response, id
