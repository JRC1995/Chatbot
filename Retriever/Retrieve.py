import numpy as np
import sys
import pickle
import Utils.functions as utils
import copy
import random
import faiss

def top_candidates(candidates, scores, top=1):

    sorted_score_idx = np.flip(np.argsort(scores), axis=-1)
    candidates = [candidates[i] for i in sorted_score_idx.tolist()]
    scores = [scores[i] for i in sorted_score_idx.tolist()]

    return candidates[0:top], scores[0:top], sorted_score_idx.tolist()


def retrieve(conn, c, idx, index, query_encoding, query_context, top=5):

    #print("hello?")

    _, I =  index.search(query_encoding,k=top)

    #print(I)

    #print("hello?")

    idx = [idx[i] for i in I[0].tolist()]

    thread_idx = ()
    string_sql = 'SELECT * FROM responses WHERE '

    for i, id in enumerate(idx):
        if i == 0:
            string_sql += "parent_id = ?"
        else:
            string_sql += " or parent_id = ?"

        thread_idx += (id,)

    candidates = []

    for row in c.execute(string_sql, thread_idx):

        comment = str(row[-1])
        candidates.append(comment)

    return candidates
