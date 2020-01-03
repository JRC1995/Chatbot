import sys
import numpy as np
import math
import pickle
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.CRITICAL)
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
sys.path.append("../")
import Utils.functions as utils
"""
USE_QA_model = hub.load('Embeddings/USE_QA/')
ConvRT_model = encoder_client.EncoderClient("Embeddings/ConvRT",use_extra_context=True)
"""


def encode(texts, USE_QA_model, ConvRT_model, contexts=None):

    texts = [utils.simple_preprocess(text) for text in texts]

    if contexts is not None:

        new_contexts = []

        for context in contexts:
            new_context = [utils.simple_preprocess(text) for text in context]
            new_contexts.append(new_context)
        contexts = new_contexts

    context_encodings = ConvRT_model.encode_contexts(texts, extra_contexts=contexts)
    question_embeddings = USE_QA_model.signatures['question_encoder'](tf.constant(texts))["outputs"]

    return np.concatenate([np.asarray(question_embeddings), np.asarray(context_encodings)], axis=-1)


"""
texts = ["Hello boy!","How are you doing bro?","May be you should go die"]

for i in range(1000):
    print(i)
    print(encode(texts,USE_QA_model,ConvRT_model,contexts=[["x","y","z"],[],[]]).shape)
"""
