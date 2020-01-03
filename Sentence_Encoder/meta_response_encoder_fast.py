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
    contexts = [utils.simple_preprocess(text) for text in contexts]

    ConvRT_encodings = ConvRT_model.encode_responses(texts)
    USE_embeddings = USE_QA_model.signatures['response_encoder'](input=tf.constant(texts),
                                                                 context=tf.constant(contexts))["outputs"]

    return np.concatenate([np.asarray(USE_embeddings), np.asarray(ConvRT_encodings)], axis=-1)


"""
texts = ["xyz","abc"]
contexts = ["sdfdf","sssd"]

for i in range(1000):
    print(i)
    print(encode(texts,USE_QA_model,ConvRT_model,contexts=contexts))
"""
