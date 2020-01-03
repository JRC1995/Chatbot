import sys
import numpy as np
import math
import pickle
import os
import tensorflow.compat.v1 as tf
import tensorflow_text
import tensorflow_hub as hub
sys.path.append("../")
import Utils.functions_old as utils

def encode(sess, texts, contexts, USE_QA_model, ConvRT_model):

    new_texts = []

    for text in texts:

        text, tokenized_text = utils.simple_preprocess(text)
        new_texts.append(text)

    texts = new_texts

    new_contexts = []

    for context in contexts:

        context, tokenized_context = utils.simple_preprocess(context)
        new_contexts.append(context)

    contexts = new_contexts

    text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    response_encoder = ConvRT_model(text_placeholder,signature="encode_response")

    ConvRT_response_embeddings = np.asarray(sess.run(response_encoder, feed_dict={text_placeholder: texts}))

    USE_QA_encoder = USE_QA_model.signatures['response_encoder'](input=tf.constant(texts),
                                                                context=tf.constant(contexts))["outputs"]
    USE_QA =  np.asarray(sess.run(USE_QA_encoder),np.float32)

    meta_response_embd = np.concatenate([USE_QA,ConvRT_response_embeddings],axis=-1)

    return meta_response_embd

