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
    context_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    context_encoder = ConvRT_model({
        'context': text_placeholder,
        'extra_context': context_placeholder,
    },
        signature="encode_context",)

    ConvRT_context_embeddings = np.asarray(
        sess.run(context_encoder, feed_dict={text_placeholder: texts, context_placeholder: contexts}))

    USE_QA_encoder = USE_QA_model.signatures['question_encoder'](tf.constant(texts))["outputs"]

    USE_QA = np.asarray(sess.run(USE_QA_encoder), np.float32)

    meta_query_embd = np.concatenate([USE_QA, ConvRT_context_embeddings], axis=-1)

    return meta_query_embd
