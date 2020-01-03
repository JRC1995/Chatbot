
import sys
sys.path.append("../")
from Sentence_Encoder.meta_query_encoder import encode
import tensorflow.compat.v1 as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np
import pickle
import collections


def process():
    tf.disable_eager_execution()
    sess = tf.InteractiveSession(graph=tf.Graph())

    ConvRT_model = hub.Module("../Sentence_Encoder/Embeddings/ConvRT/")

    USE_model = hub.load('../Sentence_Encoder/Embeddings/USE_QA/')

    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    with open("Processed_Scripts/Bot_Profile.pkl", "rb") as fp:
        bot_profile = pickle.load(fp)

    with open("Processed_Scripts/Chatterbot.pkl", "rb") as fp:
        chatterbot = pickle.load(fp)

    bot_queries = [k for k, v in bot_profile.items()]
    bot_contexts = ["" for k, v in bot_profile.items()]

    chatterbot_queries = [k for k, v in chatterbot.items()]
    chatterbot_contexts = ["" for k, v in chatterbot.items()]

    embedded_bot_queries = encode(sess,bot_queries,bot_contexts,USE_model, ConvRT_model)
    embedded_chatterbot_queries = encode(sess,chatterbot_queries,chatterbot_contexts,USE_model, ConvRT_model)

    with open("Processed_Scripts/embedded_bot_queries.pkl","wb") as fp:
        pickle.dump(embedded_bot_queries,fp)

    with open("Processed_Scripts/embedded_chatterbot_queries.pkl","wb") as fp:
        pickle.dump(embedded_chatterbot_queries,fp)
