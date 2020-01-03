import sqlite3
import csv
import pickle
import numpy as np
import sys
sys.path.append("../")
from Sentence_Encoder.meta_query_encoder import encode
import tensorflow.compat.v1 as tf
import tensorflow_text
import tensorflow_hub as hub

tf.disable_eager_execution()
sess = tf.InteractiveSession(graph=tf.Graph())

ConvRT_model = hub.Module("../Sentence_Encoder/Embeddings/ConvRT/")
USE_QA_model = hub.load('../Sentence_Encoder/Embeddings/USE_QA/')

sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())

conn = sqlite3.connect('Database/reddit.db')

conn.execute('''CREATE TABLE queries (id text, title text, embedding BLOB)''')
conn.execute('''CREATE TABLE responses (thread_id text, parent_id text, comment text)''')


def save_queries(queries, query_idx):
    global conn
    global sess

    global ConvRT_model
    global USE_QA_model

    rows = []
    contexts = ["" for _ in queries]
    embeddings = encode(sess, queries, contexts, USE_QA_model, ConvRT_model)
    embeddings = embeddings.tolist()
    for query, query_id, embedding in zip(queries, query_idx, embeddings):
        embedding = np.asarray(embedding, np.float32)
        rows.append((query_id, query, embedding))
    conn.executemany('INSERT INTO queries VALUES (?,?,?)', rows)


def save_responses(comments, parent_idx, thread_idx):
    global conn
    rows = []
    for thread_id, parent_id, comment in zip(thread_idx, parent_idx, comments):
        rows.append((thread_id, parent_id, comment))
    conn.executemany('INSERT INTO responses VALUES (?,?,?)', rows)


filepaths_q = [
    'Data/adviceq.csv',
    'Data/askphilosophyq.csv',
    'Data/askredditq.csv',
    'Data/mlq.csv',
    'Data/casualq.csv',
    'Data/eli5q.csv',
    'Data/askscienceq.csv']

filepaths_a = [
    'Data/advicea.csv',
    'Data/askphilosophya.csv',
    'Data/askreddita.csv',
    'Data/mla.csv',
    'Data/casuala.csv',
    'Data/eli5a.csv',
    'Data/asksciencea.csv']


for filename_q, filename_a in zip(filepaths_q, filepaths_a):

    queries = []
    responses = []
    query_idx = []
    response_thread_idx = []
    response_parent_idx = []

    print("\n\n Processing {} and {}....\n\n".format(filename_q, filename_a))

    comment_thread_idx = {}  # choose thread to store as query only if id in here
    thread_idx = {}  # choose comment as response if parent_id in here

    with open(filename_a, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            id = str(row['id'])
            thread_id = str(row['link_id'])[3:]
            parent_id = str(row['parent_id'])[3:]
            comment = str(row['body'])

            if len(comment.split(" ")) <= 300:

                if parent_id == thread_id:
                    if thread_id not in comment_thread_idx:
                        comment_thread_idx[thread_id] = 1


    with open(filename_q, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            id = str(row['id'])
            title = str(row['title'])

            if len(title.split(" ")) <= 200:
                if id not in thread_idx:
                    thread_idx[id] = 1

    print("\n\nProcessing Queries\n\n")

    with open(filename_q, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            id = str(row['id'])
            title = str(row['title'])

            if len(title.split(" ")) <= 200:
                if id in comment_thread_idx:
                    queries.append(title)
                    query_idx.append(id)
                    if len(queries) > 500:
                        print(i)
                        save_queries(queries, query_idx)
                        del queries
                        del query_idx
                        queries = []
                        query_idx = []

    if queries:
        save_queries(queries, query_idx)
        del queries
        del query_idx


    conn.commit()

    print("\n\nProcessing Responses\n\n")

    with open(filename_a, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):

            id = str(row['id'])
            comment = str(row['body'])
            thread_id = str(row['link_id'])[3:]
            parent_id = str(row['parent_id'])[3:]

            if len(comment.split(" ")) <= 300:
                if parent_id == thread_id:
                    if parent_id in thread_idx:
                        responses.append(comment)
                        response_parent_idx.append(parent_id)
                        response_thread_idx.append(thread_id)
                        if len(responses) > 1000:
                            print(i)
                            save_responses(responses, response_parent_idx, response_thread_idx)
                            del responses
                            del response_parent_idx
                            del response_thread_idx
                            responses = []
                            response_parent_idx = []
                            response_thread_idx = []

        if responses:
            save_responses(responses, response_parent_idx, response_thread_idx)
            del responses
            del response_parent_idx
            del response_thread_idx


    conn.commit()


conn.close()
