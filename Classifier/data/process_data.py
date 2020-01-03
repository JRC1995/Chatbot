import pickle
import sys
sys.path.append("../../")  # nopep8
from Sentence_Encoder.meta_query_encoder import encode
import tensorflow.compat.v1 as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np

tf.disable_eager_execution()
sess = tf.InteractiveSession(graph=tf.Graph())

ConvRT_model = hub.Module("../../Sentence_Encoder/Embeddings/ConvRT/")
USE_QA_model = hub.load('../../Sentence_Encoder/Embeddings/USE_QA/')

sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())

train_dir = "train.txt"
dev_dir = "dev.txt"

filename = train_dir
all_targets = []


def extract_data(filename):

    contexts = []
    queries = []
    acts = []

    with open(filename) as file:
        global all_targets
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            split_line = line.split(" : ")
            line = split_line[1]
            context1 = split_line[0]

            if "what" in context1.lower() \
                    or "why" in context1.lower() \
                    or "where" in context1.lower() \
                    or "how" in context1.lower() \
                    or "who" in context1.lower():
                punc = "?"
            else:
                punc = "."

            split_line = line.split(" > ")

            context2 = split_line[0].strip()
            if context2 == "EMPTY":
                context = context1+punc
            else:
                context = context1+punc+" "+context2

            line = split_line[1]
            split_line = line.split(" ## ")
            current_uttr = split_line[0]
            targets = split_line[1]
            targets = targets.split(";")
            targets = [target for target in targets if target != '']

            if len(targets) < 2:
                targets.append("NULL")

            all_targets += targets

            contexts.append(context)
            queries.append(current_uttr)
            acts.append(targets)

    return contexts, queries, acts


train_contexts, train_queries, train_acts = extract_data(train_dir)
test_contexts, test_queries, test_acts = extract_data(dev_dir)

all_targets = list(set(all_targets))

labels2idx = {v: i for i, v in enumerate(all_targets)}

train_queries_vec = []
i = 0
batch_size = 2000
while i < len(train_queries):
    print(i)
    if i+batch_size > len(train_queries):
        batch_size = len(train_queries)-i

    train_query_vec = encode(sess, train_queries[i:i+batch_size], train_contexts[i:i +
                                                                                 batch_size], USE_QA_model, ConvRT_model)
    train_queries_vec.append(train_query_vec)
    i += batch_size

train_queries_vec = np.concatenate(train_queries_vec, axis=0)

test_queries_vec = []
i = 0
while i < len(test_queries):
    if i+batch_size > len(test_queries):
        batch_size = len(test_queries)-i
    test_query_vec = encode(sess, test_queries[i:i+batch_size], test_contexts[i:i +
                                                                              batch_size], USE_QA_model, ConvRT_model)
    test_queries_vec.append(test_query_vec)
    i += batch_size
test_queries_vec = np.concatenate(test_queries_vec, axis=0)

print(train_queries_vec.shape)
print(test_queries_vec.shape)

train_acts_vec = []
for acts in train_acts:
    train_acts_vec.append([labels2idx[act] for act in acts])

test_acts_vec = []
for acts in test_acts:
    test_acts_vec.append([labels2idx[act] for act in acts])

train_acts_vec = np.asarray(train_acts_vec, np.int)
test_acts_vec = np.asarray(test_acts_vec, np.int)

print(train_acts_vec.shape)
print(test_acts_vec.shape)

data = {}

data["labels2idx"] = labels2idx

data["train_contexts"] = train_contexts
data["test_contexts"] = test_contexts

data["train_queries"] = train_queries
data["train_acts"] = train_acts

data["test_queries"] = test_queries
data["test_acts"] = test_acts

data["test_queries_vec"] = test_queries_vec
data["test_acts_vec"] = test_acts_vec

data["train_queries_vec"] = train_queries_vec
data["train_acts_vec"] = train_acts_vec

with open("processed_data.pkl", 'wb') as fp:
    pickle.dump(data, fp)
