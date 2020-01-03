from TTS.text2speech import tts_class
from multiprocessing import Process
import faiss
import time
import sqlite3
import csv
import random
import copy
import tensorflow_hub as hub
import tensorflow_text
import math
import numpy as np
import pickle
from Retriever.Retrieve import retrieve
import Utils.functions as utils
from ReRanker.rerank import rank_and_choose
from Generator.generator import generate as DialoGPT_Generate
from Classifier.model.dialog_acts import Encoder as Classifier
from Sentence_Encoder.meta_response_encoder_fast import encode as response_encode
from Sentence_Encoder.meta_query_encoder_fast import encode as query_encode
import Sentence_Encoder.encoder_client as encoder_client
import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.nn.functional as F
import torch.nn as nn
import torch as T
import os
import sys
import argparse
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.CRITICAL)


parser = argparse.ArgumentParser(description="Chatbot")
parser.add_argument('--voice', dest='voice', action='store_true')
parser.add_argument('--no-voice', dest='voice', action='store_false')
parser.set_defaults(voice=True)
flags = parser.parse_args()

device = "cuda"

# LOAD DATABASE
with open("Retriever/Faiss_index/thread_idx.pkl", 'rb') as fp:
    idx = pickle.load(fp)
index = faiss.read_index('Retriever/Faiss_index/large.index')
# LOAD DATABASE
conn = sqlite3.connect('Retriever/Database/reddit.db')
c = conn.cursor()


# LOAD SCRIPTS

with open('Scripted/Processed_Scripts/Bot_Profile.pkl', 'rb') as fp:
    bot_profile = pickle.load(fp)

bot_queries = [k for k, v in bot_profile.items()]

with open('Scripted/Processed_Scripts/Chatterbot.pkl', 'rb') as fp:
    chatterbot = pickle.load(fp)

chatterbot_queries = [k for k, v in chatterbot.items()]

# LOAD SCRIPT EMBEDDINGS

with open('Scripted/Processed_Scripts/embedded_bot_queries.pkl', 'rb') as fp:
    bot_queries_embd = pickle.load(fp)

with open('Scripted/Processed_Scripts/embedded_chatterbot_queries.pkl', 'rb') as fp:
    chatterbot_queries_embd = pickle.load(fp)
# Load Dialog Acts Classifer
with open("Classifier/data/processed_data.pkl", "rb") as fp:
    data = pickle.load(fp)

labels2idx = data["labels2idx"]
idx2labels = {v: k for k, v in labels2idx.items()}


# Load TTS model
with T.no_grad():
    text2speech = tts_class()

with T.no_grad():
    dialog_act_classifier = Classifier(
        D=bot_queries_embd.shape[-1], classes_num=len(labels2idx)).cuda()
    checkpoint = T.load("Classifier/Model_Backup/model.pt")
    dialog_act_classifier.load_state_dict(checkpoint['model_state_dict'])
    dialog_act_classifier = dialog_act_classifier.eval()

# LOAD DialoGPT Generator

with T.no_grad():
    tokenizer = GPT2Tokenizer.from_pretrained('Generator/DialoGPT/Configs/')
    weights = T.load('Generator/DialoGPT/Parameters/medium_ft.pkl')
    weights_reverse = T.load('Generator/DialoGPT/Parameters/small_reverse.pkl')
    cfg = GPT2Config.from_json_file('Generator/DialoGPT/Configs/config.json')
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


with tf.device("/cpu:0"):
    # sess = tf.InteractiveSession(graph=tf.Graph())

    # LOAD STUFF

    # LOAD SENTENCE ENCODERS

    # Hub Models

    ConvRT_model = encoder_client.EncoderClient(
        "Sentence_Encoder/Embeddings/ConvRT", use_extra_context=True)
    USE_QA_model = hub.load("Sentence_Encoder/Embeddings/USE_QA/")

    # %%

    command_codes = ["<PASS>", "<JOKE>", "<GENERATE>",
                     "<INITIATE>", "<TIL>", "<STORY>", "<SHOWER>", "<STOP>"]

    code_map = {"<INITIATE>": ["Scripted/Random_Reddit_Data/nostupidq.csv",
                               "Scripted/Random_Reddit_Data/jokesq.csv",
                               "Scripted/Random_Reddit_Data/showerthoughtsq.csv",
                               "Scripted/Random_Reddit_Data/tilq.csv"],
                "<TIL>": ["Scripted/Random_Reddit_Data/tilq.csv"],
                "<SHOWER>": ["Scripted/Random_Reddit_Data/showerthoughtsq.csv"],
                "<STORY>": ["Scripted/Random_Reddit_Data/writingpromptsa.csv"],
                "<JOKE>": ["Scripted/Random_Reddit_Data/jokesq.csv"]}

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

    # %%

    def load_random_reddit(directory, conversation_history):

        candidates = []

        with open(directory, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for i, row in enumerate(csv_reader):
                if 'writing' in directory:
                    parent_id = str(row['parent_id'])[3:]
                    thread_id = str(row['link_id'])[3:]
                    if parent_id == thread_id:
                        candidate = str(row["body"])
                else:
                    candidate = str(row["title"])
                    if 'joke' in directory:
                        candidate += ".... "+str(row['selftext'])
                candidates.append(candidate)

        return random_response(candidates, conversation_history)

    # extract top candidates (queries or responses)

    def top_candidates(candidates, scores, top=1):
        sorted_score_idx = np.flip(np.argsort(scores), axis=-1)
        candidates = [candidates[i] for i in sorted_score_idx.tolist()]
        scores = [scores[i] for i in sorted_score_idx.tolist()]
        return candidates[0:top], scores[0:top], sorted_score_idx.tolist()

    # %%

    def generate(texts, past):
        candidates, _ = DialoGPT_Generate(texts, model, tokenizer)
        return candidates, past

    # START DOING STUFF

    conversation_history = []
    past = None
    stop_flag = 0
    print("\n")

    while True:

        utterance = input("Say Something: ")  # ,hello how are ya today"

        response_code = ""
        retrieved_candidates = []

        utils.delay_print("\nThinking......")

        candidates = []
        temp_candidates = []
        temp_scores = []
        if not conversation_history:
            query_context = []
            response_context = [""]
        else:
            if len(conversation_history) > 5:
                truncated_history = copy.deepcopy(conversation_history[-5:])
            else:
                truncated_history = copy.deepcopy(conversation_history)

            response_context = [conversation_history[-1]]

            # ConveRT needs reversed Context, not sure about USE QA but assuming it's not reverse

            query_context = [stuff for stuff in truncated_history]

        query_encoding = query_encode([utterance], USE_QA_model, ConvRT_model, [query_context])

        if conversation_history:
            if len(conversation_history) > 5:
                truncated_history = conversation_history[-5:]
            else:
                truncated_history = conversation_history
            generated_responses, past = generate(truncated_history+[utterance], past)
        else:
            generated_responses, past = generate([utterance], past)

        bot_cosine_scores = utils.cosine_similarity_nd(query_encoding, bot_queries_embd)
        bot_queries_, bot_cosine_scores_, _ = top_candidates(bot_queries, bot_cosine_scores, top=1)

        active_codes = []

        bot_candidates = bot_profile[bot_queries_[0]]

        filtered_bot_candidates = []
        for candidate in bot_candidates:
            flag = 0
            for code in command_codes:
                if code in candidate:
                    active_codes.append(code)
                    candidate = candidate.replace(code, "")
                    filtered_bot_candidates.append(candidate)
                    flag = 1
                    break

            if flag == 0:
                candidates.append(candidate)
                filtered_bot_candidates.append(candidate)
                active_codes.append("")

        with T.no_grad():
            logits = dialog_act_classifier(T.tensor(query_encoding).to(device))
            _, sorted_idx = T.sort(logits, dim=-1, descending=True)
            sorted_idx = sorted_idx.squeeze(0)
            sorted_idx = sorted_idx[0:2].cpu().tolist()

        labels = [idx2labels[i] for i in sorted_idx]

        print("\nClassified Dialog Acts: {}\n".format(", ".join(labels)))

        # print(labels)

        """
        Possible Dialog Acts:
        ['nonsense', 'dev_command', 'open_question_factual', 'appreciation', 'other_answers', 'statement', \
        'respond_to_apology', 'pos_answer', 'closing', 'comment', 'neg_answer', 'yes_no_question', 'command', \
        'hold', 'NULL', 'back-channeling', 'abandon', 'opening', 'other', 'complaint', 'opinion', 'apology', \
        'thanking', 'open_question_opinion']
        """

        if bot_cosine_scores_[0] >= 0.75:
            response, id = rank_and_choose(USE_QA_model, ConvRT_model,
                                           tokenizer,
                                           model_reverse,
                                           utterance,
                                           query_encoding,
                                           filtered_bot_candidates,
                                           response_context,
                                           conversation_history)
            code = active_codes[id]
            if code in code_map:
                response_code = "(Reddit JOKE/WRITING/TIL ETC.)"
                directories = code_map[code]
                directory = random.choice(directories)
                response += " "+load_random_reddit(directory, conversation_history)

            elif code == "<GENERATE>":
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              generated_responses,
                                              response_context,
                                              conversation_history)
            elif code == "<STOP>":
                stop_flag = 1

        elif stop_flag != 1:
            mode = "DEFAULT"
            bias = None

            if 'open_question_factual' in labels \
               or ('yes_no_question' in labels and 'NULL' not in labels) \
               or 'open_question_opinion' in labels or 'command' in labels:
                bias = 0.07  # biases towards retrieval

            elif "apology" in labels:
                mode = "BREAK"
                candidates = ["Apology accepted.", "No need to apologize.",
                              "No worries.", "You are forgiven"]
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              candidates,
                                              response_context,
                                              conversation_history)

            elif "abandon" in labels or "nonsense" in labels:

                mode = np.random.choice(["BREAK", "INITIATE"], p=[0.6, 0.4])

                if mode == "BREAK":
                    candidates = ["what?", "Can you rephrase what you mean?",
                                  "What do you mean exactly?"]
                    response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                                  tokenizer,
                                                  model_reverse,
                                                  utterance,
                                                  query_encoding,
                                                  generated_responses+candidates,
                                                  response_context,
                                                  conversation_history)
                else:
                    directories = code_map['<INITIATE>']
                    directory = random.choice(directories)
                    response = load_random_reddit(directory, conversation_history)

            elif 'hold' in labels:
                mode = "BREAK"
                candidates = ["Do you want to add something more?",
                              "I think you want to say something more."]
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              generated_responses+candidates,
                                              response_context,
                                              conversation_history)

            elif 'closing' in labels:
                mode = "BREAK"
                candidates = ["Nice talking to you.", "Goodbye.", "See you later."]
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              candidates,
                                              response_context,
                                              conversation_history)
                stop_flag = 1

            elif 'opening' in labels:
                mode = "BREAK"
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              generated_responses,
                                              response_context,
                                              conversation_history)
                stop_flag = 1

            elif 'thanking' in labels:
                mode = np.random.choice(["BREAK", "INITIATE"], p=[0.6, 0.4])

                if mode == "BREAK":
                    candidates = ["No need to mention", "You are welcome."]
                    response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                                  tokenizer,
                                                  model_reverse,
                                                  utterance,
                                                  query_encoding,
                                                  generated_responses+candidates,
                                                  response_context,
                                                  conversation_history)
                else:
                    directories = code_map['<INITIATE>']
                    directory = random.choice(directories)
                    response = load_random_reddit(directory, conversation_history)

            elif 'apology' in labels:
                mode = "BREAK"
                candidates = ["Apology accepted.", "Apology granted",
                              "No Worries!", "No need to apologize."]
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              generated_responses+candidates,
                                              response_context,
                                              conversation_history)

            elif 'response_to_apology' in labels\
                 or 'pos_answer' in labels or 'neg_answer' in labels\
                 or 'appreciation' in labels or 'back_channeling' in labels:

                mode = np.random.choice(["BREAK", "INITIATE"], p=[0.6, 0.4])

                if mode == "BREAK":
                    response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                                  tokenizer,
                                                  model_reverse,
                                                  utterance,
                                                  query_encoding,
                                                  generated_responses,
                                                  response_context,
                                                  conversation_history)
                else:
                    directories = code_map['<INITIATE>']
                    directory = random.choice(directories)
                    response = load_random_reddit(directory, conversation_history)

            if mode != "BREAK":

                chatterbot_cosine_scores = utils.cosine_similarity_nd(
                    query_encoding, chatterbot_queries_embd)
                chatterbot_queries_, chatterbot_cosine_scores_, _ = top_candidates(
                    chatterbot_queries, chatterbot_cosine_scores, top=1)
                chatterbot_candidates = chatterbot[chatterbot_queries_[0]]
                candidates += chatterbot_candidates

                retrieved_candidates = retrieve(
                    conn, c, idx, index, query_encoding, query_context)

                if bias is not None:
                    biases = [0.0 for _ in candidates]
                    for _ in generated_responses:
                        biases.append(0.0)
                    for _ in retrieved_candidates:
                        biases.append(bias)
                    biases = np.asarray(biases, np.float32)
                else:
                    biases = None

                candidates += generated_responses + retrieved_candidates
                response, _ = rank_and_choose(USE_QA_model, ConvRT_model,
                                              tokenizer,
                                              model_reverse,
                                              utterance,
                                              query_encoding,
                                              candidates,
                                              response_context,
                                              conversation_history,
                                              bias=biases)
        if response_code == "":
            if response in generated_responses:
                response_code = "(GENERATED)"
            elif response in retrieved_candidates:
                response_code = "(RETRIEVED)"
            elif response in filtered_bot_candidates:
                response_code = "(FROM SCRIPT)"
            elif response in chatterbot_candidates:
                response_code = "(FROM CHATTERBOT SCRIPT)"
            else:
                response_code = "(I DON'T KNOW WHERE IT IS FROM)"

        print("\n")
        if len(str(response).split(" ")) <= 100:

            if flags.voice:

                entry = utils.simple_preprocess(str(response).lower(),
                                                for_speech=True,
                                                return_tokenized=True)
                entry = " ".join(entry)

                wavefiles = text2speech.process(entry)

                def f1():
                    utils.delay_print("Bot: "+response)

                def f2():
                    text2speech.play(wavefiles)

                p1 = Process(target=f1)
                p2 = Process(target=f2)
                p1.start()
                p2.start()
                p1.join()
                p2.join()
            else:
                utils.delay_print("Bot: "+response)

        else:
            utils.delay_print("Bot: "+response, t=0.01)

        print("\n")

        conversation_history.append(utterance)
        conversation_history.append(response)

        if stop_flag == 1:
            break

        # break
