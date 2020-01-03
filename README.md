# Conversational-AI

Hybrid Conversational Bot based on both neural retrieval and neural generative mechanism with TTS. 

See Project Report [here](https://github.com/JRC1995/Chatbot/blob/master/Project%20Report.pdf). Note, however, that the report refers to an older version of the project which was much slower, which didn't use FAISS or TTS. 

## Credits:

* The [TTS Folder](https://github.com/JRC1995/Chatbot/tree/master/TTS) is based on an earlier version of [Mozilla TTS](https://github.com/mozilla/TTS). I prepared the [text2speech.py](https://github.com/JRC1995/Chatbot/blob/master/TTS/text2speech.py) file which acts as an interface to the rest of the project.
* The [encoder_client](https://github.com/JRC1995/Chatbot/blob/master/Sentence_Encoder/encoder_client.py) file in Sentence_Encoder is taken from [Poly-AI](https://github.com/PolyAI-LDN/polyai-models/blob/master/encoder_client.py). I modified it slightly to run it with Tensorflow 2. We also use their pre-trained models. 
* [DialoGPT (inside Generator)](https://github.com/JRC1995/Chatbot/tree/master/Generator/DialoGPT/Configs) config files and vocab are taken from the [DialoGPT repository](https://github.com/microsoft/DialoGPT). We also use their pre-trained models.
* The [data here in Classifier](https://github.com/JRC1995/Chatbot/tree/master/Classifier/data) is from [MIDAS_dialog_act repository](https://github.com/DianDYu/MIDAS_dialog_act)
* The corpus [here](https://github.com/JRC1995/Chatbot/tree/master/Scripted/Chatterbot_Corpus) is taken from [Chatterbot_Corpus library](https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/english)
* [Mobashir Sadat](https://www.linkedin.com/in/mobashir-sadat-2b32a3112/) prepared and processed the Chatterbot Corpus and created the diagrams. 
* [Faiss](https://github.com/facebookresearch/faiss), [Huggingface's Transformers](https://github.com/huggingface/transformers), and [Tensorflow Hub](https://www.tensorflow.org/hub/overview) form important components of the project. We use pre-trained models of [Universal Sentence Encoder QA](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3) through Tensorflow Hub.

## How conversation looks like:

(Most of them are retrieved from Reddit Database)
(See the project report for generative examples)

```
Say Something: Hello, how are you?

Thinking......

Bot: I am doing well.

Say Something: What is your name?

Thinking......

Bot: My name is Ray Solomonon

Say Something: Do you like movies? If so, what kind?

Thinking......

Bot: I love a good demon possession movie. I love a good social thriller. I love science fiction. Really, I'm a big fan of all sorts of movies in general. But my favourites are movies like A Clockwork Orange, Moon, The Conjuring, Paranormal Activity 3 (specifically 3 because I think they did some really interesting/new things for the genre as a whole), etc.

Say Something: Are you sentient?

Thinking......

Bot: Probably no. What about you? Are YOU conscious or you just a P-Zombie?

Say Something: Tell me a Joke.

Thinking......

Bot:  This girl said she knew me from the vegetarian restaurant....... But I've never met herbivore!

Say Something: What can you tell me about Quantum Field Theory?

Thinking......

Bot: Quantum field theory put simply is relativistic quantum mechanics where you can create and destroy particles. It's not "*the*" quantum field theory; there are lots of quantum field theories. For example, quantum electrodynamics describes electromagnetic interactions.

Say Something: Do you know anything about Sellar's myth of the given?

Thinking......

Bot: An important clarificatory note: the pairing of this question with this passage seems to imply that Sellars is suggesting that the Myth of the Given isn't really a myth. But he's actually doing the opposite: he's suggesting that his myth (we might call it the Myth of Genius Jones), which is supposed to *supplant* the Myth of the Given, is not really a myth. So I guess there are really two questions implicit in this post:

(1) Is the Myth of the Given really just a myth?
(2) Is the Myth of Genius Jones, which is supposed to provide an alternative to the Myth of the Given, really just a myth?

Sellars's view is that the answer to (1) is "yes" and that the answer to (2) is "no" (though, of course, he doesn't mean to endorse the Myth of Genius Jones taken *too* literally, i.e. he doesn't think the theory of thoughts was literally devised by one guy all at once).

Since you ask for thoughts, I'll add that my own view is that Sellars is right on both of these questions. The Myth of the Given is a Myth, and the story of Genius Jones is likely an apt metaphor for how we came to be aware of our own internal episodes.


Say Something: I am quite tired today; I didn’t know that the project deadline was so soon. I had a sleepless night working on the project.

Thinking......

Bot: Honestly i'm pretty irresponsible when it comes to projects. I always try to finish them the last second which is really bad. I have a set sleep time that i refuse to mess up. So i'm going to try and finish work before it's due so i don't start having panic attacks.

As for you just talk to your teacher. They've probably been in the same shoes as you: they'll understand.

Say Something: Did you know I accidentally deleted major portions of the project and had to reconstruct it again? Version control is important huh. Who knew?

Thinking......

Bot: Reminds me of the time I was working on a web development project in uni, the night before due date I kinda blew up the php code, and couldn't figure out what I broke.

Remember kids, source control is important!

Say Something: Just my luck, I have three project presentations in the same day.

Thinking......

Bot: Make sure you plan out studying this week and create a schedule for yourself. However, don't forget to keep sleep in the schedule!

Trust me, I was in college for 4.5 years and the one thing I learned was that all-nighters are *not* what you need. I failed calc the first time because I made so many stupid mistakes on the final that I would have at least been able to *think* about had I studied just a little less and slept just a little more. You need to find the right balance.

Say Something: By the way, can you tell me about Language Models?

Thinking......

Bot: On 1B dataset, [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) has 30 (23.7 with ensembles)


Say Something: Please tell me more about them.

Thinking......

Bot: Learn about **GRAPHENE**

https://www.youtube.com/watch?v=Mcg9_ML2mXY&amp;t=2s

it will be the solution.  it is the solution.



```



## Requirements:

Here's my explorted miniconda environment: https://github.com/JRC1995/Chatbot/blob/master/chatbot.yml

## Original Directory

This shows how the directory should be set up. It shows all the files and folders unfiltered along with all the big model weights that were not uploaded to the repository:

```
├── Classifier
│   ├── data
│   │   ├── dev.txt
│   │   ├── inference.txt
│   │   ├── process_data.py
│   │   ├── processed_data.pkl
│   │   └── train.txt
│   ├── DataLoader
│   │   └── bucket_and_batch.py
│   ├── model
│   │   ├── dialog_acts.py
│   │   └── __init__.py
│   ├── Model_Backup
│   │   └── model.pt
│   └── train_and_test
│       └── train.py
├── evaluate.py
├── Generator
│   ├── DialoGPT
│   │   ├── Configs
│   │   │   ├── config.json
│   │   │   ├── merges.txt
│   │   │   └── vocab.json
│   │   └── Parameters
│   │       ├── medium_ft.pkl
│   │       └── small_reverse.pkl
│   ├── Experimental Codes
│   │   ├── test_advanced_experimental_2.py
│   │   └── test_advanced_experimental.py
│   ├── generator.py
│   └── __init__.py
├── __init__.py
├── interact_generator_only.py
├── interact.py
├── interact_retrieval_only.py
├── interact_verbose.py
├── Readme.txt
├── ReRanker
│   └── rerank.py
├── Retriever
│   ├── Data
│   │   ├── advicea.csv
│   │   ├── adviceq.csv
│   │   ├── askphilosophya.csv
│   │   ├── askphilosophyq.csv
│   │   ├── askreddita.csv
│   │   ├── askredditq.csv
│   │   ├── asksciencea.csv
│   │   ├── askscienceq.csv
│   │   ├── casuala.csv
│   │   ├── casualq.csv
│   │   ├── eli5a.csv
│   │   ├── eli5q.csv
│   │   ├── mla.csv
│   │   ├── mlq.csv
│   │   └── SQL-format
│   ├── Database
│   │   └── reddit.db
│   ├── Faiss_index
│   │   ├── large.index
│   │   └── thread_idx.pkl
│   ├── faiss_it.py
│   ├── fill_data.py
│   └── Retrieve.py
├── Scripted
│   ├── Chatterbot_Corpus
│   │   ├── ai.yml
│   │   ├── botprofile.yml
│   │   ├── computers.yml
│   │   ├── conversations.yml
│   │   ├── emotion.yml
│   │   ├── food.yml
│   │   ├── gossip.yml
│   │   ├── greetings.yml
│   │   ├── health.yml
│   │   ├── history.yml
│   │   ├── humor.yml
│   │   ├── literature.yml
│   │   ├── money.yml
│   │   ├── movies.yml
│   │   ├── politics.yml
│   │   ├── psychology.yml
│   │   ├── science.yml
│   │   ├── sports.yml
│   │   └── trivia.yml
│   ├── Processed_Scripts
│   │   ├── Bot_Profile.pkl
│   │   ├── Chatterbot.pkl
│   │   ├── embedded_bot_queries.pkl
│   │   ├── embedded_chatterbot_queries.pkl
│   │   ├── intent_query_script.pkl
│   │   └── intent_response_script.pkl
│   ├── Random_Reddit_Data
│   │   ├── jokesq.csv
│   │   ├── nostupidq.csv
│   │   ├── showerthoughtsq.csv
│   │   ├── tilq.csv
│   │   └── writingpromptsa.csv
│   ├── setup.py
│   └── Subscripts
│       ├── fill_bot_profile.py
│       ├── fill_chatterbot.py
│       ├── intent_query_script.py
│       ├── intent_response_script.py
│       └── process_pkl.py
├── Sentence_Encoder
│   ├── Embeddings
│   │   ├── ConvRT
│   │   │   ├── assets
│   │   │   ├── saved_model.pb
│   │   │   ├── tfhub_module.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00002
│   │   │       ├── variables.data-00001-of-00002
│   │   │       └── variables.index
│   │   └── USE_QA
│   │       ├── assets
│   │       ├── saved_model.pb
│   │       └── variables
│   │           ├── variables.data-00000-of-00001
│   │           └── variables.index
│   ├── encoder_client.py
│   ├── meta_query_encoder_fast.py
│   ├── meta_query_encoder.py
│   ├── meta_response_encoder_fast.py
│   └── meta_response_encoder.py
├── TTS
│   ├── best_model_config.json
│   ├── config.json
│   ├── config_kusal.json
│   ├── dataset_analysis
│   │   ├── AnalyzeDataset.ipynb
│   │   ├── analyze.py
│   │   └── README.md
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── Kusal.py
│   │   ├── LJSpeechCached.py
│   │   ├── LJSpeech.py
│   │   └── TWEB.py
│   ├── debug_config.py
│   ├── extract_feats.py
│   ├── hard-sentences.txt
│   ├── images
│   │   ├── example_model_output.png
│   │   └── model.png
│   ├── layers
│   │   ├── attention.py
│   │   ├── custom_layers.py
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   └── tacotron.py
│   ├── LICENSE.txt
│   ├── models
│   │   ├── __init__.py
│   │   └── tacotron.py
│   ├── notebooks
│   │   ├── Benchmark.ipynb
│   │   ├── ReadArticle.ipynb
│   │   ├── synthesis.py
│   │   └── TacotronPlayGround.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── server
│   │   ├── conf.json
│   │   ├── README.md
│   │   ├── server.py
│   │   ├── synthesizer.py
│   │   └── templates
│   │       └── index.html
│   ├── setup.py
│   ├── synthesis.py
│   ├── tests
│   │   ├── generic_utils_text.py
│   │   ├── __init__.py
│   │   ├── layers_tests.py
│   │   ├── loader_tests.py
│   │   ├── tacotron_tests.py
│   │   └── test_config.json
│   ├── text2speech.py
│   ├── ThisBranch.txt
│   ├── train.py
│   ├── TTS.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── tts_model
│   │   ├── best_model.pth.tar
│   │   └── config.json
│   ├── utils
│   │   ├── audio_lws.py
│   │   ├── audio.py
│   │   ├── data.py
│   │   ├── generic_utils.py
│   │   ├── __init__.py
│   │   ├── text
│   │   │   ├── cleaners.py
│   │   │   ├── cmudict.py
│   │   │   ├── __init__.py
│   │   │   ├── numbers.py
│   │   │   └── symbols.py
│   │   └── visual.py
│   └── version.py
└── Utils
    ├── functions_old.py
    ├── functions.py
    └── __init__.py

```



## Project Setup

### Component # 0: Sentence Encoder Module
The sentence encoder is one of the most fundamental components of the project. It uses a concatenation of [Multi-context ConveRT](https://github.com/PolyAI-LDN/polyai-models#multi-context-convert) and [Universal Sentence Encoder QA](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3) to encode utterances. The encoder utterances are used for almost all the other modules - for retrieval, for classification etc. For this module to work you need download and store the model in the appropriate directory inside Sentence_Encoder. Refer to the directory tree for the extact location. Specifically, the content inside tensorflow-hub pretrained ConveRT model should go within Sentence_Encoder/Embeddings/ConvRT directory, and Universal Sentence Encoder Multilingual QA should go within Sentence_Encoder/Embeddings/USE_QA. The sentence encoder uses different components for encoding user utterances than that for encoding response candidates. It also can take context into account. See project report for details.


### Component # 1: Scripted Module

Rum setup.py inside the [scripted folder](https://github.com/JRC1995/Chatbot/tree/master/Scripted) to process all the necessary documents for this module. This module mostly prepares "query-responses" pair mappings. Given an utterance, the model finds the best matching query (by measuring cosine similarity or dot product of their encodings) and returns its corresponding responses. 

[Here](https://github.com/JRC1995/Chatbot/blob/master/Scripted/Subscripts/intent_response_script.py) you can write your "intent to response" maps. You can create any kind of new 'intent' (say '<fetch_water>') and you can then write the list of candidate responses for that intent. Instead of writing natural language responses you can also type in "command codes" (say <get_water>) and later upon detecting the command codes you can make the AI take any related action you want. [Here](https://github.com/JRC1995/Chatbot/blob/master/Scripted/Subscripts/intent_query_script.py) you can write "intent to query" maps. For example:

```
'<fetch_water>':['bring me some water','I want some water']
```
That is here you map intents to potential queries and utterances related to that intent. You don't need to exhaustive - the intent will be detected based on soft embedding based semantic-similarity search not through hard pattern matching.


### Component # 2: Retrieval Module
### Component # 3: Dialog-Act Classifier Module
### Component # 4: Generative Module
### Component # 5: Ranker Module
### Component # 6: TTS Module
### Component # 7: Controller Module


(under construction)
