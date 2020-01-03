# Conversational-AI

Hybrid Conversational Bot based on both neural retrieval and neural generative mechanism with TTS. 

## Credits:

* The [TTS Folder](https://github.com/JRC1995/Chatbot/tree/master/TTS) is based on an earlier version of [Mozilla TTS](https://github.com/mozilla/TTS). I prepared the [text2speech.py](https://github.com/JRC1995/Chatbot/blob/master/TTS/text2speech.py) file which acts as an interface to the rest of the project.
* The [encoder_client](https://github.com/JRC1995/Chatbot/blob/master/Sentence_Encoder/encoder_client.py) file in Sentence_Encoder is taken from [Poly-AI](https://github.com/PolyAI-LDN/polyai-models/blob/master/encoder_client.py). I modified it slightly to run it with Tensorflow 2. We also use their pre-trained models. 
* [DialoGPT (inside Generator)](https://github.com/JRC1995/Chatbot/tree/master/Generator/DialoGPT/Configs) config files and vocab are taken from the [DialoGPT repository](https://github.com/microsoft/DialoGPT). We also use their pre-trained models.
* The [data here in Classifier](https://github.com/JRC1995/Chatbot/tree/master/Classifier/data) is from [MIDAS_dialog_act repository](https://github.com/DianDYu/MIDAS_dialog_act)
* The corpus [here](https://github.com/JRC1995/Chatbot/tree/master/Scripted/Chatterbot_Corpus) is taken from ]Chatterbot_Corpus library](https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/english)
* [Mobashir Sadat](https://www.linkedin.com/in/mobashir-sadat-2b32a3112/) prepared and processed the Chatterbot Corpus and created the diagrams. 
* [Faiss](https://github.com/facebookresearch/faiss), [Huggingface's Transformers](https://github.com/huggingface/transformers), and [Tensorflow Hub](https://www.tensorflow.org/hub/overview) form important components of the project. We use pre-trained models of [Universal Sentence Encoder QA](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3) through Tensorflow Hub.

## Requirements:

Here's my explorted miniconda environment: https://github.com/JRC1995/Chatbot/blob/master/chatbot.yml

## Project Setup

### Component # 0: Sentence Encoder Module
### Component # 1: Scripted Module
### Component # 2: Retrieval Module
### Component # 3: Dialog-Act Classifier Module
### Component # 4: Generative Module
### Component # 5: Ranker Module
### Component # 6: TTS Module
### Component # 7: Controller Module


(under construction)
