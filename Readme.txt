DISCLAIMER:
THERE IS NO FOCUSED CHECKING AND FILTERING FOR TOXIC OR OTHERWISE NSFW CONTENT. I HAVE TRIED TO STICK TO RELIABLE SUBREDDITS - BUT IT'S REDDIT
SIMILARLY DIALOGPT2 IS TRAINED ON REDDIT DATA AND CAN GENERATE OFFENSIVE CONTENT.

How to run?

chatbot.yml is an export of the anaconda environment in which the project was done. Make sure the necessary dependencies are installed
(though some dependencies may be less necessary)

Then:

python interact.py (for full chatbot mode)
python interact_verbose.py (for verbose mode)
python interact_retrieval_only.py (to use retrieval/custom-script based modules only with ranker - no generation)
python interact_generator_only.py (to use only generator with ranker)

(Use interact_faster.py for the best performance)

(There may or may not be path related issues with different OS.)

Utils/functions.py utilizes a code from here: https://gist.github.com/nealrs/96342d8231b75cf4bb82
for contraction expansion.

in the same file, "delay_print" was used from: https://stackoverflow.com/questions/9246076/how-to-print-one-character-at-a-time-on-one-line

The dataset in Classifier/data/ is from: https://github.com/DianDYu/MIDAS_dialog_act

The dataset in Scripted/Chatterbot_Corpus/ is from: https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/english

The dataset in Paraphraser/Data/ is from: http://www.cs.cmu.edu/~jwieting/

The reddit csv files were personally queried out from Google Big Query, Reddit comments and submission tables.

Config files and some functions were used from DialoGPT repository: https://github.com/microsoft/DialoGPT (we also get the pre-trained weights from here)
We started building our decoding code from: https://colab.research.google.com/drive/1KTLqiAOdKM_3RnBWfqgrvOQLqumUyOdA
but our end result should be significantly different.

Sentence Encoders rely on Pre-trained Transformers loaded from Tensorflow Hub.
Syntax helps were taken from:
https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/2
https://github.com/PolyAI-LDN/polyai-models

Sentence_Encoder/encoder_client.py is taken from https://github.com/PolyAI-LDN/polyai-models

We use Huggingface's Transformers library: https://github.com/huggingface/transformers for DialogPT, and tokenization for paraphrasing and loading the underlying GPT2 model.

faiss is used now: https://github.com/facebookresearch/faiss
