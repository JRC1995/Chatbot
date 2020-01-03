import string
import random
from nltk.tokenize import TweetTokenizer
import re
import time
import sys
import numpy as np

# source: https://gist.github.com/nealrs/96342d8231b75cf4bb82
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how're": "how are",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text):

    global cList
    global c_re

    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)

def replace(text,regex,replacement):
    def replace_fn(match):
        return replacement
    return regex.sub(replace_fn,text)


def clean(text):

    filter_list_1 = ['’']
    replacement_1 = "'"
    regex_1 = re.compile('(%s)' % '|'.join(filter_list_1))
    text = replace(text,regex_1,replacement_1)

    filter_list_2 = ['\[wp\]', 'eli5\:', 'cmv\:',
                    '\[d\]', '\[r\]', '\[n\]', '\&gt\;', '/r/', 'r/']
    replacement_2 = ''
    regex_2 = re.compile('(%s)' % '|'.join(filter_list_2))
    text = replace(text,regex_2,replacement_2)

    filter_list_3 = ['@[a-z0-9]+', '\/u\/[0-9a-z]+', '\[[0-9a-z]+\]\(\/u\/[0-9a-z]+\)']
    replacement_3 = 'someone'
    regex_3 = re.compile('(%s)' % '|'.join(filter_list_3))
    text = replace(text,regex_3,replacement_3)

    filter_list_4 = ['\[', '\]']
    replacement_4 = ' '
    regex_4 = re.compile('(%s)' % '|'.join(filter_list_4))
    text = replace(text,regex_4,replacement_4)

    return text


def simple_preprocess(text):

    tw = TweetTokenizer()

    text = text.lower()

    text = clean(text)

    text = expandContractions(text)
    # sometimes two iterations are needed for double contractions
    text = expandContractions(text)

    text = re.sub(r"[' '\(]*https\:[^ ]*|[' '\(]*http\:[^ ]*|[' ']*www\..[^ ]*", ' (url) ', text)

    tokenized_text = tw.tokenize(text)

    return text,tokenized_text

# "but hey aren ’ t snobby . the wayne ’ s are well known for their philanthropy ."

#print(simple_preprocess("[Zaflis000](/u/Zaflis000) @fodor000, &gt; [can] http://plato.stanford.edu/entries/other-minds/ you're she'll do this for me? http://plato.stanford.edu/entries/other-minds/"))


# Adapted from: https://stackoverflow.com/questions/9246076/how-to-print-one-character-at-a-time-on-one-line

def delay_print(s,t=0.05):
    for c in s:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(t)


def cosine_similarity_nd(embd1, embd2):
    numerator = np.multiply(embd1, embd2)
    numerator = np.sum(numerator, axis=1)
    eucli_norm_1 = np.sqrt(np.sum(np.power(embd1, 2), axis=1))
    eucli_norm_2 = np.sqrt(np.sum(np.power(embd2, 2), axis=1))
    denominator = np.multiply(eucli_norm_1, eucli_norm_2)
    denominator = denominator + 1e-10  # remove zeros
    cosine_similarity = np.divide(numerator, denominator)
    return cosine_similarity.reshape((-1))

def normalize(values):
    # shift and normalize - create probability distribution
    minimum_val = np.amin(values)
    values = values - minimum_val
    norm_denom = np.sum(values)
    if norm_denom == 0:
        size = values.shape[-1]
        return np.asarray([1/size for _ in range(size)],np.float32)
    else:
        return values/norm_denom
