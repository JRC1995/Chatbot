
import os
import sys
import io
import torch
import time
import numpy as np
from collections import OrderedDict

import librosa
import librosa.display

from TTS.synthesis import *
from TTS.models.tacotron import Tacotron
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config
from TTS.utils.text import text_to_sequence
import spacy
from pydub import AudioSegment
from pydub.playback import play



class tts_class:

    def __init__(self):

        # Set constants
        ROOT_PATH = 'TTS/tts_model/'
        MODEL_PATH = ROOT_PATH + '/best_model.pth.tar'
        # MODEL_PATH_TMP = ROOT_PATH + '/best_model.pth.tar'
        CONFIG_PATH = ROOT_PATH + '/config.json'
        OUT_FOLDER = ROOT_PATH + '/test'
        self.CONFIG = load_config(CONFIG_PATH)
        self.use_cuda = True  # True

        # load the model
        self.model = Tacotron(self.CONFIG.embedding_size,
                              self.CONFIG.num_freq, self.CONFIG.num_mels, self.CONFIG.r)

        # load the audio processor

        self.ap = AudioProcessor(self.CONFIG.sample_rate, self.CONFIG.num_mels, self.CONFIG.min_level_db,
                                 self.CONFIG.frame_shift_ms, self.CONFIG.frame_length_ms,
                                 self.CONFIG.ref_level_db, self.CONFIG.num_freq, self.CONFIG.power, self.CONFIG.preemphasis,
                                 60)

        # load model state
        if self.use_cuda:
            cp = torch.load(MODEL_PATH)
        else:
            cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

        # load the model
        self.model.load_state_dict(cp['model'])
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        self.model.decoder.max_decoder_steps = 500

        self.nlp = spacy.load("en")

    def process(self, text):
        self.model.decoder.max_decoder_steps = 500
        wavefiles = self.text2audio(text, self.model, self.CONFIG, self.use_cuda, self.ap)
        return wavefiles



    def tts(self, model, text, CONFIG, use_cuda, ap, wavefile, figures=True):
        waveform, alignment, spectrogram, stop_tokens = create_speech(
            model, text, CONFIG, use_cuda, ap)

        self.ap.save_wav(waveform, wavefile)

    def text2audio(self, text, model, CONFIG, use_cuda, ap):
        wavefiles = []
        base_name = "gen_{}.wav"

        doc = self.nlp(text)
        for i, sent in enumerate(doc.sents):
            text = sent.text.strip()
            wavefile = base_name.format(i)
            self.tts(model, text, CONFIG, use_cuda, ap, wavefile)
            wavefiles.append(wavefile)

        return wavefiles

    def play(self,wavefiles):

        voice = AudioSegment.empty()

        for wavefile in wavefiles:
            voice += AudioSegment.from_wav(wavefile)

        play(voice)

        for w in wavefiles:
            os.remove(w)
