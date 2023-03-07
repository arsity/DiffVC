import argparse
import json
import os
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch

use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn

mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import params
from model import DiffVC

import sys

sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

import torchaudio

source_path = "mels"
target_path = "embeds"


def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


if __name__ == '__main__':

    print("\n", "\n", "-" * 10, "Get Embeds", "-" * 10)

    src_path = Path(source_path)
    tgt_path = Path(target_path)

    for subdir in src_path.iterdir():
        for fn in list(subdir.glob("**/*.wav")):

            tn = tgt_path
            for i in range(1, len(fn.parts)):
                tn = tn / fn.parts[i]
                if not tn.parent.exists():
                    tn.parent.mkdir(parents=True)

            print(fn, " --------> ", tn)
            torchaudio.save(tn, get_embed(fn), 22050, format="wav")
