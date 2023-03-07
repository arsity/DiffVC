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

# mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

source_path = "wavs"
target_path = "mels"


### Legacy implement by librosa
def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


### New implement by torchaudio
# def get_mel(wav_path):
#     wav, _ = torchaudio.load(wav_path)
#     mel_spectrogram = T.MelSpectrogram(
#         sample_rate=22050,
#         n_fft=1024,
#         n_mels=80,
#         f_min=0,
#         f_max=8000,
#         hop_length=256,
#         win_length=1024,
#         center=False,
#         pad_mode="reflect",
#         power=2.0,
#         norm='slaney',
#         onesided=True,
#         mel_scale="slaney",
#         window_fn=torch.hann_window
#     )
#     return mel_spectrogram(wav)[0]


if __name__ == '__main__':

    print("\n", "\n", "-" * 10, "Get Mel Spectrogram", "-" * 10)

    src_path = Path(source_path)
    tgt_path = Path(target_path)

    for subdir in src_path.iterdir():
        for fn in list(subdir.glob("**/*.wav")):

            tn = tgt_path
            for i in range(1, len(fn.parts)):
                tn = tn / fn.parts[i]
                if not tn.parent.exists():
                    tn.parent.mkdir(parents=True)

            tn = tn.__str__().split(".")[0] + ".npy"
            print(fn, " --------> ", tn)
            np.save(tn, get_mel(fn))
