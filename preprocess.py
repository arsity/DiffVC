import sys
from pathlib import Path

import librosa
import numpy as np
from encoder import inference as spk_encoder
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn

m4singer_path = "m4singer"


def to_half_sample_rate(m4singer_path):
    path = Path("m4singer")
    for subdir in path.iterdir():
        for fn in list(subdir.glob("**/*.wav")):
            print(fn)
            sys.stdout.flush()


mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)


def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


if __name__ == '__main__':
    to_half_sample_rate(m4singer_path)
