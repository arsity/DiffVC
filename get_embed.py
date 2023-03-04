import sys
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed