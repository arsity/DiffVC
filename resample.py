import sys
from pathlib import Path
import torchaudio
import torchaudio.transforms as T

m4singer_path = "m4singer"
target_path = "wavs"


def to_resample_rate(m4singer_path: str, target_path: str, target_sr: int):
    src_path = Path(m4singer_path)
    tgt_path = Path(target_path)

    for subdir in src_path.iterdir():
        for fn in list(subdir.glob("**/*.wav")):
            waveform, sr = torchaudio.load(fn)
            resampler = T.Resample(sr, target_sr, dtype=waveform.dtype)
            rs_waveform = resampler(waveform)

            tn = tgt_path
            for i in range(1, len(fn.parts)):
                tn = tn / fn.parts[i]
                if not tn.parent.exists():
                    tn.parent.mkdir(parents=True)

            torchaudio.save(tn, rs_waveform, target_sr)


if __name__ == '__main__':
    to_resample_rate(m4singer_path, target_path, 22050)
