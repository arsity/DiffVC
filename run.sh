#!/usr/bin/env sh

rm -rf wavs || return
rm -rf mels || return

python3 resample.py
python3 get_mel.py

