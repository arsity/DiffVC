#!/usr/bin/env sh

rm -rf wavs || return
rm -rf mels || return
rm -rf embeds || return

python3 resample.py
python3 get_mel.py
python3 get_embed.py
