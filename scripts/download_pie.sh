#!/usr/bin/env bash
mkdir -p ./data/pie
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QxNCh6vfNSZkod1Rg_zHLI1FM8WyXix4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QxNCh6vfNSZkod1Rg_zHLI1FM8WyXix4" -O ./data/pie/pie.zip && rm -rf /tmp/cookies.txt
cd ./data/pie && unzip pie.zip && rm pie.zip
