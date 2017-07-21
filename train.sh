#!/usr/bin/env sh

python tokenize_program.py \
    -w 5 \
    -s ./source \
    -o ./database/data.dat

python nlppreprocess/replace_rare_words.py \
    --input ./database/data.dat \
    --output ./database/data.replace_rare_words \
    --min_count 7

python nlppreprocess/append_eos.py \
    --input ./database/data.replace_rare_words \
    --output ./database/data.corpus

python train.py \
    --gpu 0 \
    --corpus ./database/data.corpus \
    --config ./config/parameter.ini

