#!/usr/bin/env sh

python tokenize_program.py \
    -w 5 \
    -s ./corrected_cookpad_source \
    -o ./database/data.dat

python nlppreprocess/replace_rare_words.py \
    --input ./database/data.dat \
    --output ./database/data.replace_rare_words \
    --min_count 7

python nlppreprocess/append_edge.py \
    --input ./database/data.replace_rare_words \
    --output ./database/data.corpus

python train_with_context.py \
    --gpu 0 \
    --corpus ./database/data.corpus \
    --config ./config/cxt_blstm.ini

