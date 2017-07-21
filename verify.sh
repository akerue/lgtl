#!/usr/bin/env sh

if [ $# -ne 1 ]; then
    echo "検証したい一つのルビーファイルを引数に指定してください" 1>&2
    exit 1
fi

ln -f $1 target/target.rb

python tokenize_program.py \
    -w 5 \
    -s ./target \
    -o ./database/target.dat

# python nlppreprocess/replace_rare_words.py \
#     --input ./database/target.dat \
#     --output ./database/target.replace_rare_words \
#     --min_count 7

python nlppreprocess/append_eos.py \
    --input ./database/target.dat \
    --output ./database/target.corpus

python verify.py \
    --gpu 0 \
    --winsize 5 \
    --model ./snapshot/rnnlm.data.corpus.parameter.iter_402600.epoch_18.model \
    --corpus ./database/data.corpus \
    --config ./config/parameter.ini \
    --target ./database/target.corpus \
    --target_raw ./target/target.rb \
    --img ./img/heatmap.png

