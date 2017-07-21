#!/usr/bin/env sh

main() {
    # $1: 出力のイメージファイル名
    python tokenize_program.py \
        -w 5 \
        -s ./target \
        -o ./database/target.dat

    # python nlppreprocess/replace_rare_words.py \
    #     --input ./database/target.dat \
    #     --output ./database/target.replace_rare_words \
    #     --min_count 7

    python nlppreprocess/append_edge.py \
        --input ./database/target.dat \
        --output ./database/target.corpus

    python verify_with_context.py \
        --gpu 0 \
        --winsize 5 \
        --model ./corrected_cookpad_snapshot/redmine_iter35400_epoch50_2.model \
        --corpus ./database/data.corpus \
        --config ./config/cxt_blstm.ini \
        --target ./database/target.corpus \
        --target_raw ./target/target.rb \
        --json ./json/$1 \
        --img ./img/$2
}

if [ $# -ne 1 ]; then
    echo "検証したい一つのルビーファイル、もしくはフォルダを引数に指定してください" 1>&2
    exit 1
fi

if [ -f $1 ]; then
    echo "${1} を検査します"
    json_name=`basename $1 | sed 's/\.[^\.]*$/\.json/'`
    img_name=`basename $1 | sed 's/\.[^\.]*$/\.png/'`
    ln -f $1 target/target.rb
    main $json_name $img_name
fi

if [ -d $1 ]; then
    echo "${1} の中にあるルビーファイルを検査します"
    for filepath in `\find $1 -name '*.rb'`; do
        echo "${filepath}を検査します"
        json_name=`basename $filepath | sed 's/\.[^\.]*$/\.json/'`
        img_name=`basename $filepath | sed 's/\.[^\.]*$/\.png/'`
        ln -f $filepath target/target.rb
        main $json_name $img_name
    done
fi
