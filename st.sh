#!/usr/bin/env sh

if [ $# -ne 1 ]; then
    echo "検証したい一つのルビーファイルを引数に指定してください" 1>&2
    exit 1
fi

ln -f $1 target/target.rb
