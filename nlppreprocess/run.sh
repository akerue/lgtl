#!/usr/bin/env sh

RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.head_50000.txt
CORPUS_ALL=./books_large.merge.head_50000.txt.preprocessed
CORPUS_TRAIN=./books_large.merge.head_50000.txt.preprocessed.train
CORPUS_VAL=./books_large.merge.head_50000.txt.preprocessed.val

TMP=./tmp.txt
python lowercase.py \
    --input $RAW \
    --output $TMP.lowercase

# echo "[nlppreprocess;StanfordCoreNLP] Processing ..."
# rm tmp.properties
# touch tmp.properties
# echo "annotators = tokenize, ssplit" >> tmp.properties
# echo "ssplit.eolonly = true" >> tmp.properties
# echo "outputFormat = conll" >> tmp.properties
# echo "file = $TMP.lowercase" >> tmp.properties
# java -Xmx10g edu.stanford.nlp.pipeline.StanfordCoreNLP -props tmp.properties
# python conll2lines.py \
#     --input $TMP.lowercase.conll \
#     --output $TMP.tokenize

python tokenizer.py \
    --input $TMP.lowercase \
    --output $TMP.tokenize

python replace_digits.py \
    --input $TMP.tokenize \
    --output $TMP.replace_digits

python replace_rare_words.py \
    --input $TMP.replace_digits \
    --output $TMP.replace_rare_words \
    --prune_at 300000 \
    --min_count 5

python append_eos.py \
    --input $TMP.replace_rare_words \
    --output $CORPUS_ALL

python split_corpus.py \
    --all $CORPUS_ALL \
    --train $CORPUS_TRAIN \
    --val $CORPUS_VAL \
    --size 5000

python create_dictionary.py \
    --corpus $CORPUS_TRAIN
