# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import time
import re
import json

import chainer
from chainer import cuda, serializers, optimizers
import chainer.functions as F
import numpy as np
import pyprind

import models
import utils

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from tokenize_program import tokenize

def parse(vocab, path_target):
    # split
    print "[info] Loading the preprocessed corpus ..."
    sents = open(path_target)
    sents = [s.strip().decode("utf-8").split() for s in sents]

    # All sentences must be end with the "<EOS>" token 
    print "[info] Checking '<EOS>' tokens ..."
    sents = [s + ["<EOS>"] if s[-1] != "<EOS>" else s for s in sents]

    # transform words to IDs
    print "[info] Transforming words to IDs ..."
    sents = [[vocab.get(w, vocab["<UNK>"]) for w in s]for s in sents]

    # transform list to numpy.ndarray
    print "[info] Transforming list to numpy.ndarray"
    sents = np.asarray(sents)

    print "[info] # of verified sentences: %d" % len(sents)
    
    return sents


def inspect(model, sents):
    train = False

    probs = []
    for index in pyprind.prog_bar(xrange(len(sents))):
        words = sents[index:index+1]

        xs, ms = utils.make_batch(words, train=train, tail=False, mask=True)

        ys = model.forward(xs=xs, ms=ms, train=train)
        ys = F.concat(ys, axis=0)
        ys = F.softmax(ys)
        ys = cuda.to_cpu(ys.data)

        probs.append(ys[np.arange(len(ys)), words[0][1:-1]])

    return probs


def aggregate(sents, probs, vocab, ivocab, win_size):
    assert len(sents) == len(probs)

    # BOSとEOSを取り除く
    sents = [sent[1:-1] for sent in sents]

    for i in range(len(sents)):
        assert len(sents[i]) == len(probs[i])

    aggregated_sents = sents[0]
    aggregated_probs = probs[0]
    
    pre_lag = 0
    lag_index = [0,]
    for sent_index in xrange(1, len(sents)):
        pre_words = sents[sent_index-1]
        words = sents[sent_index]
        for word_index in xrange(len(pre_words)):
            # 一行前の最初の改行を探す
            if pre_words[word_index] == vocab["<NEWLINE>"]:
                pre_lag += word_index + 1
                lag_index.append(pre_lag)
                break

        # 最初の改行を前のデータの塊とのズレとして、ズレの分だけ0でパディングする
        pre_padding = np.zeros(pre_lag)
        appended_probs = np.concatenate((pre_padding, probs[sent_index]), axis=0)

        # 後方のズレも0でパディングする
        post_lag = len(appended_probs) - len(aggregated_probs)
        aggregated_probs = np.concatenate((aggregated_probs, np.zeros(post_lag)), axis=0)

        # ズレを補正した上で確率を足し合わせる
        aggregated_probs = aggregated_probs + appended_probs

        # トークンもつなぎ合わせる
        aggregated_sents = aggregated_sents[:pre_lag] + words

    tail_sent = aggregated_sents[lag_index[-1]:]
    for i in xrange(len(tail_sent)):
        if tail_sent[i] == vocab["<NEWLINE>"]:
            lag_index.append(pre_lag + i + 1)

    if len(sents) > 5:
        for i in xrange(win_size - 1):
            # 確率の平均を計算する時、前方と後方はwin_size回より少ない回数しか
            # 足されていないので、足された回数だけ割る
            aggregated_probs[lag_index[i]:lag_index[i+1]] /= (i + 1)
            aggregated_probs[lag_index[-i-2]:lag_index[-i-1]] /= (i + 1)

        # 残りはwin_size回だけ割る
        aggregated_probs[lag_index[win_size-1]:lag_index[-win_size]] /= win_size

    return aggregated_sents, aggregated_probs


def collate(tokens, probs, path_program):
    token_streams, _ = tokenize(path_program, raw=True)
    
    raw_tokens = []
    for line_of_token in token_streams:
        for token in line_of_token:
            raw_tokens.extend(token.split(" ")[:-1]) # 最後に空文字が入るので除去
    
    from pprint import pprint

    # pprint(map(None, raw_tokens, tokens))

    assert len(raw_tokens) == len(tokens)

    # これらの特殊文字に該当する文字列を列挙
    special_tokens = {"<NEWLINE>": (1, "\n"),
                      "<SPACE>"  : (1, " "),}

    prob_dist = []
    grid_text = []
    line_prob_dist = []
    line_text = []
    last_indent_count = 0

    for index in xrange(len(raw_tokens)):
        t = raw_tokens[index]
        if t == "<NEWLINE>":
            char_len = 1 
            char_list = list("\n")
            try:
                if raw_tokens[index+1] != "<NEWLINE>" and not re.search("INDENT", raw_tokens[index+1]):
                    # 改行が二回続いた場合を除いて
                    # 改行の後、インデントが無いということはその行は
                    # インデントされていないということになるが、raw_token
                    # 内にはその情報が存在しないため、先にlast_indent_count
                    # をアップデートしておく
                    last_indent_count = 0
            except IndexError:
                # 最終行なので無視
                pass
        elif t == "<SPACE>":
            char_len = 1 
            char_list = list(" ")
        elif re.search("INDENT", t):
            indent_count = int(t[7:-1]) + last_indent_count
            char_len = 2 * indent_count
            char_list = list(" "*char_len)
            last_indent_count = indent_count
        else:
            if len(t) != 0 and t[0] == "#":
                # コメント内部のSPACEを空白に置換する
                t = t.replace("<SPACE>", " ")

            char_len = len(t)
            char_list = list(t)

        line_prob_dist.append(np.array([probs[index]]*char_len))
        line_text.extend(char_list)

        # print (t, char_len, probs[index])

        if t == "<NEWLINE>":
            prob_dist.append(line_prob_dist)
            grid_text.append(line_text)
            line_prob_dist = []
            line_text = []

    max_length = 0
    for i in xrange(len(prob_dist)):
        prob_dist[i] = np.concatenate(prob_dist[i], axis=0)
        if max_length < len(prob_dist[i]):
            max_length = len(prob_dist[i])

    for i in xrange(len(prob_dist)):
        padding_length = max_length - len(prob_dist[i])
        prob_dist[i] = np.pad(prob_dist[i], padding_length, "constant", constant_values=1)[padding_length:]

    return prob_dist, grid_text


def generate_json(prob_dist, thre, program_path, json_path):
    # 評価用のjsonファイルを生成
    # とりえあず手前味噌だが確率が低い位置を全て出す
    line = 0
    wrong_points = []

    for p in prob_dist:
        line_index = np.where(p < thre)

        for i in line_index[0]:
            wrong_points.append((line, i))

        line += 1

    json_dict = {"path": os.path.realpath(program_path),
                 "points": wrong_points}

    with open(json_path, "w") as f:
        json.dump(json_dict, f)


def draw_heatmap(data, grid_text, path_img, min_value=0.0, max_value=1.0):
    # data = np.log(1.0 - data)
    data = 1.0 - data

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    heatmap = ax.pcolor(data, cmap=plt.cm.Reds, vmin=min_value)

    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)    

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    for i in xrange(len(grid_text)):
        for j in xrange(len(grid_text[i])):
            plt.text(j+0.5, i+0.5, grid_text[i][j], horizontalalignment="center", verticalalignment="center")

    fig.set_figheight(data.shape[0]/5)
    fig.set_figwidth(data.shape[1]/8)

    plt.savefig(path_img)


def main(gpu, path_model, path_corpus, path_config, path_target, path_program, path_json, path_img, win_size, path_word2vec):
    MAX_LENGTH = 70

    config = utils.Config(path_config)
    word_dim = config.getint("word_dim") 
    state_dim = config.getint("state_dim")
    batch_size = config.getint("batch_size")

    print "[info] CONFIG: %s" % path_config
    print "[info] PRE-TRAINED WORD EMBEDDINGS: %s" % path_word2vec
    print "[info] LOADED MODEL: %s" % path_model
    print "[info] WORD DIM: %d" % word_dim
    print "[info] STATE DIM: %d" % state_dim
    print "[info] BATCH SIZE: %d" % batch_size

    sents_train, sents_val, vocab, ivocab = \
            utils.load_corpus(path_corpus=path_corpus, max_length=MAX_LENGTH)

    cuda.get_device(gpu).use()

    model = utils.load_cxt_model(path_model, path_config, vocab)
    model.to_gpu(gpu)

    sents = parse(vocab, path_target)
    probs = inspect(model, sents)

    words, probs = aggregate(sents, probs, vocab, ivocab, win_size)

    tokens = [ivocab[w] for w in words] 

    prob_dist, grid_text = collate(tokens, probs, path_program)

    generate_json(prob_dist, 0.05, path_program, path_json)
    draw_heatmap(np.array(prob_dist), grid_text, path_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="gpu", type=int, default=0)
    parser.add_argument("--model", help="path to model", type=str, required=True)
    parser.add_argument("--corpus", help="path to corpus", type=str, required=True)
    parser.add_argument("--config", help="path to config", type=str, required=True)
    parser.add_argument("--target", help="path to target", type=str, required=True)
    parser.add_argument("--target_raw", help="path to target of source", type=str, required=True)
    parser.add_argument("--json", help="path to json file", type=str, required=True)
    parser.add_argument("--img", help="path to img", type=str, required=True)
    parser.add_argument("--winsize", help="window size", type=int, required=True)
    parser.add_argument("--word2vec", help="path to pre-trained word vectors", type=str, default=None)
    args = parser.parse_args()

    gpu = args.gpu
    path_model = args.model
    path_corpus = args.corpus
    path_config = args.config
    path_target = args.target
    path_program = args.target_raw
    path_json = args.json
    path_img = args.img
    win_size = args.winsize
    path_word2vec = args.word2vec

    main(
        gpu=gpu,
        path_model=path_model,
        path_corpus=path_corpus,
        path_config=path_config,
        path_target=path_target,
        path_program=path_program,
        path_json=path_json,
        path_img=path_img,
        win_size=win_size,
        path_word2vec=path_word2vec)
