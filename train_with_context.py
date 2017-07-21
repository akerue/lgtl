# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import time

import chainer
from chainer import cuda, serializers, optimizers
import chainer.functions as F
import numpy as np
import pyprind

import models
import utils
import pickle

from collections import Counter


def evaluate(model, sents, ivocab, word_dim):
    train = False
    loss = 0.0
    acc = 0.0
    count = 0
    vocab_size = model.vocab_size
    for data_i in pyprind.prog_bar(xrange(len(sents))):
        words = sents[data_i:data_i+1]
        T = len(words) - 2 # BOSとEOSを取り除く

        xs, ms = utils.make_batch(words, train=train, tail=False, mask=True)

        batch_size = len(xs[0])

        words_without_edge = [w[1:-1] for w in words]
        xs_without_edge, ms_without_edge = utils.make_batch(words_without_edge, train=train, tail=False, mask=True)

        ys = model.forward(xs=xs, ms=ms, train=train)

        masked_ys = []
        for y, m in zip(ys, ms_without_edge):
            masked_ys.append(y*F.broadcast_to(F.reshape(m, (batch_size, 1)), (batch_size, vocab_size)))

        ys = F.concat(masked_ys, axis=0)
        ts = F.concat(xs_without_edge, axis=0)

        ys = F.reshape(ys, (-1, vocab_size))
        ts = F.reshape(ts, (-1,))

        loss += F.softmax_cross_entropy(ys, ts) * T 
        acc += F.accuracy(ys, ts, ignore_label=-1) * T 
        count += T

    loss_data = float(cuda.to_cpu(loss.data)) /count
    acc_data = float(cuda.to_cpu(acc.data)) / count
    return loss_data, acc_data


def main(gpu, path_corpus, path_config, path_word2vec):
    MAX_EPOCH = 50
    EVAL = 200
    MAX_LENGTH = 70
    COUNTS_CACHE = "./cache/counts.pkl"
    
    config = utils.Config(path_config)
    word_dim = config.getint("word_dim") 
    state_dim = config.getint("state_dim")
    grad_clip = config.getfloat("grad_clip")
    weight_decay = config.getfloat("weight_decay")
    batch_size = config.getint("batch_size")
    sample_size = config.getint("sample_size")
    
    print "[info] CORPUS: %s" % path_corpus
    print "[info] CONFIG: %s" % path_config
    print "[info] PRE-TRAINED WORD EMBEDDINGS: %s" % path_word2vec
    print "[info] WORD DIM: %d" % word_dim
    print "[info] STATE DIM: %d" % state_dim
    print "[info] GRADIENT CLIPPING: %f" % grad_clip
    print "[info] WEIGHT DECAY: %f" % weight_decay
    print "[info] BATCH SIZE: %d" % batch_size

    path_save_head = os.path.join(config.getpath("snapshot"),
            "rnnlm.%s.%s" % (
                os.path.basename(path_corpus),
                os.path.splitext(os.path.basename(path_config))[0]))
    print "[info] SNAPSHOT: %s" % path_save_head
    
    sents_train, sents_val, vocab, ivocab = \
            utils.load_corpus(path_corpus=path_corpus, max_length=MAX_LENGTH)

    #counts = None

    #print("[info] Load word counter")
    #if os.path.exists(COUNTS_CACHE):
    #    print("[info] Found cache of counter")
    #    counts = pickle.load(open(COUNTS_CACHE, "rb"))

    #    if len(counts) != len(vocab):
    #        counts = None

    #if counts is None:
    #    counts = Counter()

    #    for sent in list(sents_train) + list(sents_val):
    #        counts += Counter(sent)

    #    pickle.dump(counts, open(COUNTS_CACHE, "wb"))

    #cs = [counts[w] for w in range(len(counts))]

    if path_word2vec is not None:
        word2vec = utils.load_word2vec(path_word2vec, word_dim)
        initialW = utils.create_word_embeddings(vocab, word2vec, dim=word_dim, scale=0.001)
    else:
        initialW = None

    cuda.get_device(gpu).use()

    model = models.CXT_BLSTM(
            vocab_size=len(vocab),
            word_dim=word_dim,
            state_dim=state_dim,
            initialW=initialW,
            EOS_ID=vocab["<EOS>"])

    model.to_gpu(gpu)

    opt = optimizers.SMORMS3()
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # sampler = utils.RandomSampler(cs, sample_size)

    #print "[info] Evaluating on the validation sentences ..."
    #loss_data = evaluate(model, sents_val, ivocab, word_dim, sampler)
    #print "[validation] iter=0, epoch=0, loss=%f" \
    #    % (loss_data)
    
    it = 0
    n_train = len(sents_train)
    vocab_size = model.vocab_size

    for epoch in xrange(1, MAX_EPOCH+1):
        perm = np.random.permutation(n_train)
        for data_i in xrange(0, n_train, batch_size):
            if data_i + batch_size > n_train:
                break
            words = sents_train[perm[data_i:data_i+batch_size]]
            xs, ms = utils.make_batch(words, train=True, tail=False, mask=True)

            ys = model.forward(xs=xs, ms=ms, train=True)
            
            words_without_edge = [w[1:-1] for w in words]
            xs_without_edge, ms_without_edge = utils.make_batch(words_without_edge, train=True, tail=False, mask=True)

            masked_ys = []
            for y, m in zip(ys, ms_without_edge):
                m_ext = F.broadcast_to(F.reshape(m, (batch_size, 1)), (batch_size, vocab_size))
                masked_ys.append(y*m_ext)

            #ts = model.embed_words(xs_without_edge, ms_without_edge, train=True) # BOS, EOSは除く

            #  T : バッチの中の最大長
            #  N : バッチサイズ
            # |D|: word_dim
            ys = F.concat(masked_ys, axis=0) # (TN, |V|)
            ts = F.concat(xs_without_edge, axis=0) # (TN, |D|)

            ys = F.reshape(ys, (-1, vocab_size)) # (TN, |D|)
            ts = F.reshape(ts, (-1,)) # (TN,)

            loss = F.softmax_cross_entropy(ys, ts)
            acc = F.accuracy(ys, ts, ignore_label=-1)
        
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            opt.update()
            it += 1

            loss_data = float(cuda.to_cpu(loss.data))
            perp = math.exp(loss_data)
            acc_data = float(cuda.to_cpu(acc.data))

            print "[training] iter=%d, epoch=%d (%d/%d=%.03f%%), perplexity=%f, accuracy=%.2f%%" \
                    % (it, epoch, data_i+batch_size, n_train,
                        float(data_i+batch_size)/n_train*100,
                        perp, acc_data*100)

            if it % EVAL == 0:
                print "[info] Evaluating on the validation sentences ..."
                loss_data, acc_data = evaluate(model, sents_val, ivocab, word_dim)
                perp = math.exp(loss_data)
                print "[validation] iter=%d, epoch=%d, perplexity=%f, accuracy=%.2f%%" \
                        % (it, epoch, perp, acc_data*100)

                serializers.save_npz(path_save_head + ".iter_%d.epoch_%d.model" % (it, epoch),
                        model)
                # utils.save_word2vec(path_save_head + ".iter_%d.epoch_%d.vectors.txt" % (it, epoch),
                #         utils.extract_word2vec(model, vocab))
                print "[info] Saved."

    print "[info] Done."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="gpu", type=int, default=0)
    parser.add_argument("--corpus", help="path to corpus", type=str, required=True)
    parser.add_argument("--config", help="path to config", type=str, required=True)
    parser.add_argument("--word2vec", help="path to pre-trained word vectors", type=str, default=None)
    args = parser.parse_args()

    gpu = args.gpu
    path_corpus = args.corpus
    path_config = args.config
    path_word2vec = args.word2vec

    main(
        gpu=gpu,
        path_corpus=path_corpus,
        path_config=path_config,
        path_word2vec=path_word2vec)

