# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
from chainer import cuda, serializers, Variable
import chainer.functions as F
import gensim

import models
from Config import Config


def load_corpus(path_corpus, max_length):
    N_VAL = 50

    start_time = time.time()
    
    # split
    print "[info] Loading the preprocessed corpus ..."
    sents = open(path_corpus)
    sents = [s.decode("utf-8").strip().split() for s in sents]

    # All sentences must be end with the "<EOS>" token
    print "[info] Checking '<EOS>' tokens ..."
    sents = [s + ["<EOS>"] if s[-1] != "<EOS>" else s for s in sents]
    
    # construct a dictionary
    print "[info] Constructing a dictionary ..."
    dictionary = gensim.corpora.Dictionary(sents, prune_at=None)
    vocab = dictionary.token2id

    ivocab = {i:w for w,i in vocab.items()}

    print "[info] Vocabulary size: %d" % len(vocab)
    
    # transform words to IDs
    print "[info] Transforming words to IDs ..."
    sents = [[vocab[w] for w in s] for s in sents]

    # XXX: filter sentences
    print "[info] Filtering sentences with more than %d words ..." % max_length
    sents = [s for s in sents if len(s) <= max_length]

    # transform list to numpy.ndarray
    print "[info] Transforming list to numpy.ndarray"
    sents = np.asarray(sents)
    
    perm = np.random.RandomState(1234).permutation(len(sents))
    sents_train = sents[perm[0:-N_VAL]]
    sents_val = sents[perm[-N_VAL:]]
    print "[info] # of training sentences: %d" % len(sents_train)
    print "[info] # of validation sentences: %d" % len(sents_val)

    print "[info] Completed. %d [sec]" % (time.time() - start_time)
    return sents_train, sents_val, vocab, ivocab


def load_word2vec(path, dim):
    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            l = line.strip().split()
            if len(l[1:]) != dim:
                print "[info] dim %d(actual) != %d(expected), skipped line %d" % \
                        (len(l[1:]), dim, line_i+1)
                continue
            word2vec[l[0].decode("utf-8")] = np.asarray([float(x) for x in l[1:]])
    return word2vec


def create_word_embeddings(vocab, word2vec, dim, scale):
    task_vocab = vocab.keys()
    print "[info] Vocabulary size (corpus): %d" % len(task_vocab)
    word2vec_vocab = word2vec.keys()
    print "[info] Vocabulary size (pre-trained): %d" % len(word2vec_vocab)
    common_vocab = set(task_vocab) & set(word2vec_vocab)
    print "[info] Pre-trained words in the corpus: %d (%d/%d = %.2f%%)" \
        % (len(common_vocab), len(common_vocab), len(task_vocab),
            float(len(common_vocab))/len(task_vocab)*100)
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in common_vocab:
        W[vocab[w], :] = word2vec[w]
    return W


def make_batch(xs, train, tail=True, mask=False):
    N = len(xs)
    max_length = -1

    for i in xrange(N):
        l = len(xs[i])
        if l > max_length:
            max_length = l

    ys = np.zeros((N, max_length), dtype=np.int32)
    ms = np.zeros((N, max_length), dtype=np.float32)

    if tail:
        for i in xrange(N):
            l = len(xs[i])
            ys[i, 0:max_length-l] = -1
            ys[i, max_length-l:] = xs[i]
            ms[i, 0:max_length-l] = 0.0
            ms[i, max_length-l:] = 1.0
    else:
        for i in xrange(N):
            l = len(xs[i])
            ys[i, 0:l] = xs[i]
            ys[i, l:] = -1
            ms[i, 0:l] = 1.0
            ms[i, l:] = 0.0

    ys = [Variable(cuda.cupy.asarray(ys[:, j]), volatile=not train)
            for j in xrange(ys.shape[1])]
    ms = [Variable(cuda.cupy.asarray(ms[:, j]), volatile=not train)
            for j in xrange(ms.shape[1])]

    if mask:
        return ys, ms
    else:
        return ys


def load_model(path_model, path_config, vocab):
    config = Config(path_config)
    model_name = config.getstr("model")
    word_dim = config.getint("word_dim")
    state_dim = config.getint("state_dim")

    if model_name == "rnn":
        model = models.RNN(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "lstm":
        model = models.LSTM(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "gru":
        model = models.GRU(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "bd_lstm":
        model = models.BD_LSTM(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    else:
        print "[error] Unkwown model name: %s" % model_name
        sys.exit(-1)
    serializers.load_npz(path_model, model)
    return model


def load_cxt_model(path_model, path_config, vocab):
    config = Config(path_config)
    word_dim = config.getint("word_dim")
    state_dim = config.getint("state_dim")

    model = models.CXT_BLSTM(
            vocab_size=len(vocab),
            word_dim=word_dim,
            state_dim=state_dim,
            initialW=None,
            EOS_ID=vocab["<EOS>"])

    serializers.load_npz(path_model, model)
    return model


def extract_word2vec(model, vocab):
    word2vec = {}
    for w in vocab.keys():
        word2vec[w] = cuda.to_cpu(model.embed.W.data[vocab[w]])
    return word2vec


def save_word2vec(path, word2vec):
    with open(path, "w") as f:
        for w, v in word2vec.items():
            line = " ".join([w] + [str(v_i) for v_i in v]).encode("utf-8") + "\n"
            f.write(line)


class RandomSampler():
    def __init__(self, counts, sample_size):
        self.vocab_size = len(counts)
        counts = np.asarray(counts, dtype=np.float32)
        counts = np.exp(np.log(np.power(counts, 3))/4)
        self.weight = counts/np.sum(counts)

        self.sample_size = sample_size

    def sampling(self):
        return np.random.choice(range(self.vocab_size), self.sample_size, p=self.weight)


def pairwise_ranking_loss(h_cxt, h_wrd_pos, h_wrd_negs):
    N = h_cxt.shape[0]

    score_pos = F.batch_matmul(h_cxt, h_wrd_pos, transa=True)
    score_pos = F.reshape(score_pos, (N,))

    loss = 0.0
    for h_wrd_neg in h_wrd_negs:
        score_neg = F.batch_matmul(h_cxt, h_wrd_pos, transa=True)
        score_neg = F.reshape(score_neg, (N,))
        loss += F.sum(F.clip(1.0 + score_neg - score_pos, 0.0, 100000000.0))

    loss /= (N * len(h_wrd_negs))

    return loss


def negative_sampling_loss(h_cxt, h_wrd_pos, h_wrd_negs):
    N = h_cxt.shape[0]

    h_wrd_negs = [F.broadcast_to(x, h_wrd_pos.shape) for x in h_wrd_negs]

    score_pos = F.batch_matmul(h_cxt, h_wrd_pos, transa=True)
    score_pos = F.log(F.sigmoid(score_pos)+0.00000001)
    print(score_pos.data)
    #score_pos = F.reshape(score_pos, (N,))

    score_neg = None
    for h_wrd_neg in h_wrd_negs:
        score_neg_tmp = F.batch_matmul(h_cxt, h_wrd_neg, transa=True)
        score_neg_tmp = F.log(F.sigmoid(-1 * score_neg_tmp)+0.00000001)
        score_neg = score_neg_tmp if score_neg is None else score_neg + score_neg_tmp
    print(score_neg.data)

    #score_neg = F.reshape(score_neg, (N,))
    
    loss = F.sum(- score_pos - score_neg)
    loss /= N
    
    return loss


def softmax_loss(h_cxt, h_wrd_pos, xs, embed_dic):
    N = h_cxt.shape[0]
    xs = F.concat(xs, axis=0)
    zeros = cuda.cupy.zeros_like(xs)

    score_pos = F.batch_matmul(h_cxt, h_wrd_pos, transa=True)
    score_pos = F.log(F.sigmoid(score_pos)+0.00000001)

    score_neg = None
    for token_id, h_wrd_neg in embed_dic.items():
        cond = cuda.cupy.equal((xs-token_id).data, zeros)
        mask = Variable(cuda.cupy.where(cond, 0., 1.).astype(np.float32))
        mask = F.expand_dims(mask, axis=1)
        mask = F.broadcast_to(mask, h_cxt.shape)
        score_neg_tmp = F.batch_matmul(h_cxt*mask, h_wrd_neg, transa=True)
        score_neg_tmp = F.log(F.sigmoid(-1 * score_neg_tmp)+0.00000001)
        score_neg = score_neg_tmp if score_neg is None else score_neg + score_neg_tmp
    
    loss = F.sum(- score_pos - score_neg)
    loss /= N
    
    return loss
