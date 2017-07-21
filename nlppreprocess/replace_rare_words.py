# -*- coding: utf-8 -*-

import argparse

import gensim

import utils


class ReplaceRareWords(object):
    def __init__(self, iterator, prune_at, min_count):
        self.iterator = iterator
        self.vocab = self.create_dictionary(prune_at, min_count)

    def __iter__(self):
        identical = dict(zip(self.vocab, self.vocab))
        for s in self.iterator:
            yield [identical.get(w, "<UNK>") for w in s]

    def create_dictionary(self, prune_at, min_count):
        # temporal dictionary
        dictionary = gensim.corpora.Dictionary(self.iterator, prune_at=prune_at)
        dictionary.filter_extremes(no_below=min_count, no_above=1.0, keep_n=prune_at)
        vocab = dictionary.token2id
        print "[nlppreprocess.replace_rare_words] Vocabulary size: %d (w/o '<UNK>')" % len(vocab)
        return vocab


def count_UNK_rate(iterator):
    n_unk = 0
    n_total = 0
    for s in iterator:
        for w in s:
            if w == "<UNK>":
                n_unk += 1
        n_total += len(s)
    print "[nlppreprocess.replace_rare_words] # of '<UNK>' tokens: %d (%d/%d = %.2f%%)" % \
            (n_unk, n_unk, n_total, float(n_unk)/n_total * 100)


def main(path_in, path_out, prune_at, min_count):
    print "[nlppreprocess.replace_rare_words] Processing ..."
    iterator = utils.read_sentences(path_in)
    iterator = ReplaceRareWords(iterator, prune_at, min_count)
    count_UNK_rate(iterator)
    utils.write_sentences(iterator, path_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input corpus", type=str, required=True)
    parser.add_argument("--output", help="path to output corpus", type=str, required=True)
    parser.add_argument("--prune_at", help="prune at", type=int, default=1000000)
    parser.add_argument("--min_count", help="min count", type=int, default=0)
    args = parser.parse_args()

    path_in = args.input
    path_out = args.output
    prune_at = args.prune_at
    min_count = args.min_count

    main(path_in=path_in, path_out=path_out, prune_at=prune_at, min_count=min_count)
