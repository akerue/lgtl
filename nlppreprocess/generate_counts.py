# -*- coding: utf-8 -*-

import argparse
from collections import Counter

import utils

def main(path_in, path_out):
    print "[nlppreprocess.generate_counts] Processing ..."
    iterator = utils.read_sentences(path_in)
    iterator = AppendEdgeOfSent(iterator)
    utils.write_sentences(iterator, path_out)


if __name__ == "__main__":
    counts = Counter()
    for sent in list(sents_train) + list(sents_val):
        counts += Counter(sent)
