# -*- coding: utf-8 -*-

import argparse
import os

import gensim

import utils


class CharIterator(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for s in open(self.path):
            yield list(s.decode("utf-8").replace("\n", "\\"))


def main(path, char):
    assert os.path.exists(path)
    
    if not char:
        iterator = utils.read_sentences(path)
    else:
        print "[nlppreprocess.create_dictionary] NOTE: char-level mode!"
        iterator = CharIterator(path)

    print "[nlppreprocess.create_dictionary] Processing ..."
    dictionary = gensim.corpora.Dictionary(iterator, prune_at=None)
    vocab = dictionary.token2id
    print "[nlppreprocess.create_dictionary] Vocabulary size: %d" % len(vocab)
    
    if not char:
        dictionary.save_as_text(path + ".dictionary")
        print "[nlppreprocess.create_dictionary] Saved the dictionary to %s" % (path + ".dictionary")
    else:
        dictionary.save_as_text(path + ".char.dictionary")
        print "[nlppreprocess.create_dictionary] Saved the dictionary to %s" % (path + ".char.dictionary")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--char", type=int, default=0)
    args = parser.parse_args()

    path = args.corpus
    char = bool(args.char)

    main(path=path, char=char)
