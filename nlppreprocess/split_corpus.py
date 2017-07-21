# -*- coding: utf-8 -*-

import argparse

import numpy as np

import utils


def main(path_all, path_train, path_val, size):
    print "[nlppreprocess.split_corpus] Processing ..."
    iterator = utils.read_sentences(path_all)
    
    count = 0
    for s in open(path_all):
        count += 1
    N = count
    print "[nlppreprocess.split_corpus] Total size: %d" % N
    perm = np.random.RandomState(1234).permutation(N)
    val_index = perm[-size:]
    
    print "[nlppreprocess.split_corpus] Writing ..."
    f_train = open(path_train, "w")
    f_val = open(path_val, "w")
    for i, s in enumerate(iterator):
        line = " ".join(s)
        if i in val_index:
            f_val.write("%s\n" % line.encode("utf-8"))
        else:
            f_train.write("%s\n" % line.encode("utf-8"))
    f_train.flush()
    f_train.close()
    f_val.flush()
    f_val.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", help="path to input corpus", type=str, required=True)
    parser.add_argument("--train", help="path to output training corpus", type=str, required=True)
    parser.add_argument("--val", help="path to output validation corpus", type=str, required=True)
    parser.add_argument("--size", help="validation size", type=int, required=True)
    args = parser.parse_args()

    path_all = args.all
    path_train = args.train
    path_val = args.val
    size = args.size


    main(path_all=path_all, path_train=path_train, path_val=path_val, size=size)
