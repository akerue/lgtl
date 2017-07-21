# -*- coding: utf-8 -*-

import argparse

import utils


def conll2lines(path_in):
    print "[nlppreprocess.conll2lines] Processing ..."
    sents = []
    s = []
    for line in open(path_in):
        line = line.decode("utf-8").strip().split()
        if len(line) == 0:
            continue
        index = int(line[0])
        w = line[1]
        if index == 1:
            if len(s) != 0:
                sents.append(s)
                s = []
            else:
                pass
        s.append(w)
    if len(s) != 0:
        sents.append(s)
    return sents


def main(path_in, path_out):
    sents = conll2lines(path_in)
    utils.write_sentences(sents, path_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input corpus", type=str, required=True)
    parser.add_argument("--output", help="path to output corpus", type=str, required=True)
    args = parser.parse_args()

    path_in = args.input
    path_out = args.output

    main(path_in=path_in, path_out=path_out)
