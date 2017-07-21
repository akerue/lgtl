# -*- coding: utf-8 -*-


class MyIterator(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for s in open(self.path):
            yield s.decode("utf-8").strip().split()


def read_sentences(path_in):
    # f = open(path_in)
    # sents = [s.decode("utf-8").strip().split() for s in f]
    # return sents
    return MyIterator(path_in)


def write_sentences(iterator, path_out):
    print "[nlppreprocess.utils] Writing ..."
    with open(path_out, "w") as f:
        for s in iterator:
            line = " ".join(s)
            f.write("%s\n" % line.encode("utf-8"))
