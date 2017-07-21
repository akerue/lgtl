# _*_coding:utf-8_*_

from log_tool import generate_logger

logger = generate_logger(__file__)

import argparse
import kenlm

from nltk.util import ngrams
from pygments.lexers.ruby import RubyLexer

import numpy as np
import matplotlib.pyplot as plt


def replace_special_char(token):
    # 空白と改行を特別な文字に変換する
    token = token.replace(" ", "SPACE ").replace("\n", "NEWLINE ")
    # 空白を最後に一つだけつけたいので、SPACEやNEWLINEでついた空白を
    # 除いた後、空白を最後につける
    return token.rstrip() + " "


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="N", required=True, type=int)
    parser.add_argument("--threshould", dest="threshould", required=True, type=float)
    parser.add_argument("-t", "--target", required=True, type=argparse.FileType("r"), dest="target_file")
    parser.add_argument("-v", "--vector", required=True, type=str, dest="vector_path")
    args = parser.parse_args()

    model = kenlm.Model(args.vector_path)

    lexer = RubyLexer()

    token_stream = lexer.get_tokens(args.target_file.read())

    token_str = ""

    for token_data in token_stream:
        token_str += replace_special_char(token_data[-1])

    token_list = token_str.split(" ")

    bag_of_ngrams = ngrams(token_list, args.N)

    index = 0
    x = []
    y = []
    for ngram in bag_of_ngrams:
        probabilty = 1/model.perplexity(" ".join(ngram))
        x.append(index)
        y.append(probabilty)
        logger.debug("{0}: {1} ---> {2}".format(index, ngram, probabilty))

        if probabilty < args.threshould:
            logger.debug("error!!")
        index += 1
    plt.plot(x, y, ".")
    plt.show()
