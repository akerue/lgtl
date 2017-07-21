# _*_coding:utf-8_*_

import os
import re
import argparse

from pygments.lexers.ruby import RubyLexer
from pygments.token import is_token_subtype

from pygments.token import Comment, Literal, String, Number, Name, Token

reserved = ("BEGIN",    "class",    "ensure",   "nil",      "self",     "when",
            "END",      "def",      "false",    "not",      "super",    "while",
            "alias",    "defined?", "for",      "or",       "then",     "yield",
            "and",      "do",       "if",       "redo",     "true",     "__LINE__",
            "begin",    "else",     "in",       "rescue",   "undef",    "__FILE__",
            "break",    "elsif",    "module",   "retry",    "unless",   "__ENCODING__",
            "case",     "end",      "next",     "return",   "until",    "except",
            "let",      "raise")


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


def replace_special_char(token, comment=False):
    # 空白と改行を特別な文字に変換する
    # コメントの場合、コメント中に存在する空白をそのままにしておくと
    # コーパスファイル中でトークンの区切りと認識されるので、特殊文字
    # に変換している
    if comment:
        token = token.replace(" ", "<SPACE>")
    else:
        token = token.replace(" ", "<SPACE> ").replace("\n", "<NEWLINE> ")
    # 空白を最後に一つだけつけたいので、SPACEやNEWLINEでついた空白を
    # 除いた後、空白を最後につける
    return token.rstrip()


def tokenize(program_path, raw=False):
    lexer = RubyLexer()

    token_streams = []
    with open(program_path, "r") as f:
        program = f.readlines()

    num_of_lines = len(program)

    last_indent_count = 0

    for line in program:
        line_of_token = []
        for token_data in lexer.get_tokens(line):
            token_type = token_data[0]
            token = token_data[-1]

            if raw:
                if is_token_subtype(token_type, Comment) or is_token_subtype(token_type, Literal):
                    arranged_token = replace_special_char(token, comment=True)
                else:
                    arranged_token = replace_special_char(token, comment=False)
            else:
                if is_token_subtype(token_type, Literal):
                    arranged_token = "<LITERAL>"
                elif is_token_subtype(token_type, String):
                    arranged_token = "<STRING>"
                elif is_token_subtype(token_type, Number):
                    arranged_token = "<NUMBER>"
                elif token_type == Token.Name.Operator:
                    arranged_token = "<OPERATOR>"
                elif token_type == Name and token not in reserved:
                    arranged_token = "<ID>"
                elif token_type == Name.Variable.Instance:
                    arranged_token = "<INSTANCE_VAL>"
                elif token_type == Name.Variable.Class:
                    arranged_token = "<CLASS_VAL>"
                elif token_type == Name.Constant:
                    arranged_token = "<CONSTANT_ID>"
                elif token_type == Name.Function:
                    arranged_token = "<FUNCTION>"
                elif token_type == Name.Class:
                    arranged_token = "<CLASS>"
                elif token_type == Name.Namespace:
                    arranged_token = "<NAMESPACE>"
                elif token_type == Token.Name.Variable.Global:
                    arranged_token = "<GLOBAL_VAL>"
                elif token_type == Token.Error:
                    arranged_token = "<ERROR>"  # pygments内で字句解析が失敗した際のトークン (絵文字など)
                elif is_token_subtype(token_type, Comment):
                    arranged_token = "<COMMENT>"
                else:
                    arranged_token = replace_special_char(token)
                    # if arranged_token not in reserved and "SPACE" not in arranged_token and "NEWLINE" not in arranged_token:
                    #     if token_type not in (Token.Punctuation, Token.Operator, Token.Name.Builtin, Token.Keyword.Pseudo):
                    #         print("==============")
                    #         print(program_path)
                    #         print(line.rstrip())
                    #         print("{} : {}".format(arranged_token.encode("utf-8"), token_type))
                    #         print("==============")

            line_of_token.append(arranged_token + " ") # 空白区切りにするため、最後にスペースをつける
            
        # 行頭の空白二つはインデントとみなす
        line_of_token[0] = line_of_token[0].replace("<SPACE> <SPACE> ", "<INDENT> ")

        # インデントは前の行との相対的な値を番号として付与する
        indent_count = len(re.findall("<INDENT>", line_of_token[0]))

        if indent_count != 0:
            # 空行がインデントされていると0番目の要素にインデントと改行が両方含まれている場合があるため、
            # インデント情報を取り除いてから、相対的なインデント情報を付け加える
            indent_char = "<INDENT{}> ".format(indent_count - last_indent_count)
            line_of_token[0] = line_of_token[0].replace("<INDENT> ", "")
            line_of_token[0] = indent_char + line_of_token[0]

        if len(line_of_token) != 1:
            last_indent_count = indent_count

        token_streams.append(line_of_token)

    return token_streams, num_of_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--win_size", required=True, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=argparse.FileType("w"))
    args = parser.parse_args()

    print "[info] Start tokenizing"

    program_path_list = find_all_files(args.source)
    program_path_list = filter(lambda p: os.path.splitext(p)[1] == ".rb",
                               program_path_list)

    print "[info] Num of files: {}".format(len(program_path_list))

    tokenized_program_list = []

    sum_of_lines = 0
    for program_path in program_path_list:
        token_streams, num_of_lines = tokenize(program_path)
        tokenized_program_list.append(token_streams)
        sum_of_lines += num_of_lines

    print "[info] Num of all lines: {}".format(sum_of_lines)

    win_size = args.win_size
    output = args.output

    num_of_tokens = 0
    num_of_windows = 0
    for tokenized_program in tokenized_program_list:
        for index in xrange(len(tokenized_program) - win_size + 1):
            for line in tokenized_program[index:index+win_size]:
                for token_data in line:
                    output.write(token_data.encode("utf-8"))
                    num_of_tokens += 1

            output.write("\n")
            num_of_windows += 1

    print "[info] Num of tokens: {}".format(num_of_tokens)
    print "[info] Num of windows: {}".format(num_of_windows)
