import argparse
import nltk
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/train_raw/test.ru_RU_sum', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/train_raw/test.ru_RU_sum.tok',  help='output stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'ru', help='output stream', choices=['zh', 'ar', 'ru', "ja"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.input, 'r', encoding="utf-8") as input:
        with open(args.output, 'w', encoding="utf-8") as output:
            for line in input:
                if args.lang == "zh" or args.lang == "ja":
                    tokenized_line = [token for token in line.strip()]
                    tokenized_line = " ".join(tokenized_line).replace("    ", " ").replace("  ", " ").replace("   ", " ").replace("  ", " ")
                    output.write("{}\n".format(tokenized_line))
                elif args.lang == "ru":
                    tokenized_line = nltk.word_tokenize(line.strip())
                    tokenized_line = " ".join(tokenized_line).replace("    ", " ").replace("  ", " ").replace("   ", " ").replace("  ", " ")
                    output.write("{}\n".format(tokenized_line))
                elif args.lang == "ar":
                    tokenized_line = nltk.word_tokenize(line.strip())
                    tokenized_line = " ".join(tokenized_line).replace("    ", " ").replace("  ", " ").replace("   ", " ").replace("  ", " ")
                    output.write("{}\n".format(tokenized_line))



