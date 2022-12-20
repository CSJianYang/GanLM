import argparse
import nltk
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation-str', '-generation-str', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/train_raw/test.ru_RU_sum', help='input stream')
    parser.add_argument('--reference-str', '-reference-str', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/train_raw/test.ru_RU_sum',  help='output stream')
    parser.add_argument('--generation-id', '-generation-id', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/train_raw/test.ru_RU_sum.id', help='input stream')
    parser.add_argument('--reference-id', '-reference-id', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/train_raw/test.ru_RU_sum.id', help='output stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.generation_str, 'r', encoding="utf-8") as input1:
        with open(args.reference_str, 'r', encoding="utf-8") as input2:
            with open(args.generation_id, 'w', encoding="utf-8") as output1:
                with open(args.reference_id, 'w', encoding="utf-8") as output2:
                    for line1, line2 in zip(input1, input2):
                        tokens1 = line1.strip().split()
                        tokens2 = line2.strip().split()
                        tokens = list(set(tokens1 + tokens2))
                        id_dict = {tokens[i]:str(i) for i in range(len(tokens))}
                        ids1 = " ".join([id_dict[token] for token in tokens1])
                        ids2 = " ".join([id_dict[token] for token in tokens2])
                        output1.write("{}\n".format(ids1))
                        output2.write("{}\n".format(ids2))





