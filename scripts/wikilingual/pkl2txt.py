import argparse
import os
import io
import sys
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/WikiLingua/indonesian.pkl', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/home/v-jiaya/mBART/data/WikiLingual/download/raw/',  help='output stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'en', help='output stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    random.seed(10086)
    args = parse_args()
    document = []
    summary = []
    with open(args.input, 'rb') as input:
        data = pickle.load(input)
        for web in data.values():
            for article in web:
                document.append(web[article]["document"].replace("\n"," ").replace("  "," ").strip())
                summary.append(web[article]["summary"].replace("\n"," ").replace("  "," ").strip())
    assert len(document) == len(summary)
    document_summary = list(zip(document, summary))
    random.shuffle(document_summary)
    TEST_NUMBER = 500
    with open("{}/train.{}_doc".format(args.output, args.lang), 'w', encoding="utf-8") as output1:
        with open("{}/train.{}_sum".format(args.output, args.lang), 'w', encoding="utf-8") as output2:
            for i in range(len(document_summary) - TEST_NUMBER * 2):
                output1.write("{}\n".format(document_summary[i][0]))
                output2.write("{}\n".format(document_summary[i][1]))
    with open("{}/valid.{}_doc".format(args.output, args.lang), 'w', encoding="utf-8") as output1:
        with open("{}/valid.{}_sum".format(args.output, args.lang), 'w', encoding="utf-8") as output2:
            for i in range(len(document_summary) - TEST_NUMBER * 2, len(document_summary) - TEST_NUMBER * 1):
                output1.write("{}\n".format(document_summary[i][0]))
                output2.write("{}\n".format(document_summary[i][1]))
    with open("{}/test.{}_doc".format(args.output, args.lang), 'w', encoding="utf-8") as output1:
        with open("{}/test.{}_sum".format(args.output, args.lang), 'w', encoding="utf-8") as output2:
            for i in range(len(document_summary) - TEST_NUMBER * 1, len(document_summary)):
                output1.write("{}\n".format(document_summary[i][0]))
                output2.write("{}\n".format(document_summary[i][1]))



