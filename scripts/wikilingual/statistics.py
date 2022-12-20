import argparse
import os
import io
import sys
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/mnt/input/mBART/WikiLingual/download-split/train_spm/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/mnt/input/mBART/WikiLingual/download-split/statistics.txt', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    random.seed(10086)
    args = parse_args()
    MAX_LEN = 1024
    output_infos = []
    if os.path.isdir(args.input):
        files = os.listdir(args.input)
        files.sort()
        for file in files:
            exceed_lengths = []
            with open(os.path.join(args.input, file), "r", encoding="utf-8") as input:
                lens = []
                lines = input.readlines()
                max_len = -1
                min_len = 1e8
                for line in lines:
                    length = len(line.split())
                    max_len = length if length > max_len else max_len
                    min_len = length if length < min_len else min_len
                    if length > MAX_LEN:
                        exceed_lengths.append((file, len(line.split())))
                    lens.append(len(line.split()))
                output_info = "{} | Average Length: {} | Exceed Examples {} | min_len {} | max_len {}".format(file, int(sum(lens)/len(lens)), len(exceed_lengths), min_len, max_len)
                output_infos.append(output_info)
                print(output_info)
    else:
        with open(args.input, 'r', encoding="utf-8") as input:
            lens = []
            lines = input.readlines()
            for line in lines:
                length = len(line.split())
                lens.append(length)
            output_info = "{} | Average Length: {} | Exceed Examples {} | min_len {} | max_len {}".format(file, int(sum(lens) / len(lens)), len(exceed_lengths), min_len, max_len)
            output_infos.append(output_info)
            print(output_info)

    with open(args.output, "w", encoding="utf-8") as w:
        w.write("\n".join(output_infos))
        print("Successfully saving statistics to {}".format(args.output))


