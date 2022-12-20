#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from fairseq.data import Dictionary, data_utils, indexed_dataset
import numpy as np
import linecache
import os
def get_parser():
    parser = argparse.ArgumentParser(
        description="writes text from binarized file to stdout"
    )
    # fmt: off
    parser.add_argument('--dataset-impl', help='dataset implementation',
                        choices=indexed_dataset.get_available_dataset_impl())
    parser.add_argument('--dict', metavar='FP', default="/home/v-jiaya/unilm-moe/data/wmt16-en-ro/data-bin/dict.en.txt", help='dictionary containing known words')
    parser.add_argument('--input', metavar='FP', default="/home/v-jiaya/unilm-moe/data/wmt16-en-ro/train/", help='binarized file to read')
    parser.add_argument('--src', default="en", help='')
    parser.add_argument('--tgt', default="ro", help='')
    parser.add_argument('--output', metavar='FP', default="/home/v-jiaya/unilm-moe/data/wmt16-en-ro/ablation/", help='binarized file to read')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    lang_pair = f"{args.src}-{args.tgt}"
    for lg in [args.src, args.tgt]:
        input_file = f"{args.input}/train.{lang_pair}.{lg}"
        lines = linecache.getlines(input_file)
        for rate in np.arange(0.1, 1, 0.1):
            output_lines = lines[: int(len(lines) * rate)]
            rate = round(rate, 1)
            output_file = f"{args.output}/{rate}/train/train.{lang_pair}.{lg}"
            if not os.path.exists(f"{args.output}/{rate}/train/"):
                os.makedirs(f"{args.output}/{rate}/train/")
            with open(output_file, "w", encoding="utf-8") as w:
                for line in output_lines:
                    w.write(line)
            print(f"Successfully Saving to {output_file}: {len(output_lines)}")






if __name__ == "__main__":
    main()
