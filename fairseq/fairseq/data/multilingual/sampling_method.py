# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List


logger = logging.getLogger(__name__)


def uniform(dataset_sizes: List[int]):
    return [1.0] * len(dataset_sizes)


def temperature_sampling(dataset_sizes, temp):
    total_size = sum(dataset_sizes)
    return [(size / total_size) ** (1.0 / temp) for size in dataset_sizes]


def make_temperature_sampling(temp=1.0):
    def sampling_func(dataset_sizes):
        return temperature_sampling(dataset_sizes, temp)

    return sampling_func


def make_linear_temperature_sampling(temp=5.0, min_temp=1.0, warmup=5):
    def sampling_func(dataset_sizes, epoch: int = 1):
        t = min((temp - min_temp) * (epoch - 1) / warmup + min_temp, temp)
        if t == 0:
            t = 1
        return temperature_sampling(dataset_sizes, t)

    return sampling_func


def make_ratio_sampling(ratios):
    def sampling_func(dataset_sizes):
        return ratios

    return sampling_func


class SamplingMethod:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--sampling-method",
            choices=[
                "uniform",
                "temperature",
                "concat",
                "RoundRobin",
                "linear",
            ],
            type=str,
            default="concat",
            help="The method to sample data per language pairs",
        )
        parser.add_argument(
            "--sampling-temperature",
            default=1.5,
            type=float,
            help="only work with --sampling-method temperature",
        )
        parser.add_argument(
            "--min-sampling-temperature",
            default=1.0,
            type=float,
            help="only work with --sampling-method linear",
        )
        parser.add_argument(
            "--warmup-epoch",
            default=5,
            type=int,
            help="only work with --sampling-method linear",
        )

    @staticmethod
    def build_sampler(args, task):
        return SamplingMethod(args, task)

    def __init__(self, args, task):
        self.args = args
        self.task = task

    def is_adaptive(self):
        return False

    def sampling_method_selector(self):
        args = self.args
        #logger.info(f"selected sampler: {args.sampling_method}")
        if args.sampling_method == "uniform":
            return uniform
        elif args.sampling_method == "temperature" or self.is_adaptive():
            return make_temperature_sampling(float(args.sampling_temperature))
        elif args.sampling_method == "linear":
            return make_linear_temperature_sampling(float(args.sampling_temperature), float(args.min_sampling_temperature), int(args.warmup_epoch))
        else:
            # default to concating all data set together
            return None
