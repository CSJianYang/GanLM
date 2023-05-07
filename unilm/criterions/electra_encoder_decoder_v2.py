# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from unilm.criterions.unilm import UniLmConfig


@dataclass
class ElectraV2Config(UniLmConfig):
    mlm_only: bool = field(
        default=False, metadata={"help": "use mlm objective only"}
    )
    seq2seq_only: bool = field(
        default=False, metadata={"help": "use seq2seq objective only"}
    )
    discriminator_weight: float = field(
        default=50.0, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    discriminator_warmup_steps: float = field(
        default=-1, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    tpu: bool = II("common.tpu")


def mask_tokens(i):
    return f"<mask_{i}>"


@register_criterion("electra_encoder_decoder_v2", dataclass=ElectraV2Config)
class ElectraEncoderDecoderV2Loss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.cfg = cfg
        self.mask_idx = task.mask_idx
        self.mask_span_idx = task.mask_idx + 1
        self.warmup_interval = cfg.discriminator_warmup_steps // cfg.discriminator_weight

    def seq2seq_loss(self, model, sample, reduce, update_num=-1):
        src_lengths = sample["src_lengths"]
        targets = sample["targets"]
        masked_tokens = targets.le(self.mask_idx - 1) & targets.ne(self.padding_idx)
        decoder_out = model(src_tokens=sample["src_tokens"], src_lengths=src_lengths,
                            prev_output_tokens=sample["tgt_tokens"], masked_tokens=masked_tokens,
                            targets=sample["targets"], features_only=False)
        sample_size = targets.ne(self.padding_idx).int().sum().detach().tolist()

        generator_logits = decoder_out["generator_out"][0]
        generator_loss = modules.cross_entropy(
            generator_logits.view(-1, generator_logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        discriminator_logits = decoder_out["discriminator_out"][0]
        input_tokens = decoder_out["input_tokens"]
        discriminator_targets = targets.eq(input_tokens).long()

        non_pad_tokens = targets.le(self.mask_idx - 1) & targets.ne(self.padding_idx)
        discriminator_logits = discriminator_logits[non_pad_tokens]
        discriminator_targets = discriminator_targets[non_pad_tokens]
        discriminator_sample_size = non_pad_tokens.int().sum().detach().tolist()

        discriminator_loss = F.nll_loss(
            F.log_softmax(
                discriminator_logits.view(-1, discriminator_logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            discriminator_targets.view(-1),
            reduction='sum',
            ignore_index=-100,
        )
        discriminator_loss = discriminator_loss * sample_size / discriminator_sample_size

        discriminator_weight = min(update_num // self.warmup_interval,
                                   self.cfg.discriminator_weight) if self.warmup_interval > 0 else self.cfg.discriminator_weight
        loss = generator_loss + discriminator_weight * discriminator_loss
        with torch.no_grad():
            positive = discriminator_logits[:, 1] >= discriminator_logits[:, 0]
            negative = discriminator_logits[:, 1] < discriminator_logits[:, 0]
            tp = (positive & (discriminator_targets == 1)).long().sum()
            fp = (positive & (discriminator_targets == 0)).long().sum()
            tn = (negative & (discriminator_targets == 0)).long().sum()
            fn = (negative & (discriminator_targets == 1)).long().sum()
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "seq2seq_loss": loss if self.tpu else loss.data,
            "generator_loss": generator_loss if self.tpu else generator_loss.data,
            "discriminator_loss": discriminator_loss if self.tpu else discriminator_loss.data * self.cfg.discriminator_weight,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "tp": tp.data,
            "fp": fp.data,
            "tn": tn.data,
            "fn": fn.data,
            "discriminator_weight": discriminator_weight,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True, epoch=-1, update_num=-1):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, sample_size, logging_outputs = 0, 0, {}
        seq2seq_loss, seq2seq_sample_size, seq2seq_logging_output = self.seq2seq_loss(
            model, sample["seq2seq"], reduce=reduce, update_num=update_num
        )
        logging_outputs.update(seq2seq_logging_output)
        loss += seq2seq_loss
        sample_size = seq2seq_sample_size
        return loss, sample_size, logging_outputs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        if "mlm_loss" in logging_outputs[0]:
            mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "mlm_loss", mlm_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if "seq2seq_loss" in logging_outputs[0]:
            seq2seq_loss_sum = sum(log.get("seq2seq_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "seq2seq_loss", seq2seq_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if "discriminator_loss" in logging_outputs[0] and "generator_loss" in logging_outputs[0]:
            generator_loss_sum = sum(log.get("generator_loss", 0) for log in logging_outputs)
            discriminator_loss_sum = sum(log.get("discriminator_loss", 0) for log in logging_outputs)
            tp = sum(log.get("tp", 0) for log in logging_outputs)
            fp = sum(log.get("fp", 0) for log in logging_outputs)
            tn = sum(log.get("tn", 0) for log in logging_outputs)
            fn = sum(log.get("fn", 0) for log in logging_outputs)
            replace_acc = tn / (tn + fp + 1e-5)
            non_replace_acc = tp / (tp + fn + 1e-5)
            replace_rate = (tn + fp) / (tn + fp + tp + fn + 1e-5)
            metrics.log_scalar(
                "generator_loss", generator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "discriminator_loss", discriminator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "replace_acc", replace_acc, tn + fp, round=4
            )
            metrics.log_scalar(
                "non_replace_acc", non_replace_acc, tp + fn, round=4
            )
            metrics.log_scalar(
                "replace_rate", replace_rate, tn + fp + tp + fn, round=4
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
