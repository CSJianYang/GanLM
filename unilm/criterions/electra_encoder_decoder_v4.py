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
class ElectraV4Config(UniLmConfig):
    mlm_only: bool = field(
        default=False, metadata={"help": "use mlm objective only"}
    )
    seq2seq_only: bool = field(
        default=False, metadata={"help": "use seq2seq objective only"}
    )
    discriminator_start_warmup_steps: float = field(
        default=0, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    discriminator_weight: float = field(
        default=25.0, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    inconsistency_start_warmup_steps: float = field(
        default=0, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    inconsistency_weight: float = field(
        default=1.0, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    discriminator_warmup_steps: float = field(
        default=-1, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    inconsistency_warmup_steps: float = field(
        default=-1, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    tpu: bool = II("common.tpu")


def mask_tokens(i):
    return f"<mask_{i}>"


@register_criterion("electra_encoder_decoder_v4", dataclass=ElectraV4Config)
class ElectraEncoderDecoderV4Loss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.cfg = cfg
        self.mask_idx = task.mask_idx
        self.eos_idx = task.dictionary.eos()
        self.mask_span_idx = task.mask_idx + 1
        self.discriminator_warmup_interval = cfg.discriminator_warmup_steps / cfg.discriminator_weight
        self.inconsistency_warmup_interval = cfg.inconsistency_warmup_steps / cfg.inconsistency_weight

    def get_loss_weight(self, update_num, warmup_interal, weight, start_warmup_steps, return_float=True):
        if update_num < start_warmup_steps:
            return 0
        if warmup_interal < 0:
            return weight
        update_num = update_num - start_warmup_steps
        if return_float:
            return min(update_num / warmup_interal, weight) if warmup_interal > 0 else weight
        else:
            return min(update_num // warmup_interal, weight) if warmup_interal > 0 else weight


    def seq2seq_loss(self, model, sample, reduce, update_num=-1):
        src_lengths = sample["src_lengths"]
        targets = sample["targets"]
        masked_tokens = targets.le(self.mask_idx - 1) & targets.ne(self.padding_idx)
        decoder_out = model(src_tokens=sample["src_tokens"], src_lengths=src_lengths, prev_output_tokens=sample["tgt_tokens"], masked_tokens=masked_tokens, targets=sample["targets"], features_only=False)
        sample_size = targets.ne(self.padding_idx).int().sum().detach().tolist()

        generator_logits = decoder_out["generator_out"][0]
        generator_loss = modules.cross_entropy(
            generator_logits.view(-1, generator_logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        discriminator_weight = self.get_loss_weight(update_num, self.discriminator_warmup_interval, self.cfg.discriminator_weight, start_warmup_steps=self.cfg.discriminator_start_warmup_steps, return_float=True)
        inconsistency_weight = self.get_loss_weight(update_num, self.inconsistency_warmup_interval, self.cfg.inconsistency_weight, start_warmup_steps=self.cfg.inconsistency_start_warmup_steps, return_float=True)

        discriminator_logits = decoder_out["discriminator_out"][0]
        input_tokens = decoder_out["input_tokens"]
        discriminator_targets = targets.eq(input_tokens).long()

        non_pad_tokens = masked_tokens
        non_pad_discriminator_logits = discriminator_logits[non_pad_tokens]
        non_pad_discriminator_targets = discriminator_targets[non_pad_tokens]
        discriminator_sample_size = non_pad_tokens.int().sum().detach().tolist()

        # discriminator_loss = modules.cross_entropy(
        #     non_pad_discriminator_logits.view(-1, non_pad_discriminator_logits.size(-1)),
        #     non_pad_discriminator_targets.view(-1),
        #     reduction="sum",
        #     ignore_index=-100,
        # )
        if discriminator_weight > 0:
            discriminator_loss = F.nll_loss(
                F.log_softmax(
                    non_pad_discriminator_logits.view(-1, non_pad_discriminator_logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                non_pad_discriminator_targets.view(-1),
                reduction='sum',
                ignore_index=-100,
            )
            discriminator_loss = discriminator_loss * sample_size / discriminator_sample_size
            with torch.no_grad():
                positive = non_pad_discriminator_logits[:, 1] >= non_pad_discriminator_logits[:, 0]
                negative = non_pad_discriminator_logits[:, 1] < non_pad_discriminator_logits[:, 0]
                tp = (positive & (non_pad_discriminator_targets == 1)).long().sum()
                fp = (positive & (non_pad_discriminator_targets == 0)).long().sum()
                tn = (negative & (non_pad_discriminator_targets == 0)).long().sum()
                fn = (negative & (non_pad_discriminator_targets == 1)).long().sum()
        else:
            discriminator_loss = generator_loss.new([0])
            tp = targets.new([0])
            fp = targets.new([0])
            tn = targets.new([0])
            fn = targets.new([0])
        #
        if inconsistency_weight > 0:
            # predicted_tokens != target_tokens && predict original token
            non_pad_mask_tokens = non_pad_tokens & sample["non_last_token_tgt"].ne(self.padding_idx) & input_tokens.le(self.mask_idx - 1)
            discriminator_predicts = discriminator_logits.max(dim=-1)[1].bool()
            negative_inconsistency_index = (~discriminator_targets.bool()) & discriminator_predicts & non_pad_mask_tokens
            positive_inconsistency_index = discriminator_targets.bool() & (~discriminator_predicts) & non_pad_mask_tokens
            negative_inconsistency_ntokens = negative_inconsistency_index.sum()
            positive_inconsistency_ntokens = positive_inconsistency_index.sum()
            inconsistency_ntokens = negative_inconsistency_ntokens + positive_inconsistency_ntokens
            inconsistency_tgt_tokens = sample["tgt_tokens"].clone()
            inconsistency_tgt_tokens[:, 1:][negative_inconsistency_index[:, :-1]] = input_tokens[negative_inconsistency_index]
            inconsistency_tgt_tokens[:, 1:][positive_inconsistency_index[:, :-1]] = self.mask_idx
            inconsistency_decoder_out = model.decoder(encoder_out=decoder_out["encoder_out"], prev_output_tokens=inconsistency_tgt_tokens, masked_tokens=masked_tokens, targets=sample["targets"], features_only=True)
            inconsistency_generator_logits = inconsistency_decoder_out[0]
            inconsistency_generator_loss = modules.cross_entropy(
                inconsistency_generator_logits.view(-1, inconsistency_generator_logits.size(-1)),
                targets.view(-1),
                reduction="sum",
                ignore_index=self.padding_idx,
            )
        else:
            inconsistency_ntokens = 0
            negative_inconsistency_ntokens = 0
            positive_inconsistency_ntokens = 0
            inconsistency_generator_loss = generator_loss.new([0])
        loss = generator_loss + discriminator_weight * discriminator_loss + inconsistency_weight * inconsistency_generator_loss
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "seq2seq_loss": loss if self.tpu else loss.data,
            "generator_loss": generator_loss if self.tpu else generator_loss.data,
            "discriminator_loss": discriminator_loss * self.cfg.discriminator_weight if self.tpu else discriminator_loss.data * self.cfg.discriminator_weight,
            "inconsistency_generator_loss": inconsistency_generator_loss * self.cfg.inconsistency_weight if self.tpu else inconsistency_generator_loss.data * self.cfg.inconsistency_weight,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "tp": tp.data,
            "fp": fp.data,
            "tn": tn.data,
            "fn": fn.data,
            "discriminator_weight": discriminator_weight,
            "inconsistency_weight": inconsistency_weight,
            "inconsistency_ntokens": inconsistency_ntokens,
            "positive_inconsistency_ntokens": positive_inconsistency_ntokens,
            "negative_inconsistency_ntokens": negative_inconsistency_ntokens,
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
            inconsistency_generator_loss_sum = sum(log.get("inconsistency_generator_loss", 0) for log in logging_outputs)
            tp = sum(log.get("tp", 0) for log in logging_outputs)
            fp = sum(log.get("fp", 0) for log in logging_outputs)
            tn = sum(log.get("tn", 0) for log in logging_outputs)
            fn = sum(log.get("fn", 0) for log in logging_outputs)
            negative_inconsistency_ntokens = sum(log.get("negative_inconsistency_ntokens", 0) for log in logging_outputs)
            positive_inconsistency_ntokens = sum(log.get("positive_inconsistency_ntokens", 0) for log in logging_outputs)
            inconsistency_ntokens = sum(log.get("inconsistency_ntokens", 0) for log in logging_outputs)
            replace_acc = tn / (tn + fp + 1e-5)
            non_replace_acc = tp / (tp + fn + 1e-5)
            replace_rate = (tn + fp) / (tn + fp + tp + fn + 1e-5)
            negative_inconsistency_rate = negative_inconsistency_ntokens / (tn + fp + tp + fn + 1e-5)
            positive_inconsistency_rate = positive_inconsistency_ntokens / (tn + fp + tp + fn + 1e-5)
            inconsistency_rate = inconsistency_ntokens / (tn + fp + tp + fn + 1e-5)
            metrics.log_scalar(
                "generator_loss", generator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "discriminator_loss", discriminator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "inconsistency_generator_loss", inconsistency_generator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "generator_loss", generator_loss_sum / sample_size / math.log(2), sample_size, round=3
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
            metrics.log_scalar(
                "negative_inconsistency_rate", negative_inconsistency_rate, tn + fp + tp + fn, round=4
            )
            metrics.log_scalar(
                "positive_inconsistency_rate", positive_inconsistency_rate, tn + fp + tp + fn, round=4
            )
            metrics.log_scalar(
                "inconsistency_rate", inconsistency_rate, tn + fp + tp + fn, round=4
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
