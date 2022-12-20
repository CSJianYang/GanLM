# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
import logging
logger = logging.getLogger(__name__)

@register_model("transformer_from_xlmr")
class TransformerFromPretrainedXLMRModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into encoder",
        )
        parser.add_argument(
            "--shared-cross-attn",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into encoder",
        )

    def upgrade_xlmr_state_for_encoder(
            self, state_dict: Dict[str, Any], xlmr_state_dict: str, num_layers: int, shared_cross_attn: bool = False, prefix="decoder.sentence_encoder.", new_prefix="encoder."
    ) -> Dict[str, Any]:
        for key in xlmr_state_dict.keys():
            if 'layers' in key and int(key.split('.')[3]) > num_layers - 1:
                continue
            if not key.startswith('decoder.'):
                continue
            if 'lm_head' not in key:
                if 'in_proj_weight' in key:
                    q, k, v = xlmr_state_dict[key].chunk(3, dim=0)
                    state_dict[key.replace(prefix, new_prefix).replace('in_proj_weight', 'q_proj.weight')] = q
                    state_dict[key.replace(prefix, new_prefix).replace('in_proj_weight', 'k_proj.weight')] = k
                    state_dict[key.replace(prefix, new_prefix).replace('in_proj_weight', 'v_proj.weight')] = v
                    if shared_cross_attn:
                        state_dict[key.replace(prefix, new_prefix).replace('in_proj_weight', 'q_proj.weight').replace('self_attn', 'encoder_attn')] = q
                        state_dict[key.replace(prefix, new_prefix).replace('in_proj_weight', 'k_proj.weight').replace('self_attn', 'encoder_attn')] = k
                        state_dict[key.replace(prefix, new_prefix).replace('in_proj_weight', 'v_proj.weight').replace('self_attn', 'encoder_attn')] = v
                elif 'in_proj_bias' in key:
                    q, k, v = xlmr_state_dict[key].chunk(3, dim=0)
                    state_dict[key.replace(prefix, new_prefix).replace('in_proj_bias', 'q_proj.bias')] = q
                    state_dict[key.replace(prefix, new_prefix).replace('in_proj_bias', 'k_proj.bias')] = k
                    state_dict[key.replace(prefix, new_prefix).replace('in_proj_bias', 'v_proj.bias')] = v
                    if shared_cross_attn:
                        state_dict[key.replace(prefix, new_prefix).replace('in_proj_bias', 'q_proj.bias').replace('self_attn', 'encoder_attn')] = q
                        state_dict[key.replace(prefix, new_prefix).replace('in_proj_bias', 'k_proj.bias').replace('self_attn', 'encoder_attn')] = k
                        state_dict[key.replace(prefix, new_prefix).replace('in_proj_bias', 'v_proj.bias').replace('self_attn', 'encoder_attn')] = v
                elif 'emb_layer_norm' in key:
                    state_dict[key.replace(f'{prefix}emb_layer_norm', f'{new_prefix}layernorm_embedding')] = xlmr_state_dict[key]
                elif 'embed_positions' in key:
                    state_dict[key.replace(prefix, new_prefix)] = xlmr_state_dict[key][:state_dict[key.replace(prefix, new_prefix)].size(0)]
                elif 'embed_tokens' in key:
                    state_dict[key.replace(prefix, new_prefix)][:xlmr_state_dict[key].size(0)] = xlmr_state_dict[key]
                else:
                    state_dict[key.replace(prefix, new_prefix)] = xlmr_state_dict[key]

        return state_dict



    def upgrade_state_dict_named(self, state_dict, name):
        def upgrade_position_embed(k, cur_state_dict, state_dict, prefix, new_prefix="encoder", max_positions=-1):
            if max_positions == -1:
                max_positions = cur_state_dict[k.replace(prefix, new_prefix)].size(0)
            assert k.replace(prefix, new_prefix) in cur_state_dict.keys()
            if max_positions < state_dict[k].size(0):
                logger.info(f"{k} | Clipping {state_dict[k].size(0)} -> {max_positions} positions (start from 2th pos)")
                cur_state_dict[k.replace(prefix, new_prefix)] = state_dict[k][:max_positions, :]
            else:
                logger.info(f"{k} | Clipping {max_positions} -> {state_dict[k].size(0)} positions (start from 2th pos)")
                cur_state_dict[k.replace(prefix, new_prefix)][: state_dict[k].size(0)] = state_dict[k]

        cur_state_dict = self.state_dict()
        shared_cross_attn = getattr(self.args, "shared_cross_attn", False)
        if "decoder.sentence_encoder.embed_tokens.weight" in state_dict:  # XLM-R State
            if getattr(self.args, "init_encoder_only", False):
                cur_state_dict = self.upgrade_xlmr_state_for_encoder(cur_state_dict, xlmr_state_dict=state_dict, num_layers=self.args.encoder_layers, prefix="decoder.sentence_encoder.", new_prefix="encoder.")
                logger.info(f"Loading XLM-R for Encoder of {self.args.arch}")
            if getattr(self.args, "init_decoder_only", False):
                cur_state_dict = self.upgrade_xlmr_state_for_encoder(cur_state_dict, xlmr_state_dict=state_dict, num_layers=self.args.decoder_layers, prefix="decoder.sentence_encoder.", new_prefix="decoder.", shared_cross_attn=shared_cross_attn)
                logger.info(f"Loading XLM-R for Decoder of {self.args.arch} (self-attn = cross-attn)")
            state_dict.clear()
            for k, v in cur_state_dict.items():
                state_dict[k] = v
            # #Adjust Position Embedding
            # if "encoder.embed_positions.weight" in state_dict.keys() and self.args.encoder_learned_pos:
            #     upgrade_position_embed("encoder.embed_positions.weight", state_dict, state_dict, prefix="encoder", new_prefix="encoder", max_positions=self.args.max_source_positions + 2)
            # if "decoder.embed_positions.weight" in state_dict.keys() and self.args.decoder_learned_pos:
            #     upgrade_position_embed("decoder.embed_positions.weight", state_dict, state_dict, prefix="decoder", new_prefix="decoder", max_positions=self.args.max_target_positions + 2)
        else:
            logger.info(f"Directly Loading Checkpoint without Any Change")
        return state_dict


@register_model_architecture("transformer_from_xlmr", "transformer_from_xlmr")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("transformer_from_xlmr", "transformer_from_xlmr_base")
def transformer_from_xlmr(args):
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("transformer_from_xlmr", "transformer_from_xlmr_large")
def large_electra_encoder_decoder_architecture(args):
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    base_architecture(args)