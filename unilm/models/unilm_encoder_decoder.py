from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
import math
import torch
import torch.nn as nn
import copy
from typing import Dict, List, Optional, Tuple
from fairseq import utils
import logging
from torch import Tensor
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    TransformerModelBase,
    TransformerConfig,
    Embedding,
    Linear,
    module_name_fordropout
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from unilm.models.unilm import UniLMBody, UniLMModelConfig, base_unilm_architecture

logger = logging.getLogger(__name__)


class UniLMEncoder(UniLMBody):
    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)

    def forward(
            self,
            src_tokens,
            tgt_tokens=None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            return_all_hiddens: bool = False,
            src_lengths=None
    ):
        # embed positions
        positions = None
        if self.embed_positions is not None:
            if src_tokens is not None:
                src_positions = self.embed_positions(
                    src_tokens, incremental_state=incremental_state
                )
            if tgt_tokens is not None:
                tgt_positions = self.embed_positions(
                    tgt_tokens, incremental_state=incremental_state
                )

        tokens, self_attn_mask = None, None

        if src_tokens is not None and tgt_tokens is not None:
            tokens = torch.cat([src_tokens, tgt_tokens], dim=1)
            self_attn_mask = self.build_seq2seq_attn_mask(src_tokens, tgt_tokens)
            if self.embed_positions is not None:
                positions = torch.cat([src_positions, tgt_positions], dim=1)
        elif src_tokens is not None:
            tokens = src_tokens
            self_attn_mask = self.build_self_attn_mask(src_tokens, bidirectional=True)
            if self.embed_positions is not None:
                positions = src_positions
        else:
            tokens = tgt_tokens
            self_attn_mask = self.build_self_attn_mask(tgt_tokens, bidirectional=False)
            if self.embed_positions is not None:
                positions = tgt_positions

        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=tokens.size(0),
                qlen=tokens.size(1),
                klen=tokens.size(1)
            )
            self_attn_mask = self_attn_mask.unsqueeze(0) + rel_pos_bias

        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = tokens.eq(self.padding_idx)

        # encoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return {
            "encoder_out": [x.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [self_attn_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": inner_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class UniLMDecoder(TransformerDecoderBase):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self,
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        self.cfg = cfg
        super(TransformerDecoderBase, self).__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)


@dataclass
class UniEncoderDecoderConfig(FairseqDataclass):
    initialization_strategy: ChoiceEnum(["mae_encoder2decoder", "mae_decoder2decoder", "mlm_encoder2decoder"]) = field(
        default=None, metadata={"help": "initialization strategy"}
    )


@register_model("unilm_encoder_decoder")
class UniLMEncoderDecoder(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super(FairseqEncoderDecoderModel, self).__init__()
        self.args = args
        self.cfg = TransformerConfig.from_namespace(args)
        self.supports_align_args = True
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        # parser.add_argument("")
        gen_parser_from_dataclass(
            parser, UniLMModelConfig(), delete_default=True, with_prefix=""
        )
        gen_parser_from_dataclass(
            parser, UniEncoderDecoderConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_unilm_encoder_decoder_architecture(args)

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        args.decoder_input_dim = int(args.decoder_input_dim)
        args.decoder_output_dim = int(args.decoder_output_dim)
        # --

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if args.offload_activations:
            args.checkpoint_activations = True  # offloading implies checkpointing
        cfg = TransformerConfig.from_namespace(args)
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=args.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=args.min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        wrapper_cfg = copy.deepcopy(cfg)
        wrapper_cfg.max_target_positions = wrapper_cfg.max_source_positions
        return UniLMEncoder(wrapper_cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return UniLMDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        def upgrade_embed_tokens(k, cur_state_dict, state_dict):
            assert "encoder.embed_tokens.weight" in cur_state_dict.keys()
            assert "decoder.embed_tokens.weight" in cur_state_dict.keys()
            assert "decoder.output_projection.weight" in cur_state_dict.keys()
            cur_state_dict["encoder.embed_tokens.weight"] = state_dict[k]
            cur_state_dict["decoder.embed_tokens.weight"] = state_dict[k]
            cur_state_dict["decoder.output_projection.weight"] = state_dict[k]

        def upgrade_position_embed(k, cur_state_dict, state_dict, prefix, new_prefix="encoder", max_positions=-1):
            if max_positions == -1:
                max_positions = cur_state_dict[k.replace(prefix, new_prefix)].size(0)
            assert k.replace(prefix, new_prefix) in cur_state_dict.keys()
            if max_positions < state_dict[k].size(0):
                logger.info(f"Clipping {state_dict[k].size(0)} -> {max_positions} positions (start from 2th pos)")
                cur_state_dict[k.replace(prefix, new_prefix)] = state_dict[k][:max_positions, :]
            else:
                logger.info(f"Clipping {max_positions} -> {state_dict[k].size(0)} positions (start from 2th pos)")
                cur_state_dict[k.replace(prefix, new_prefix)][: state_dict[k].size(0)] = state_dict[k]

        def upgrade_layer(k, cur_state_dict, state_dict, prefix, new_prefix="encoder"):
            if k.replace(prefix, new_prefix) not in cur_state_dict.keys():
                logger.info(f"Missing Keys: {k}")
            assert k.replace(prefix, new_prefix) in cur_state_dict.keys()
            cur_state_dict[k.replace(prefix, new_prefix)] = state_dict[k]

        cur_state_dict = self.state_dict()
        if "discriminator.embed_tokens.weight" in state_dict.keys():  # Electra
            prefix = "discriminator"
            for k in state_dict.keys():
                if k.startswith(prefix):
                    if "embed_tokens" in k:
                        upgrade_embed_tokens(k, cur_state_dict, state_dict)
                    elif "_lm_" in k:
                        continue
                    elif "embed_positions" in k:
                        upgrade_position_embed(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="encoder")
                    else:
                        upgrade_layer(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="encoder")
            logger.info(f"Upgrading Electra for Encoder of UniLMEncoderDecoder")
            state_dict.clear()
            for k, v in cur_state_dict.items():
                state_dict[k] = v
        elif "body.embed_tokens.weight" in state_dict.keys() and "body.decoder_layers.0.self_attn.k_proj.weight" in state_dict.keys():  # MAE
            prefix = "body"
            for k in state_dict.keys():
                if k.startswith(f'{prefix}.embed_tokens'):
                    upgrade_embed_tokens(k, cur_state_dict, state_dict)
                elif k.startswith(f'{prefix}.embed_positions'):
                    upgrade_position_embed(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="encoder")
                elif k.startswith(f'{prefix}.layernorm_embedding') or k.startswith(f'{prefix}.layers'):
                    upgrade_layer(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="encoder")
            logger.info(f"Upgrading MAE Encoder for Encoder of UniLMEncoderDecoder")
            if self.args.initialization_strategy == "mae_decoder2decoder":
                for k in state_dict.keys():
                    if k.startswith(f'{prefix}.embed_positions'):
                        upgrade_position_embed(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="decoder")
                    elif k.startswith(f'{prefix}.layernorm_embedding'):
                        upgrade_layer(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="decoder")
                    elif k.startswith(f'{prefix}.decoder_layers'):
                        upgrade_layer(k, cur_state_dict, state_dict, prefix=f"{prefix}.decoder_layers",
                                      new_prefix="decoder.layers")
                logger.info(f"Upgrading MAE Decoder for Decoder of UniLMEncoderDecoder")
            state_dict.clear()
            for k, v in cur_state_dict.items():
                state_dict[k] = v
        elif "body.embed_tokens.weight" in state_dict.keys():  # MLM
            prefix = "body"
            for k in state_dict.keys():
                if k.startswith(f'{prefix}.embed_tokens'):
                    upgrade_embed_tokens(k, cur_state_dict, state_dict)
                elif k.startswith(f'{prefix}.embed_positions'):
                    upgrade_position_embed(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="encoder")
                elif k.startswith(f'{prefix}.layernorm_embedding') or k.startswith(f'{prefix}.layers'):
                    upgrade_layer(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="encoder")
            logger.info(f"Upgrading MLM Encoder for Encoder of UniLMEncoderDecoder")
            if self.args.initialization_strategy == "mlm_encoder2decoder":
                for k in state_dict.keys():
                    if k.startswith(f'{prefix}.embed_positions'):
                        upgrade_position_embed(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="decoder")
                    elif k.startswith(f'{prefix}.layernorm_embedding') or k.startswith(f'{prefix}.layers'):
                        upgrade_layer(k, cur_state_dict, state_dict, prefix=prefix, new_prefix="decoder")
                logger.info(f"Upgrading MLM Encoder for Decoder of UniLMEncoderDecoder")
            state_dict.clear()
            for k, v in cur_state_dict.items():
                state_dict[k] = v
        else:
            if "encoder.embed_positions.weight" in state_dict.keys() and self.args.encoder_learned_pos:
                upgrade_position_embed("encoder.embed_positions.weight", state_dict, state_dict, prefix="encoder",
                                       new_prefix="encoder", max_positions=self.args.max_source_positions + 2)
            if "decoder.embed_positions.weight" in state_dict.keys() and self.args.decoder_learned_pos:
                upgrade_position_embed("decoder.embed_positions.weight", state_dict, state_dict, prefix="decoder",
                                       new_prefix="decoder", max_positions=self.args.max_target_positions + 2)
            logger.info("Directly Loading Checkpoint without Any Change")
        return state_dict

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


@register_model_architecture("unilm_encoder_decoder", "unilm_encoder_decoder_base")
def base_unilm_encoder_decoder_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encode_input_dim = getattr(args, "encode_input_dim", args.encoder_embed_dim)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = getattr(args, "checkpoint_activations", True)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 0)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.initialization_strategy = getattr(args, "initialization_strategy", None)
    base_unilm_architecture(args)  # Compatibile for UniLMBody


@register_model_architecture("unilm_encoder_decoder", "unilm_encoder_decoder_large")
def large_unilm_encoder_decoder_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    base_unilm_encoder_decoder_architecture(args)
