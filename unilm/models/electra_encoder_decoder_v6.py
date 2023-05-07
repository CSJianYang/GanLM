from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
import math
import torch
import torch.nn as nn
import copy
from typing import Dict, List, Optional, Tuple, Any
from torch import Tensor
from fairseq import utils
import logging
import torch.nn.functional as F
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
from fairseq.models.roberta import RobertaClassificationHead
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.modules import (
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
)
from unilm.models.unilm import LMHead, UniLMBody, UniLMModelConfig, base_unilm_architecture
from unilm.models.unilm_encoder_decoder import UniLMEncoder, UniLMDecoder, UniLMEncoderDecoder
from fairseq.models.roberta.hub_interface import RobertaHubInterface

logger = logging.getLogger(__name__)


@dataclass
class ElectraEncoderDecoderConfigv6(FairseqDataclass):
    initialization_strategy: ChoiceEnum(["discriminator", "generator", "both"]) = field(
        default="generator", metadata={"help": "initialization strategy"}
    )
    share_generator_discriminator: bool = field(
        default=False,
        metadata={"help": "share parameters"}
    )
    encoder_discriminator_task: bool = field(
        default=False,
        metadata={"help": "share parameters"}
    )
    discriminator_decoder_layers: int = field(
        default=12,
        metadata={"help": "share parameters"}
    )
    generator_decoder_layers: int = field(
        default=12,
        metadata={"help": "share parameters"}
    )
    discriminator_sampling_temperature: float = field(
        default=1.0,
        metadata={"help": "discriminator sampling temperature"}
    )
    scale_fc: bool = field(
        default=False,
        metadata={"help": "share parameters"}
    )


class ElectraEncoderv6(TransformerEncoderBase):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            output_projection=None,
    ):
        super(TransformerEncoderBase, self).__init__(dictionary)
        self.args = args
        self.max_target_positions = getattr(args, "max_target_positions", 512)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        self.padding_idx = embed_tokens.padding_idx
        self.discriminator = UniLMBody(
            args,
            dictionary,
            embed_tokens
        )
        if args.encoder_discriminator_task:
            self.discriminator_lm_head = self.build_lm_head(
                args,
                args.encoder_embed_dim,
                2,
                args.activation_fn,
                weight=None
            )
        else:
            self.discriminator_lm_head = None
        if args.task == 'pretraining' and args.encoder_discriminator_task:
            generator_args = copy.copy(args)
            generator_args.encoder_layers = getattr(args, "generator_encoder_layers", 4)
            generator_args.dropout = 0.0
            generator_args.attention_dropout = 0.0
            generator_args.activation_dropout = 0.0
            generator_args.pooler_dropout = 0.0
            generator_args.task_moe = False
            self.generator = UniLMBody(
                generator_args,
                dictionary,
                embed_tokens
            )
            self.generator_lm_head = self.build_lm_head(
                generator_args,
                generator_args.encoder_embed_dim,
                len(dictionary),
                generator_args.activation_fn,
                weight=embed_tokens.weight
            )
        else:
            self.generator, self.generator_lm_head = None, None
        self.classification_heads = nn.ModuleDict()

    def build_lm_head(self, args, embed_dim, output_dim, activation_fn, weight):
        return LMHead(embed_dim, output_dim, activation_fn, weight)

    def output_layer(self, features):
        return self.generator_lm_head(features)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if self.generator is None:
            keys_to_delete = []
            for k in state_dict.keys():
                if 'generator' in k:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del state_dict[k]

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.discriminator.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.discriminator.embed_positions.max_positions)

    def forward(self, src_tokens=None, tgt_tokens=None, incremental_state=None, classification_head_name=None,
                masked_tokens=None, features_only=True, **kwargs):
        if classification_head_name is not None:
            x, extra = self.discriminator(src_tokens, None, incremental_state, return_all_hiddens=True)
            x = self.classification_heads[classification_head_name](x)
            return x, extra

        if tgt_tokens is not None or features_only:
            if self.discriminator is not None:
                x, extra = self.discriminator(src_tokens, tgt_tokens, incremental_state, return_all_hiddens=True)
            else:
                x, extra = self.generator(src_tokens, tgt_tokens, incremental_state, return_all_hiddens=True)
            return {
                "encoder_out": [x.transpose(0, 1)],  # T x B x C
                "encoder_padding_mask": [src_tokens.eq(self.padding_idx)],  # B x T
                "encoder_embedding": [],  # B x T x C
                "encoder_states": extra['inner_states'],  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
            }

        generator_x, generator_extra = self.generator(src_tokens, None, incremental_state, return_all_hiddens=True)
        generator_logits = self.generator_lm_head(generator_x, masked_tokens=masked_tokens)

        with torch.no_grad():
            sampled_probs = torch.softmax(generator_logits.view(-1, generator_logits.size(-1)), -1, dtype=torch.float32)
            sampled_tokens = torch.multinomial(sampled_probs, 1).view(-1)
            input_tokens = src_tokens.clone()
            input_tokens[masked_tokens] = sampled_tokens

        discriminator_x, discriminator_extra = self.discriminator(input_tokens, None, incremental_state,
                                                                  return_all_hiddens=True)
        discriminator_logits = self.discriminator_lm_head(discriminator_x)

        return {
            "electra": [input_tokens, generator_logits, discriminator_logits, generator_extra, discriminator_extra],
            "encoder_out": [discriminator_x.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [src_tokens.eq(self.padding_idx)],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": discriminator_extra['inner_states'],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class ElectraDecoderBase(UniLMDecoder):
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
            embed_positions=None,
            layernorm_embedding=None,
            layers=None,
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
        if embed_positions is None:
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
        else:
            self.embed_positions = embed_positions

        if layernorm_embedding is None:
            if cfg.layernorm_embedding:
                self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
            else:
                self.layernorm_embedding = None
        else:
            self.layernorm_embedding = layernorm_embedding

        self.cross_self_attention = cfg.cross_self_attention

        if layers is None:
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
        else:
            self.layers = layers
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
        if getattr(cfg, 'rescale_init', False):
            self.rescale_fixup()

    def rescale_fixup(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id in range(len(self.layers)):
            layer = self.layers[layer_id]
            rescale(layer.self_attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.encoder_attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)


class ElectraDecoderv6(UniLMDecoder):
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
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        self.args = args
        super(TransformerDecoderBase, self).__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        if args.task == 'pretraining':
            generator_args = copy.copy(args)
            if getattr(args, "generator_decoder_layers", -1) == -1:
                generator_args.generator_decoder_layers = args.decoder_layers
            else:
                generator_args.decoder_layers = args.generator_decoder_layers
            self.generator = ElectraDecoderBase(
                generator_args,
                dictionary,
                embed_tokens,
                no_encoder_attn=no_encoder_attn,
                output_projection=output_projection,
            )
            discriminator_args = copy.copy(args)
            if getattr(args, "discriminator_decoder_layers", -1) == -1:
                discriminator_args.discriminator_decoder_layers = args.decoder_layers
            else:
                discriminator_args.decoder_layers = args.discriminator_decoder_layers
            if args.share_generator_discriminator:
                discriminator_args.rescale_init = False
                self.discriminator = ElectraDecoderBase(
                    discriminator_args,
                    dictionary,
                    embed_tokens,
                    no_encoder_attn=no_encoder_attn,
                    layernorm_embedding=self.generator.layernorm_embedding,
                    embed_positions=self.generator.embed_positions,
                    layers=self.generator.layers[:discriminator_args.discriminator_decoder_layers],
                    output_projection=self.build_lm_head(
                        args,
                        args.decoder_embed_dim,
                        2,
                        args.activation_fn,
                        weight=None
                    )
                )
            else:
                discriminator_args.dropout = 0.0
                discriminator_args.attention_dropout = 0.0
                discriminator_args.activation_dropout = 0.0
                discriminator_args.pooler_dropout = 0.0
                discriminator_args.task_moe = False
                self.discriminator = ElectraDecoderBase(
                    discriminator_args,
                    dictionary,
                    embed_tokens,
                    no_encoder_attn=no_encoder_attn,
                    output_projection=self.build_lm_head(
                        args,
                        args.decoder_embed_dim,
                        2,
                        args.activation_fn,
                        weight=None
                    )
                )
        else:
            if args.initialization_strategy == "discriminator":
                if getattr(args, "discriminator_decoder_layers", -1) == -1:
                    args.discriminator_decoder_layers = args.decoder_layers
                else:
                    args.decoder_layers = args.discriminator_decoder_layers
                self.discriminator = ElectraDecoderBase(
                    args,
                    dictionary,
                    embed_tokens,
                    no_encoder_attn=no_encoder_attn,
                    output_projection=output_projection,
                )
                self.generator = None
            elif args.initialization_strategy == "generator":
                if getattr(args, "generator_decoder_layers", -1) == -1:
                    args.generator_decoder_layers = args.decoder_layers
                else:
                    args.decoder_layers = args.generator_decoder_layers
                self.generator = ElectraDecoderBase(
                    args,
                    dictionary,
                    embed_tokens,
                    no_encoder_attn=no_encoder_attn,
                    output_projection=output_projection,
                )
                self.discriminator = None
            else:
                generator_args = copy.copy(args)
                if getattr(args, "generator_decoder_layers", -1) == -1:
                    generator_args.generator_decoder_layers = args.decoder_layers
                else:
                    generator_args.decoder_layers = args.generator_decoder_layers
                self.generator = ElectraDecoderBase(
                    generator_args,
                    dictionary,
                    embed_tokens,
                    no_encoder_attn=no_encoder_attn,
                    output_projection=output_projection,
                )
                discriminator_args = copy.copy(args)
                if getattr(args, "discriminator_decoder_layers", -1) == -1:
                    discriminator_args.discriminator_decoder_layers = args.decoder_layers
                else:
                    discriminator_args.decoder_layers = args.discriminator_decoder_layers
                if args.share_generator_discriminator:
                    discriminator_args.rescale_init = False
                    self.discriminator = ElectraDecoderBase(
                        discriminator_args,
                        dictionary,
                        embed_tokens,
                        no_encoder_attn=no_encoder_attn,
                        layernorm_embedding=self.generator.layernorm_embedding,
                        embed_positions=self.generator.embed_positions,
                        layers=self.generator.layers[:discriminator_args.discriminator_decoder_layers],
                        output_projection=self.build_lm_head(
                            args,
                            args.decoder_embed_dim,
                            2,
                            args.activation_fn,
                            weight=None
                        )
                    )
                else:
                    discriminator_args.dropout = 0.0
                    discriminator_args.attention_dropout = 0.0
                    discriminator_args.activation_dropout = 0.0
                    discriminator_args.pooler_dropout = 0.0
                    discriminator_args.task_moe = False
                    self.discriminator = ElectraDecoderBase(
                        discriminator_args,
                        dictionary,
                        embed_tokens,
                        no_encoder_attn=no_encoder_attn,
                        output_projection=self.build_lm_head(
                            args,
                            args.decoder_embed_dim,
                            2,
                            args.activation_fn,
                            weight=None
                        )
                    )

    def build_lm_head(self, args, embed_dim, output_dim, activation_fn, weight):
        return LMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            masked_tokens=None,
            targets=None,
            only_generator_or_discriminator: bool = True
    ):
        if only_generator_or_discriminator:
            if self.generator is not None:
                decoder_out = self.generator(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
            else:
                if self.args.task == "xnli" or self.args.task == "pretraining":  # True for Language Understanding False for Generation
                    full_context_alignment = True
                else:
                    full_context_alignment = False
                decoder_out = self.discriminator(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    features_only=features_only,
                    full_context_alignment=full_context_alignment,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
            return decoder_out

        generator_out = self.generator(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        generator_logits = generator_out[0]
        with torch.no_grad():
            sampled_probs = torch.softmax(
                generator_logits.view(-1, generator_logits.size(-1)) / self.args.discriminator_sampling_temperature, -1,
                dtype=torch.float32)
            sampled_tokens = torch.multinomial(sampled_probs, 1).view(-1)
            sampled_tokens = sampled_tokens[masked_tokens.view(-1)]
            input_tokens = targets.clone()
            input_tokens[masked_tokens] = sampled_tokens
            #
            sampled_probs = sampled_probs.scatter(-1, targets.view(sampled_probs.size(0), 1),
                                                  0)  # Do not resample the gold label
            masked_sampled_tokens = torch.multinomial(sampled_probs, 1).view(-1)
            masked_sampled_tokens = masked_sampled_tokens.view(targets.size())

        discriminator_out = self.discriminator(
            input_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            full_context_alignment=True,  # Allow Bidirectional
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        return {
            "discriminator_out": discriminator_out,
            "generator_out": generator_out,
            "encoder_out": encoder_out,
            "input_tokens": input_tokens,
            "masked_sampled_tokens": masked_sampled_tokens
        }

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.args.max_target_positions


@register_model("electra_encoder_decoder_v6")
class ElectraEncoderDecoderv6(TransformerModel):
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
        self.classification_heads = nn.ModuleDict()

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        gen_parser_from_dataclass(
            parser, UniLMModelConfig(), delete_default=True, with_prefix=""
        )
        gen_parser_from_dataclass(
            parser, ElectraEncoderDecoderConfigv6(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task=None, src_dict=None, tgt_dict=None):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_electra_encoder_decoder_architecture_v6(args)

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        args.decoder_input_dim = int(args.decoder_input_dim)
        args.decoder_output_dim = int(args.decoder_output_dim)
        # --

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if task is not None:
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
        return ElectraEncoderv6(wrapper_cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return ElectraDecoderv6(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    def register_classification_head(
            self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        def upgrade_embed_tokens(embed_k, cur_state_dict, state_dict):
            keys = [k for k in cur_state_dict.keys() if
                    "embed_tokens" in k or "decoder.generator.output_projection" in k]
            cur_words_num, embed_size = cur_state_dict[embed_k].size()
            prev_words_num = state_dict[embed_k].size(0)
            if cur_words_num > prev_words_num:
                empty_tensor = state_dict[embed_k].new(cur_words_num - prev_words_num, embed_size).fill_(0)
                embed_matrix = torch.cat([state_dict[embed_k], empty_tensor], dim=0)
                logger.info(f"Expanding Word Embedding: {prev_words_num} -> {cur_words_num}")
            else:
                embed_matrix = state_dict[embed_k][:cur_words_num, :]
                logger.info(f"Clipping Word Embedding: {prev_words_num} -> {cur_words_num}")
            for k in keys:
                state_dict[k] = embed_matrix
                cur_state_dict[k] = embed_matrix

        def upgrade_position_embed(k, cur_state_dict, state_dict, prefix, new_prefix="encoder", max_positions=-1):
            if max_positions == -1:
                max_positions = cur_state_dict[k.replace(prefix, new_prefix)].size(0)
            assert k.replace(prefix, new_prefix) in cur_state_dict.keys()
            if max_positions < state_dict[k].size(0):
                logger.info(f"{k} | Clipping {state_dict[k].size(0)} -> {max_positions} positions (start from 2th pos)")
                cur_state_dict[k.replace(prefix, new_prefix)] = state_dict[k][:max_positions, :]
            else:
                logger.info(
                    f"{k} | Expanding {state_dict[k].size(0)} -> {max_positions} positions (start from 2th pos)")
                cur_state_dict[k.replace(prefix, new_prefix)][: state_dict[k].size(0)] = state_dict[k][:max_positions,
                                                                                         :]

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
        elif self.args.task != "pretraining" and (
                "decoder.generator.embed_tokens.weight" in state_dict or "decoder.discriminator.embed_tokens.weight" in state_dict):  # self.args.task != "pretraining" and "decoder.generator.embed_tokens.weight" in state_dict and "decoder.discriminator.embed_tokens.weight" in state_dict
            if "encoder.embed_tokens.weight" in state_dict:
                state_dict.pop("encoder.embed_tokens.weight")
            upgrade_embed_tokens("encoder.discriminator.embed_tokens.weight", cur_state_dict, state_dict)
            if self.args.initialization_strategy == "discriminator":
                if "encoder.discriminator.embed_positions.weight" in state_dict.keys() and self.args.encoder_learned_pos:
                    upgrade_position_embed("encoder.discriminator.embed_positions.weight", state_dict, state_dict,
                                           prefix="encoder", new_prefix="encoder",
                                           max_positions=self.args.max_source_positions + 2)
                if "decoder.discriminator.embed_positions.weight" in state_dict.keys() and self.args.decoder_learned_pos:
                    upgrade_position_embed("decoder.discriminator.embed_positions.weight", state_dict, state_dict,
                                           prefix="decoder", new_prefix="decoder",
                                           max_positions=self.args.max_target_positions + 2)
                for k in state_dict.keys():
                    if not k.startswith("decoder.generator.") and not k.startswith(
                            "decoder.discriminator.output_projection"):
                        assert k in cur_state_dict, k
                        cur_state_dict[k] = state_dict[k]
                cur_state_dict['decoder.discriminator.output_projection.weight'] = state_dict[
                    "decoder.discriminator.embed_tokens.weight"]  # share all embeddings
                state_dict.clear()
                for k, v in cur_state_dict.items():
                    state_dict[k] = v
                logger.info("Directly Loading Checkpoint without Any Change | Deleting Generator")
            elif self.args.initialization_strategy == "generator":
                if "encoder.discriminator.embed_positions.weight" in state_dict.keys() and self.args.encoder_learned_pos:
                    upgrade_position_embed("encoder.discriminator.embed_positions.weight", state_dict, state_dict,
                                           prefix="encoder", new_prefix="encoder",
                                           max_positions=self.args.max_source_positions + 2)
                if "decoder.generator.embed_positions.weight" in state_dict.keys() and self.args.decoder_learned_pos:
                    upgrade_position_embed("decoder.generator.embed_positions.weight", state_dict, state_dict,
                                           prefix="decoder", new_prefix="decoder",
                                           max_positions=self.args.max_target_positions + 2)
                for k in state_dict.keys():
                    if not k.startswith("decoder.discriminator."):
                        assert k in cur_state_dict, k
                        cur_state_dict[k] = state_dict[k]
                state_dict.clear()
                for k, v in cur_state_dict.items():
                    state_dict[k] = v
                logger.info("Directly Loading Checkpoint without Any Change | Deleting Discriminator")
            elif self.args.initialization_strategy == "both":
                if "encoder.discriminator.embed_positions.weight" in state_dict.keys() and self.args.encoder_learned_pos:
                    upgrade_position_embed("encoder.discriminator.embed_positions.weight", state_dict, state_dict,
                                           prefix="encoder", new_prefix="encoder",
                                           max_positions=self.args.max_source_positions + 2)
                if "decoder.discriminator.embed_positions.weight" in state_dict.keys() and self.args.decoder_learned_pos:
                    upgrade_position_embed("decoder.discriminator.embed_positions.weight", state_dict, state_dict,
                                           prefix="decoder", new_prefix="decoder",
                                           max_positions=self.args.max_target_positions + 2)
                if "decoder.generator.embed_positions.weight" in state_dict.keys() and self.args.decoder_learned_pos:
                    upgrade_position_embed("decoder.generator.embed_positions.weight", state_dict, state_dict,
                                           prefix="decoder", new_prefix="decoder",
                                           max_positions=self.args.max_target_positions + 2)
                for k in state_dict.keys():
                    assert k in cur_state_dict, k
                    cur_state_dict[k] = state_dict[k]
                state_dict.clear()
                for k, v in cur_state_dict.items():
                    state_dict[k] = v
                logger.info("Directly Loading Checkpoint without Any Change for GanLM Strategy")
        else:
            logger.info(f"Directly Loading without Any Change")
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
            masked_tokens=None,
            targets=None,
            classification_head_name=None,
            only_generator_or_discriminator=True,  # "discriminator" "generator"
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, features_only=True
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            masked_tokens=masked_tokens,
            targets=targets,
            only_generator_or_discriminator=only_generator_or_discriminator,
        )
        if classification_head_name is not None:
            output = self.classification_heads[classification_head_name](decoder_out[0])
            decoder_out = (output, decoder_out[1])
        if isinstance(decoder_out, tuple) and len(decoder_out) > 1 and isinstance(decoder_out[1], dict):
            decoder_out[1]["encoder_states"] = encoder_out["encoder_states"]
        return decoder_out

    def encode_tokenized(self, dictionary, bpe_sentence: str, *addl_sentences, no_separator=False,
                         add_eos_tag=True) -> torch.LongTensor:
        if add_eos_tag:
            bpe_sentence = '<s> ' + bpe_sentence + ' </s>'
        else:
            bpe_sentence = bpe_sentence
        for s in addl_sentences:
            bpe_sentence += ' ' + s + ' </s>'
        tokens = dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        prev_token = torch.roll(tokens, 1)
        return tokens.long().unsqueeze(0), prev_token.long().unsqueeze(0)

    def predict(self, head: str, tokens: torch.LongTensor, prev_tokens: torch.LongTensor, return_logits: bool = False):
        src_lengths = torch.LongTensor([tokens.size()])
        logits = \
        self.forward(src_tokens=tokens, src_lengths=src_lengths, prev_output_tokens=prev_tokens, features_only=True,
                     classification_head_name=head)[0]
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)


@register_model_architecture("electra_encoder_decoder_v6", "electra_encoder_decoder_v6")
def electra_encoder_decoder_architecture_v6(args):
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
    args.generator_encoder_layers = getattr(args, "generator_encoder_layers", 4)
    args.max_source_positions = getattr(args, "max_source_positions", 512)
    args.max_target_positions = getattr(args, "max_target_positions", 512)
    args.share_generator_discriminator = getattr(args, "share_generator_discriminator", False)
    # v6
    # args.scale_fc = True
    base_unilm_architecture(args)  # Compatibile for UniLMBody


@register_model_architecture("electra_encoder_decoder_v6", "GanLM")
def GanLM(args):
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
    args.generator_encoder_layers = getattr(args, "generator_encoder_layers", 4)
    args.max_source_positions = getattr(args, "max_source_positions", 512)
    args.max_target_positions = getattr(args, "max_target_positions", 512)
    args.share_generator_discriminator = getattr(args, "share_generator_discriminator", False)
    base_unilm_architecture(args)  # Compatibile for UniLMBody


@register_model_architecture("electra_encoder_decoder_v6", "electra_encoder_decoder_v6_base")
def base_electra_encoder_decoder_architecture_v6(args):
    electra_encoder_decoder_architecture_v6(args)


@register_model_architecture("electra_encoder_decoder_v6", "electra_encoder_decoder_v6_large")
def large_electra_encoder_decoder_architecture(args):
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
    args.generator_decoder_layers = getattr(args, "generator_decoder_layers", 24)
    args.discriminator_decoder_layers = getattr(args, "discriminator_decoder_layers", 4)
    base_electra_encoder_decoder_architecture_v6(args)


@register_model_architecture("electra_encoder_decoder_v6", "GanLM_large")
def large_GanLM_architecture(args):
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
    args.generator_decoder_layers = getattr(args, "generator_decoder_layers", 24)
    args.discriminator_decoder_layers = getattr(args, "discriminator_decoder_layers", 4)
    base_electra_encoder_decoder_architecture_v6(args)
