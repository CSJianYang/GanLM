from typing import Dict, List, Optional, Tuple, Any
import os
import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    TransformerModelBase,
    TransformerConfig,
)
from torch import Tensor
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)
try:
    from transformers.tokenization_utils_base import BatchEncoding
except:
    pass


class mT5Encoder(TransformerEncoderBase):
    def __init__(self, cfg, src_dict, encoder):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.encoder = encoder
        self.max_source_positions = cfg.max_source_positions
        self.padding_idx = src_dict.pad()

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        # compute padding mask
        attention_mask = (~ src_tokens.eq(self.padding_idx))
        # self.encoder.eval()
        encoder_out = self.encoder(input_ids=src_tokens, attention_mask=attention_mask)
        return {
            "encoder_out": [encoder_out.last_hidden_state.transpose(0, 1)],  # B x T x C -> T x B x C
            "encoder_padding_mask": [attention_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": None,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class mT5Decoder(TransformerDecoderBase):
    def __init__(
            self,
            cfg,
            tgt_dict,
            decoder
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.decoder = decoder
        self.max_target_positions = cfg.max_target_positions
        self.padding_idx = tgt_dict.pad()
        self.build_output_projection(cfg, tgt_dict, self.decoder.embed_tokens)
        self._future_mask = torch.empty(0)
        self.onnx_trace = False
        self.embed_dim = cfg.decoder_embed_dim

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
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        # x = x * (self.embed_dim ** -0.5)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # inputs_embeds = self.decoder.embed_tokens(prev_output_tokens)
        # self_attn_mask = self.buffered_future_mask(inputs_embeds)
        # self.decoder.eval()
        self_attn_mask = (~ prev_output_tokens.eq(self.padding_idx))
        decoder_out = self.decoder(
            input_ids=prev_output_tokens,
            attention_mask=self_attn_mask,
            encoder_hidden_states=encoder_out["encoder_out"][0].transpose(0, 1),  # T x B x H -> B x T x H
            encoder_attention_mask=encoder_out["encoder_padding_mask"][0],  # T x B x H -> B x T x H
            past_key_values=incremental_state,
            output_hidden_states=True
        )
        x = decoder_out.last_hidden_state
        attn = decoder_out.attentions
        inner_states = decoder_out.hidden_states
        return x, {"attn": [attn], "inner_states": inner_states}

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        self.output_projection = nn.Linear(
            embed_tokens.weight.shape[1],
            embed_tokens.weight.shape[0],
            bias=False,
        )
        # self.output_projection.weight = embed_tokens.weight

    def output_layer(self, features):
        return self.output_projection(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


@register_model("mT5")
class mT5(TransformerModelBase):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        from transformers import MT5Model, MT5Config
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        # mT5_config = MT5Config(f"{os.path.dirname(task.args.finetune_from_model)}/config.json")
        mT5_config = MT5Model.config_class.from_pretrained(
            f"{os.path.dirname(task.args.finetune_from_model)}/config.json")
        mT5_config.vocab_size = len(src_dict)
        # embed_tokens = cls.build_embedding(cfg, src_dict, mT5_config.d_model)
        # mT5 = MT5Model(mT5_config, embed_tokens=embed_tokens)
        mT5 = MT5Model(mT5_config)
        encoder = cls.build_encoder(cfg, src_dict, mT5.encoder)
        decoder = cls.build_decoder(cfg, tgt_dict, mT5.decoder)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, src_dict, encoder):
        return mT5Encoder(cfg, src_dict, encoder)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, decoder):
        return mT5Decoder(cfg, tgt_dict, decoder)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
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
        # src_tokens = torch.LongTensor([[ 10969,    443, 209522,    295,    259,   8992,    261,    259,   3648 ]]).type_as(src_tokens)
        # prev_output_tokens = torch.LongTensor([[ 54620, 191743 ]]).type_as(src_tokens)
        # self.encoder.eval()
        # self.decoder.eval()
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

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        new_state_dict = {}
        for key in state_dict.keys():
            if "encoder" in key:
                new_state_dict[key.replace("encoder", "encoder.encoder")] = state_dict[key]
            elif "decoder" in key:
                new_state_dict[key.replace("decoder", "decoder.decoder")] = state_dict[key]
        new_state_dict["decoder.output_projection.weight"] = self.state_dict()["decoder.output_projection.weight"]
        # new_state_dict["decoder.output_projection.weight"] = state_dict["decoder.embed_tokens.weight"]
        state_dict.clear()
        for k, v in new_state_dict.items():
            state_dict[k] = v
        # word embeddings
        keys = [k for k in state_dict.keys() if "embed_tokens" in k]
        # keys = [k for k in state_dict.keys() if "embed_tokens" in k or "decoder.output_projection" in k]
        cur_words_num, embed_size = self.state_dict()[keys[0]].size()
        prev_words_num = state_dict[keys[0]].size(0)
        if cur_words_num > prev_words_num:
            empty_tensor = state_dict[keys[0]].new(cur_words_num - prev_words_num, embed_size).fill_(0)
            embed_matrix = torch.cat([state_dict[keys[0]], empty_tensor], dim=0)
            logger.info(f"Expanding Word Embedding: {prev_words_num} -> {cur_words_num}")
        else:
            embed_matrix = state_dict[keys[0]]
        for k in keys:
            state_dict[k] = embed_matrix

        return state_dict


@register_model_architecture("mT5", "mT5")
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
