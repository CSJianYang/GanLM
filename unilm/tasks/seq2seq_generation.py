import logging
import os

from dataclasses import dataclass, field
from typing import Optional

from fairseq import utils
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.data import Dictionary, data_utils
from omegaconf import II
from fairseq import metrics, search, tokenizer, utils

logger = logging.getLogger(__name__)

@dataclass
class Seq2Seq_GenerationConfig(TranslationConfig):
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")

@register_task("seq2seq_generation", dataclass=Seq2Seq_GenerationConfig)
class GenerationTask(TranslationTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    def __init__(self, args, src_dict, tgt_dict):
        self.mask_idx = src_dict.index("<mask>")
        super().__init__(args, src_dict, tgt_dict)



    @classmethod
    def load_dictionary(cls, filename, extra_mask_tokens=False, required_batch_size_multiple=1):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        
        if extra_mask_tokens:
            dictionary.add_symbol("<mask>")
            for i in range(100):
                dictionary.add_symbol(f"<mask_{i}>")

        dictionary.pad_to_multiple_(required_batch_size_multiple)

        return dictionary
    
    @property
    def dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)),
            extra_mask_tokens=True,
            required_batch_size_multiple=cfg.required_batch_size_multiple,
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)),
            extra_mask_tokens=True,
            required_batch_size_multiple=cfg.required_batch_size_multiple,
        )
        # add mask token
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(100):
            src_dict.add_symbol(f"<mask_{i}>")
            tgt_dict.add_symbol(f"<mask_{i}>")

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)
