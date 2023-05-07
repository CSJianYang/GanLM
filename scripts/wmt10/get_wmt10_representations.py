import argparse
from torch.utils.data.dataloader import DataLoader
from fairseq import checkpoint_utils
import numpy as np
import torch
from fairseq.data import data_utils, FairseqDataset, Dictionary
import sys

sys.path.append("/home/v-jiaya/unilm-moe/unilm-moe/")
from unilm.models.electra_encoder_decoder_v6 import ElectraEncoderDecoderv6

LANGS = "en,fr,cs,de,fi,lv,et,ro,hi,tr,gu".split(",")


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairLangidDataset(FairseqDataset):
    def __init__(
            self, src, src_sizes, src_dict, src_langs,
            tgt=None, tgt_sizes=None, tgt_dict=None, tgt_langs=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
            encoder_langtok='tgt', decoder_langtok=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.encoder_langtok = encoder_langtok
        self.decoder_langtok = decoder_langtok

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.encoder_langtok == 'src' and src_lang is not None:
            return _lang_token_index(self.src_dict, src_lang)
        elif self.encoder_langtok == 'tgt' and tgt_lang is not None:
            return _lang_token_index(self.src_dict, tgt_lang)
        return self.src_dict.eos()

    def get_decoder_langtok(self, tgt_lang):
        if self.decoder_langtok and tgt_lang is not None:
            return _lang_token_index(self.tgt_dict, tgt_lang)
        return self.tgt_dict.eos()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        tgt_lang = self.tgt_langs[index] if self.tgt_langs is not None else None
        src_lang = self.src_langs[index]

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if src_item[-1] == eos:
                src_item = src_item[:-1]

        # append langid to source end
        if self.encoder_langtok is not None:
            new_eos = self.get_encoder_langtok(src_lang, tgt_lang)
            src_item = torch.cat([torch.LongTensor([new_eos]), src_item])

        # append langid to target start
        if self.decoder_langtok:
            new_eos = self.get_decoder_langtok(tgt_lang)
            tgt_item = torch.cat([torch.LongTensor([new_eos]), tgt_item])
        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    @property
    def sizes(self):
        return np.maximum(self.src_sizes, self.tgt_sizes)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)


def get_dataset(args, vocab):
    with open(args.src_fn) as fp: src_lines = [l for l in fp]
    # with open(args.tgt_fn) as fp: tgt_lines = [l for l in fp]
    src = [torch.tensor([vocab.index(w) for w in src_line.split()] + [vocab.eos()]) for src_line in src_lines]
    src_sizes = np.array([len(s)] for s in src_lines)
    src_lang = [args.src_fn.split('.')[-1]] * len(src)
    # src_lang = ["en"] * len(src)
    dataset = LanguagePairLangidDataset(src, src_sizes, vocab, src_lang, src, src_sizes, vocab, src_lang)
    return dataset


def run(args):
    args.tokens_per_sample = 512
    vocab = Dictionary.load(args.vocab_path)
    vocab.add_symbol("<mask>")
    for i in range(100):
        vocab.add_symbol(f"<mask_{i}>")
    # vocab.pad_to_multiple_(padding_factor=8)
    for lang in LANGS:
        vocab.add_symbol(f"__{lang}__")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt_path)
    model = ElectraEncoderDecoderv6.build_model(state["cfg"].model, src_dict=vocab, tgt_dict=vocab)
    model.load_state_dict(state["model"], strict=True)
    dataset = get_dataset(args, vocab)
    dl = DataLoader(dataset, batch_size=args.bsz, shuffle=False, collate_fn=dataset.collater)
    # model.cuda()
    model.eval()
    model_name = args.model_name

    def write_reprs(output_file, reprs):
        with open(output_file, "w") as w:
            for repr in reprs:
                repr = [str(w) for w in repr]
                w.write("{}\n".format(" ".join(repr)))
        print(f"Successfully saving to {output_file}")

    all_encoder_reprs = []
    all_decoder_reprs = []
    with torch.no_grad():
        for batch_id, sample in enumerate(dl):
            print(f"Processing {batch_id} batches")
            decoder_out = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'],
                                prev_output_tokens=sample['net_input']['prev_output_tokens'], return_all_hiddens=True)
            all_encoder_reprs.append(decoder_out[1]["encoder_states"])
            all_decoder_reprs.append(decoder_out[1]["inner_states"])
        for layer_id in range(len(all_encoder_reprs[0])):
            layer_encoder_reprs = [[round(w, 3) for w in ws] for encoder_reprs in all_encoder_reprs for ws in
                                   encoder_reprs[layer_id][0, :, :].tolist()]
            layer_decoder_reprs = [[round(w, 3) for w in ws] for decoder_reprs in all_decoder_reprs for ws in
                                   decoder_reprs[layer_id][0, :, :].tolist()]
            write_reprs(f"{args.src_fn}.{model_name}.encoder{layer_id}.{args.suffix}", layer_encoder_reprs)
            write_reprs(f"{args.src_fn}.{model_name}.decoder{layer_id}.{args.suffix}", layer_decoder_reprs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default="/home/v-jiaya/unilm-moe/data/wmt10/model/transformer/avg2_6.pt",
                        type=str)
    parser.add_argument('--model_name', default="our", type=str)
    parser.add_argument("--src_fn", type=str,
                        default="/home/v-jiaya/unilm-moe/data/representations/similarity/parallel.en")
    parser.add_argument("--tgt_fn", type=str,
                        default="/home/v-jiaya/unilm-moe/data/representations/similarity/parallel.en")
    parser.add_argument('--vocab_path', default="/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/dict.txt",
                        type=str)
    # parser.add_argument('--wa_layer', default=-1, type=int)
    parser.add_argument('--suffix', default="repr", type=str)
    parser.add_argument('--bsz', default=64, type=int)
    args = parser.parse_args()
    run(args)
