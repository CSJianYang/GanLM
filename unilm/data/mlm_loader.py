import glob
import os
import torch
import numpy as np
import time
import json
import random
import itertools
import hydra
import copy
import math
from omegaconf import DictConfig, OmegaConf
import logging
logger = logging.getLogger(__name__)
from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator


class MLMLoader(BaseBatchGen):

    def __init__(
            self,
            args,
            dataset,
            dictionary,
            tokenizer,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample
        self.sample_break_mode = args.sample_break_mode
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        #Denosing Autoencoder
        self.mask_whole_word = None
        self.mask_idx = self.dictionary.index("<mask>")
        if getattr(args, "_name", None) == "denoising_pretraining" and args.mask_length == "span-poisson":
                _lambda = args.poisson_lambda
                lambda_to_the_k = 1
                e_to_the_minus_lambda = math.exp(-_lambda)
                k_factorial = 1
                ps = []
                for k in range(0, 128):
                    ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                    lambda_to_the_k *= _lambda
                    k_factorial *= k + 1
                    if ps[-1] < 0.0000001:
                        break
                ps = torch.FloatTensor(ps)
                self.mask_span_distribution = torch.distributions.Categorical(ps)
        else:
            self.mask_span_distribution = None
        #################
        self._build_iter()


    def _build_iter(self):
        tokenized_lines = self._multilingual_tokenize()
        self.padded_batches = self._batchify(tokenized_lines)
        
        prefetch_batches = iterators.PrefetchIterator(
            self.padded_batches, 
            buffer_size=10000, 
            buffer_in_main_process=True, 
            log_empty_buffer_warning=True and self.shard_id == 0,
        )

        prefetch_batches = iterators.MapIterator(
            prefetch_batches, self._move_to_tensor
        )

        self._iter = prefetch_batches


    def _multilingual_tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize(self, data):
        '''
        data:
        {
            'source': list[Path],
            'source_lang': str,
            'count': int,
            'weight': float,
            'name': str,
        }
        '''
        dataset = list(
                zip(
                    data['source'],
                    itertools.repeat(data['source_lang']), 
                )
        )

        if self.shuffle:
            chunk_files = \
                iterators.InfinitePermutationSourceIterator(
                    dataset,
                    seed=self.seed, 
                    shuffle=self.shuffle, 
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        else:
            chunk_files = \
                iterators.ChunkedSourceIterator(
                    dataset,
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines


    def _batchify(self, lines):
        
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = iterators.BlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)
            
            batches = iterators.BucketedReadaheadBatchIterator(
                    lines,
                    read_ahead=self.batch_read_ahead, 
                    key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
                    batch_size=dynamic_batch_size, 
                    shuffle=self.shuffle,
                    seed=self.seed,
            )

        def collate(batch):
            batch_size = len(batch)

            mlm_source_max_length = max([len(x[0]) for x in batch])
            mlm_target_max_length = max([len(x[1]) for x in batch])
            s2s_source_max_length = max([len(x[2]) for x in batch])
            s2s_target_max_length = max([len(x[3]) for x in batch])

            mlm_source_ids = np.full(shape=(batch_size, mlm_source_max_length), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            mlm_target_ids = np.full(shape=(batch_size, mlm_target_max_length), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            s2s_source_ids = np.full(shape=(batch_size, s2s_source_max_length), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            s2s_target_ids = np.full(shape=(batch_size, s2s_target_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            s2s_prev_input_ids = np.full(shape=(batch_size, s2s_target_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            #
            non_last_token_s2s_target_ids = np.full(shape=(batch_size, s2s_target_max_length - 1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            for i, (mlm_input_ids, mlm_label_ids, s2s_input_ids, s2s_label_ids) in enumerate(batch):
                mlm_source_ids[i, :len(mlm_input_ids)] = mlm_input_ids
                mlm_target_ids[i, :len(mlm_label_ids)] = mlm_label_ids
                s2s_source_ids[i, :len(s2s_input_ids)] = s2s_input_ids
                s2s_target_ids[i, :len(s2s_label_ids)-1] = s2s_label_ids[1:]
                s2s_prev_input_ids[i, :len(s2s_label_ids)-1] = s2s_label_ids[:-1]
                non_last_token_s2s_target_ids[i, :len(s2s_label_ids)-2] = s2s_label_ids[1:-1]


            if getattr(self.args, "debug", False):
                logger.info(f"{self.dictionary.string(s2s_source_ids)}")
                logger.info(f"{self.dictionary.string(s2s_target_ids)}")
            ret_batch = {
                'mlm': {
                    'src_tokens': mlm_source_ids.astype(np.int64),
                    'targets': mlm_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                },
                'seq2seq': {
                    'src_tokens': s2s_source_ids.astype(np.int64),
                    'tgt_tokens': s2s_prev_input_ids.astype(np.int64),
                    'src_lengths': torch.LongTensor([len(x[2]) for x in batch]),
                    'tgt_lengths':torch.LongTensor([len(x[3]) for x in batch]),
                    'non_last_token_tgt': torch.LongTensor(non_last_token_s2s_target_ids),
                    'targets': s2s_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                },
                #Compatible for T5
                'net_input': {
                    'src_tokens': s2s_source_ids.astype(np.int64),
                    'prev_output_tokens': s2s_prev_input_ids.astype(np.int64),
                    'src_lengths': None,
                },
                'target': s2s_target_ids.astype(np.int64),
                'nsentences': batch_size,
                'ntokens': sum([len(x[0]) for x in batch]),
            }

            return ret_batch

        padded_batches = iterators.MapIterator(
            batches, collate
        )

        return padded_batches
    
    def _prepare(self, _random, doc):
        if self.args._name == "pretraining":
            nonmasked_tokens, masked_tokens = self._mask_lm(_random, doc)
            nonnoise_spans, noise_spans = self._span_corruption(_random, doc)
        elif self.args._name == "denoising_pretraining":
            nonnoise_spans, noise_spans = self._donoising_autoencoder(doc)
            #noise_spans, noise_spans = noise_spans.tolist()[:self.args.tokens_per_sample - 2], noise_spans.tolist()[:self.args.tokens_per_sample - 2]
            noise_spans, noise_spans = noise_spans.tolist(), noise_spans.tolist()
            nonmasked_tokens, masked_tokens = nonnoise_spans, noise_spans
        return nonmasked_tokens, masked_tokens, nonnoise_spans, noise_spans
    
    def _mask_lm(self, _random, doc):
        def mask_tokens():
            return f"<mask>"
        
        length = len(doc)
        mask_tokens_num = int(length * self.args.mask_prob)
        mask_tokens_num = min(max(mask_tokens_num, 1), length - 1)
        possible_mask_positions = _random.sample(range(length), k=mask_tokens_num)
        possible_mask_positions = sorted(possible_mask_positions)

        nonmasked_tokens = copy.deepcopy(doc)
        masked_tokens = []

        for position in possible_mask_positions:
            masked_tokens.append(nonmasked_tokens[position])
            nonmasked_tokens[position] = self.dictionary.indices[mask_tokens()]
        
        return nonmasked_tokens, masked_tokens



    #Add Noise
    def permute_sentences(self, source, p=1.0):
        if len(source) <= 3:
            return source
        full_stops = source == self.dictionary.eos()
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.args.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.args.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                4, len(self.dictionary), size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.args.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.args.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.dictionary), size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result


    def _donoising_autoencoder(self, tokens):
        if isinstance(tokens, list):
            tokens = torch.LongTensor(tokens)
        noise_spans, nonnoise_spans = tokens, tokens.clone()

        if self.args.permute_sentence_ratio > 0.0:
            noise_spans = self.permute_sentences(noise_spans, self.args.permute_sentence_ratio)

        if self.args.mask_ratio > 0:
            noise_spans = self.add_whole_word_mask(noise_spans, self.args.mask_ratio)

        if self.args.insert_ratio > 0:
            noise_spans = self.add_insertion_noise(noise_spans, self.args.insert_ratio)

        if self.args.rotate_ratio > 0.0 and np.random.random() < self.args.rotate_ratio:
            noise_spans = self.add_rolling_noise(noise_spans)


        # assert (noise_spans >= 0).all()
        # assert (noise_spans[1:-1] >= 1).all()
        # assert (noise_spans <= len(self.dictionary)).all()
        # assert noise_spans[0] == self.dictionary.bos()
        # assert noise_spans[-1] == self.dictionary.eos()
        return nonnoise_spans, noise_spans
    #########End#############


    def _span_corruption(self, _random, doc):
        
        def mask_tokens(i):
            return f"<mask_{i}>"

        length = len(doc)
        noise_tokens_num = int(length * self.args.mask_prob)
        noise_tokens_num = min(max(noise_tokens_num, 1), length - 1)
        noise_spans_num = int(noise_tokens_num / self.args.span_length)
        noise_spans_num = max(noise_spans_num, 1)
        nonnoise_tokens_num = length - noise_tokens_num

        if noise_spans_num == 1:
            noise_split_positions = [0, noise_tokens_num]
        else:
            possible_split_positions = list(range(1, noise_tokens_num))
            _random.shuffle(possible_split_positions)
            noise_split_positions = sorted(possible_split_positions[:noise_spans_num-1])
            noise_split_positions = [0] + noise_split_positions + [noise_tokens_num]
        
        possible_insert_positions = list(range(nonnoise_tokens_num))
        _random.shuffle(possible_insert_positions)
        noise_insert_positions = sorted(possible_insert_positions[:noise_spans_num])

        nonnoise_spans, noise_spans = [], []
        last_end = 0
        for i in range(noise_spans_num):
            start_pos = noise_insert_positions[i] + noise_split_positions[i]
            end_pos = noise_insert_positions[i] + noise_split_positions[i+1]
            mask_id = self.dictionary.indices[mask_tokens(i)]

            if getattr(self.args, "remove_target_sentinel", False):
                noise_spans.append(doc[start_pos:end_pos])
            else:
                noise_spans.append([mask_id] + doc[start_pos:end_pos])
                
            if getattr(self.args, "remove_source_sentinel", False):
                nonnoise_spans.extend(doc[last_end:start_pos])
            else:
                nonnoise_spans.extend(doc[last_end:start_pos] + [mask_id])
                
            last_end = end_pos
            
        nonnoise_spans.extend(doc[last_end:])
        noise_spans = sum(noise_spans, [])

        return nonnoise_spans, noise_spans

    def _denoising(self, _random, doc):

        def mask_tokens(i):
            return f"<mask_{i}>"

        length = len(doc)
        noise_tokens_num = int(length * self.args.mask_prob)
        noise_tokens_num = min(max(noise_tokens_num, 1), length - 1)
        noise_spans_num = int(noise_tokens_num / self.args.span_length)
        noise_spans_num = max(noise_spans_num, 1)
        nonnoise_tokens_num = length - noise_tokens_num

        if noise_spans_num == 1:
            noise_split_positions = [0, noise_tokens_num]
        else:
            possible_split_positions = list(range(1, noise_tokens_num))
            _random.shuffle(possible_split_positions)
            noise_split_positions = sorted(possible_split_positions[:noise_spans_num - 1])
            noise_split_positions = [0] + noise_split_positions + [noise_tokens_num]

        possible_insert_positions = list(range(nonnoise_tokens_num))
        _random.shuffle(possible_insert_positions)
        noise_insert_positions = sorted(possible_insert_positions[:noise_spans_num])

        nonnoise_spans, noise_spans = [], []
        last_end = 0
        for i in range(noise_spans_num):
            start_pos = noise_insert_positions[i] + noise_split_positions[i]
            end_pos = noise_insert_positions[i] + noise_split_positions[i + 1]
            mask_id = self.dictionary.indices[mask_tokens(i)]

            if getattr(self.args, "remove_target_sentinel", False):
                noise_spans.append(doc[start_pos:end_pos])
            else:
                noise_spans.append([mask_id] + doc[start_pos:end_pos])

            if getattr(self.args, "remove_source_sentinel", False):
                nonnoise_spans.extend(doc[last_end:start_pos])
            else:
                nonnoise_spans.extend(doc[last_end:start_pos] + [mask_id])

            last_end = end_pos

        nonnoise_spans.extend(doc[last_end:])
        noise_spans = sum(noise_spans, [])

        return nonnoise_spans, noise_spans



    def _read_from_files(self, source_file, source_lang):
        # data = []
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')

        doc = [self.dictionary.bos()]
        for line in lines:
            if line == "":
                if self.sample_break_mode == 'complete_doc':
                    # data.append(doc)
                    yield doc
                    doc = [self.dictionary.bos()]
                continue
            
            tokenized_line = self.tokenizer.EncodeAsPieces(line)
            tokenized_id = [self.dictionary.index(token) for token in tokenized_line] + [self.dictionary.eos_index]

            if len(tokenized_id) > self.tokens_per_sample:
                continue
            if len(doc) + len(tokenized_id) > self.tokens_per_sample:
                # data.append(doc)
                yield doc
                doc = [self.dictionary.bos()]
            doc.extend(tokenized_id)

        if len(doc) > 1 and len(doc) <= self.tokens_per_sample:
            # data.append(doc)
            yield doc

        # return data