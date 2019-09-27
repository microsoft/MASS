# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np

import torch

from collections import OrderedDict
from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    TokenBlockDataset,
)
from fairseq.tasks import FairseqTask, register_task
from .masked_dataset import MaskedLanguagePairDataset
from .bert_dictionary import BertDictionary


@register_task('masked_s2s')
class MaskedS2STask(FairseqTask):
    """
    Train a sequence-to-sequence task

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of tokens per sample for text dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')

        parser.add_argument('--mask-s2s-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--mask-s2s-mask-keep-rand', default="0.8,0.1,0.1", type=str,
                            help='Word prediction probability for decoder mask')

        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        return model

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))
        
        self.datasets[split] = self.build_s2s_dataset(dataset)

    def build_s2s_dataset(self, dataset):
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        
        pred_probs = torch.FloatTensor([float(x) for x in self.args.mask_s2s_mask_keep_rand.split(',')])

        s2s_dataset = MaskedLanguagePairDataset(
            dataset, dataset.sizes, self.source_dictionary,
            shuffle=True, mask_prob=self.args.mask_s2s_prob,
            pred_probs=pred_probs,
        )
        return s2s_dataset

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        raise NotImplementedError

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        raise NotImplementedError

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    def max_positions(self):
        max_positions = 1024
        if hasattr(self.args, 'max_positions'):
            max_positions = min(max_positions, self.args.max_positions)
        if hasattr(self.args, 'max_source_positions'):
            max_positions = min(max_positions, self.args.max_source_positions)
        if hasattr(self.args, 'max_target_positions'):
            max_positions = min(max_positions, self.args.max_target_positions)
        return max_positions
