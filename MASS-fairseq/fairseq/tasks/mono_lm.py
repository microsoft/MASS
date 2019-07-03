# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os

from collections import OrderedDict

from fairseq import tokenizer
from fairseq.data.masked_lm_dictionary import MaskedLMDictionary

from fairseq.data import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    TokenBlockDataset,
)

from fairseq.data import Dictionary
from fairseq.data.masked_lm_dataset import MaskedLMDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset

from . import FairseqTask, register_task


@register_task('mono_lm')
class MonoLingualLMTask(FairseqTask):
    """
    Task for training cross-lingual language models.
    For more details look at: https://arxiv.org/pdf/1901.07291.pdf
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments'
                                 ' per sample')
        parser.add_argument('--monolingual-lang', default='en', type=str,
                            help='comma separated list of languages for which we'
                                 ' want to train XLM on')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--shuffle', action='store_true',
                            help='shuffle each monolingual dataset while'
                            ' training')
        parser.add_argument('--max-source-positions', default=1024,
                            help='Max source positions')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.distributed_world_size = args.distributed_world_size
        self.lang = args.monolingual_lang
        self.default_key = self.lang

    @classmethod
    def load_dictionary(cls, filename):
        return MaskedLMDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = MaskedLMDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @property
    def target_dictionary(self):
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        dictionary = MaskedLMDictionary.load(os.path.join(args.data, 'dict.%s.txt' % args.monolingual_lang))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset_map = OrderedDict()

        path = os.path.join(self.args.data, '{}.{}'.format(split, self.lang))

        if self.args.raw_text and IndexedRawTextDataset.exists(path):
            ds = IndexedRawTextDataset(path, self.dictionary)
        elif not self.args.raw_text and IndexedDataset.exists(path):
            if self.args.lazy_load:
                ds = IndexedDataset(path, fix_lua_indexing=True)
            else:
                ds = IndexedCachedDataset(path, fix_lua_indexing=True)
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(
                split, self.args.data))

        # Since we append each block with the classification_token,
        # we need to effectively create blocks of length
        # tokens_per_sample-1
        block_dataset = TokenBlockDataset(
            dataset=ds,
            sizes=ds.sizes,
            block_size=self.args.tokens_per_sample-1,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos()
        )

        self.datasets[split] = MaskedLMDataset(
            dataset=block_dataset,
            sizes=block_dataset.sizes,
            vocab=self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.dictionary.mask(),
            classif_token_idx=self.dictionary.eos(),
            sep_token_idx=self.dictionary.eos(),
            shuffle=getattr(self.args, 'shuffle', False),
            has_pairs=False,
            seed=self.seed,
        )

        print('| {} {} {} examples'.format(
            self.args.data, split, len(self.datasets[split])
            )
        )
