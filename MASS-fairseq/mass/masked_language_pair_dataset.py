# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import numpy as np
import torch

from fairseq import utils

from fairseq.data import data_utils, FairseqDataset


class MaskedLanguagePairDataset(FairseqDataset):
    """Masked Language Pair dataset (only support for single language)
       [x1, x2, x3, x4, x5]
                 |
                 V
       src: [x1, _, _, x4, x5]
       tgt: [x1, x2] => [x2, x3]
    """

    def __init__(
        self, src, sizes, vocab,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, lang_id=None, ratio=None, training=True,
        pred_probs=None,
    ):
        self.src = src
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.lang_id = lang_id
        self.ratio = ratio
        self.training = training
        self.pred_probs = pred_probs

    def __getitem__(self, index):
        if self.training is False:
            source = [self.vocab.eos_index] + self.src[index].tolist()
            target, output = source[:-1], source[1:]
        else:
            src_item = self.src[index]
            src_list = [self.vocab.eos_index] + src_item.tolist()
     
            start, length = self.mask_interval(len(src_list))
            output = src_list[start     : start + length].copy()
            _target = src_list[start - 1 : start + length - 1].copy()
            
            target = []
            for w in _target:
                target.append(self.random_word(w, self.pred_probs))

            source = []
            for i, w in enumerate(src_list[1:]): # to keep consistent with finetune
                if i >= start and i <= start + length:
                    w = self.mask_word(w)
                if w is not None:
                    source.append(w)
        
        assert len(target) == len(output)
        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
        }

    def __len__(self):
        return len(self.src)

    def _collate(self, samples, pad_idx, eos_idx, segment_label):

        def merge(key, left_pad):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad,
            )
        
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        #ntokens = sum(len(s['source']) for s in samples)
        ntokens = sum(len(s['target']) for s in samples)

        prev_output_tokens = merge('target', left_pad=self.left_pad_target)
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        target = merge('output', left_pad=self.left_pad_target)
        target = target.index_select(0, sort_order)

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
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        return batch
        

    def collater(self, samples):
        return self._collate(
            samples, 
            pad_idx=self.vocab.pad(),
            eos_idx=self.vocab.eos(),
            segment_label=self.lang_id,
        )

    def get_dummy_batch(
        self, 
        num_tokens, 
        max_positions, 
        tgt_len=128
    ):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        target = self.vocab.dummy_sentence(tgt_len)
        bsz = max(num_tokens // tgt_len, 1)
        return self.collater([
            {
                'id': i,
                'source': source, 
                'target': target,
                'output': target,
            } 
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind='mergesort')]
    
    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False) and getattr(self.src, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)

    def mask_start(self, end):
        p = np.random.random()
        if p >= 0.8:
            return 1
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(1, end)

    def mask_word(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.vocab.nspecial, len(self.vocab))
        else:
            return w

    def random_word(self, w, pred_probs):
        cands = [self.vocab.mask_index, np.random.randint(self.vocab.nspecial, len(self.vocab)), w]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_interval(self, l):
        mask_length = round(l * self.ratio)
        mask_length = max(1, mask_length)
        mask_start  = self.mask_start(l - mask_length)
        return mask_start, mask_length

    def size(self, index):
        return (self.sizes[index], int(round(self.sizes[index] * self.ratio)))
