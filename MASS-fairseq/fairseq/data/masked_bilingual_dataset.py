# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


class MaskedBilingualDataset(FairseqDataset):

    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1,  _, x3,  _, x5, y1, y2, _, y4, _] => [x2, x4]
    """

    def __init__(
        self, src, src_sizes, tgt, tgt_sizes, src_vocab, tgt_vocab,
        src_lang_id, tgt_lang_id,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, ratio=0.5, training=True, lang_tok=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.lang_tok = lang_tok
        self.ratio = ratio

    def __getitem__(self, index):
        tgt_list = [self.tgt_vocab.eos_idx] + self.tgt[index].tolist()
        src_list = self.src[index].tolist()

        src_pos_list = np.arange(len(src_list)) + 1
        tgt_pos_list = np.arange(len(tgt_list)) + 1

        src_lang_list = np.full(len(src_pos_list), self.src_lang_id)
        tgt_lang_list = np.full(len(tgt_pos_list), self.tgt_lang_id)

        pos_list  = np.concatenate([src_pos_list, tgt_pos_list])
        src_lang_list = np.concatenate([src_lang_list, tgt_lang_list]) 

        source = []
        target = []
        output = []
        positions = []

        for i, w in enumerate(src_list):
            p = np.random.random()
            if i > 0 and i < len(src_pos_list) - 1 and p >= 0.75:
                source.append(self.src_vocab.mask_index)
            else:
                source.append(w)

        for i, w in enumerate(tgt_list):
            p = np.random.random()
            if i > 0 and i < len(tgt_pos_list) - 1 and p >= 0.5:
                source.append(self.src_vocab.mask_index)
                output.append(w)
                target.append(tgt_list[i - 1])
                positions.append(tgt_pos_list[i - 1])
            else:
                source.append(w)

        assert len(source) == len(pos_list)
        assert len(source) == len(src_lang_list)

        pkg = {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'src_positions': torch.LongTensor(pos_list),
            'tgt_positions': torch.LongTensor(positions),
        }
        if self.lang_tok is True:
            pkg['src_segment_label'] = torch.LongTensor(src_lang_list)
        return pkg

    def __len__(self):
        return len(self.src)

    def _collate(self, samples, pad_idx, eos_idx, segment_label):

        def merge(key, left_pad, fill_idx=pad_idx):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                fill_idx, eos_idx, left_pad,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

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
        if self.lang_tok is True:
            batch['net_input']['segment_label'] = torch.LongTensor([segment_label])

        if 'src_positions' in samples[0]:
            # For pad, positions set as 0
            src_positions = merge('src_positions', left_pad=self.left_pad_source, fill_idx=0)
            src_positions = src_positions.index_select(0, sort_order)
            batch['net_input']['src_positions'] = src_positions

        if 'tgt_positions' in samples[0]:
            tgt_positions = merge('tgt_positions', left_pad=self.left_pad_target, fill_idx=0)
            tgt_positions = tgt_positions.index_select(0, sort_order)
            batch['net_input']['tgt_positions'] = tgt_positions
        
        if 'src_segment_label' in samples[0] and self.lang_tok is True:
            # For pad, set lang embedding as 0
            src_segment_label = merge('src_segment_label', left_pad=self.left_pad_source, fill_idx=0)
            src_segment_label = src_segment_label.index_select(0, sort_order)
            batch['net_input']['src_segment_label'] = src_segment_label

        return batch
 
    def collater(self, samples):
        return self._collate(
            samples, 
            pad_idx=self.src_vocab.pad(),
            eos_idx=self.tgt_vocab.eos(),
            segment_label=self.src_lang_id,
        )

    def get_dummy_batch(
        self,
        num_tokens,
        max_positions,
        src_len=128, tgt_len=128,
    ):
        source = self.src_vocab.dummy_sentence(src_len)
        target = self.tgt_vocab.dummy_sentence(tgt_len)
        bsz = max(num_tokens // tgt_len, 1)
        return self.collater([
            {
                'id'    : i,
                'source': source,
                'target': target,
                'output': target,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        return max(self.src_sizes[index], self.tgt_sizes[index])

    def sizes(self, index):
        return (self.src_sizes[index], self.tgt_sizes[index])

    def ordered_indices(self):
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
            and (getattr(self.tgt, 'supports_prefetch', False))
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)


class MaskedBilingualDatasetV2(MaskedBilingualDataset):
    def __init__(
        self, src, src_sizes, tgt, tgt_sizes, src_vocab, tgt_vocab,
        src_lang_id, tgt_lang_id,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, ratio=0.50, training=True, lang_tok=True,
    ):
        super().__init__(src, src_sizes, tgt, tgt_sizes, src_vocab, tgt_vocab,
                         src_lang_id, tgt_lang_id,
                         left_pad_source=left_pad_source, left_pad_target=left_pad_target,
                         max_source_positions=1024, max_target_positions=1024,
                         shuffle=shuffle, ratio=ratio, training=training, lang_tok=lang_tok)


    def __getitem__(self, index):
        tgt_list = [self.src_vocab.eos_index] + self.tgt[index].tolist()
        src_list = self.src[index].tolist()

        tgt_pos_list = np.arange(len(tgt_list)) + 1

        source = []
        target = []
        output = []
        positions = []

        for i, w in enumerate(src_list):
            p = np.random.random()
            if i > 0 and i < len(src_list) - 1 and p >= self.ratio:
                source.append(self.src_vocab.mask_index)
            else:
                source.append(w)

        for i, w in enumerate(tgt_list):
            p = np.random.random()
            if i > 0 and i < len(tgt_pos_list) - 1 and p >= self.ratio:
                output.append(w)
                target.append(tgt_list[i - 1])
                positions.append(tgt_pos_list[i - 1])


        pkg = {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'tgt_positions': torch.LongTensor(positions),
        }
        return pkg

    def _collate(self, samples, pad_idx, eos_idx, segment_label):

        def merge(key, left_pad, fill_idx=pad_idx):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                fill_idx, eos_idx, left_pad,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

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
        if self.lang_tok is True:
            batch['net_input']['segment_label'] = torch.LongTensor([segment_label])

        if 'tgt_positions' in samples[0]:
            tgt_positions = merge('tgt_positions', left_pad=self.left_pad_target, fill_idx=0)
            tgt_positions = tgt_positions.index_select(0, sort_order)
            batch['net_input']['tgt_positions'] = tgt_positions
        
        return batch

