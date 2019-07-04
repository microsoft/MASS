# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, src_id=None, tgt_id=None, lang_tok=None,
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
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
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
    if src_id is not None and lang_tok is True:
        batch['net_input']['src_segment_label'] = torch.LongTensor([src_id])
    if tgt_id is not None and lang_tok is True:
        batch['net_input']['segment_label'] = torch.LongTensor([tgt_id])
    return batch


def generate_dummy_batch(num_tokens, collate_fn, vocab, src_len=128, tgt_len=128):
    """Return a dummy batch with a given number of tokens."""
    bsz = num_tokens // max(src_len, tgt_len)
    return collate_fn([
        {
            'id': i,
            'source': vocab.dummy_sentence(src_len),
            'target': vocab.dummy_sentence(tgt_len),
            'output': vocab.dummy_sentence(tgt_len),
        }
        for i in range(bsz)
    ])



class NoisyLanguagePairDataset(FairseqDataset):
    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1,  _,  _,  _, x5] [y1, y2, y3, y4, y5]
    add noisy decoder input
    """
    def __init__(
        self, src, src_sizes, tgt, tgt_sizes, src_vocab, tgt_vocab,
        src_lang_id, tgt_lang_id,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, lang_tok=True, ratio=0.5, pred_probs=None,
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
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.lang_tok = lang_tok
        self.ratio = ratio
        self.pred_probs = pred_probs

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]
        src_list = src_item.tolist()
        tgt_list = tgt_item.tolist()

        start, length = self.mask_interval(len(src_list))

        source = []
        for i, w in enumerate(src_list):
            if i >= start and i <= start + length:
                w = self.mask_word(w, self.src_vocab)
            if w is not None:
                source.append(w)

        target = []
        for w in tgt_list[:-1]:
            target.append(self.random_word(w, self.pred_probs, self.tgt_vocab))
        target.append(self.tgt_vocab.eos_index)
        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def _collate(
        self, samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True, src_id=None, tgt_id=None, lang_tok=None,
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
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        if samples[0].get('output', None) is not None:
            target = merge('output', left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            ntokens = sum(len(s['output']) for s in samples)
        else:
            ntokens = sum(len(s['source']) for s in samples)

        if samples[0].get('target', None) is not None:
            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                # if samples[0].get('target')[-1] != eos_idx:
                #     samples[0].get('target')[-1] = eos_idx
                prev_output_tokens = merge(
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

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
        if src_id is not None and lang_tok is True:
            batch['net_input']['src_segment_label'] = torch.LongTensor([src_id])
        if tgt_id is not None and lang_tok is True:
            batch['net_input']['segment_label'] = torch.LongTensor([tgt_id])
        return batch


    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return self._collate(
            samples, pad_idx=self.src_vocab.pad(), eos_idx=self.src_vocab.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, src_id=self.src_lang_id, tgt_id=self.tgt_lang_id,
            lang_tok=self.lang_tok
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return generate_dummy_batch(num_tokens, self.collater, self.src_vocab, src_len, tgt_len)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

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


    def mask_start(self, end):
        p = np.random.random()
        if p >= 0.8:
            return 0
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(0, end)

    def mask_word(self, w, vocab):
        p = np.random.random()
        if p >= 0.2:
            return vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(vocab.nspecial, len(vocab))
        else:
            return w

    def random_word(self, w, pred_probs, vocab):
        cands = [vocab.mask_index, np.random.randint(vocab.nspecial, len(vocab)), w]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_interval(self, l):
        mask_length = round(l * self.ratio)
        mask_length = max(1, mask_length)
        if l - mask_length <= 1:
            mask_start = 1
        else:
            mask_start  = self.mask_start(l - mask_length)
        return mask_start, mask_length

    def size(self, index):
        return (self.sizes[index], int(round(self.sizes[index] * self.ratio)))


class NoisyLanguagePairDatasetV2(FairseqDataset):
    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1,  _,  _,  _, x5] [y3, y4]
    add noisy decoder input
    """

    def __init__(
        self, src, src_sizes, tgt, tgt_sizes, src_vocab, tgt_vocab,
        src_lang_id, tgt_lang_id,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, lang_tok=True, ratio=0.5, pred_probs=None,
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
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.lang_tok = lang_tok
        self.ratio = ratio
        self.pred_probs = pred_probs

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]
        src_list = [self.src_vocab.eos_index] + src_item.tolist()
        tgt_list = [self.tgt_vocab.eos_index] + tgt_item.tolist()

        tgt_pos_list = np.arange(len(tgt_list)) + 1

        source = []
        output = []
        positions = []

        start, length = self.mask_interval(len(src_list))
        for i, w in enumerate(src_list[1:]):
            if i >= start and i <= start + length:
                w = self.mask_word(w, self.src_vocab)
            if w is not None:
                source.append(w)

        start, length = self.mask_interval(len(tgt_list))
        output = tgt_list[start : start + length].copy()
        positions = tgt_pos_list[start - 1 : start + length - 1].copy()
        _target = tgt_list[start - 1 : start + length - 1].copy()
        target = [_target[0]]
        for w in _target[1:]:
            target.append(self.random_word(w, self.pred_probs, self.tgt_vocab))

        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'tgt_positions': torch.LongTensor(positions),
        }

    def __len__(self):
        return len(self.src)

    def _collate(
        self, samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True, src_id=None, tgt_id=None, lang_tok=None,
    ):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False, fill_idx=pad_idx):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                fill_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        if samples[0].get('output', None) is not None:
            target = merge('output', left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            ntokens = sum(len(s['output']) for s in samples)
        else:
            ntokens = sum(len(s['source']) for s in samples)

        if samples[0].get('target', None) is not None:
            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                # if samples[0].get('target')[-1] != eos_idx:
                #     samples[0].get('target')[-1] = eos_idx
                prev_output_tokens = merge(
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=False,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

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
        if src_id is not None and lang_tok is True:
            batch['net_input']['src_segment_label'] = torch.LongTensor([src_id])
        if tgt_id is not None and lang_tok is True:
            batch['net_input']['segment_label'] = torch.LongTensor([tgt_id])
        if 'tgt_positions' in samples[0]:
            tgt_positions = merge('tgt_positions', left_pad=self.left_pad_target, fill_idx=0)
            tgt_positions = tgt_positions.index_select(0, sort_order)
            batch['net_input']['tgt_positions'] = tgt_positions
        return batch


    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return self._collate(
            samples, pad_idx=self.src_vocab.pad(), eos_idx=self.src_vocab.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, src_id=self.src_lang_id, tgt_id=self.tgt_lang_id,
            lang_tok=self.lang_tok
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return generate_dummy_batch(num_tokens, self.collater, self.src_vocab, src_len, tgt_len)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

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


    def mask_start(self, end):
        p = np.random.random()
        if p >= 0.8:
            return 1
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(1, end)

    def mask_word(self, w, vocab):
        p = np.random.random()
        if p >= 0.2:
            return vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(vocab.nspecial, len(vocab))
        else:
            return w

    def random_word(self, w, pred_probs, vocab):
        cands = [vocab.mask_index, np.random.randint(vocab.nspecial, len(vocab)), w]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_interval(self, l):
        mask_length = round(l * self.ratio)
        mask_length = max(1, mask_length)
        if l - mask_length <= 1:
            mask_start = 1
        else:
            mask_start  = self.mask_start(l - mask_length)
        return mask_start, mask_length

    def size(self, index):
        return (self.sizes[index], int(round(self.sizes[index] * self.ratio)))

