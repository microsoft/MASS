import numpy as np
import torch
import random
import time
import math

from fairseq import utils
from fairseq.data import data_utils, LanguagePairDataset


class MaskedLanguagePairDataset(LanguagePairDataset):
    """ Wrapper for masked language datasets 
        (support monolingual and bilingual)

        For monolingual dataset:
        [x1, x2, x3, x4, x5] 
                 ||
                 VV
        [x1,  _,  _, x4, x5] => [x2, x3]

        default,  _ will be replaced by 8:1:1 (mask, self, rand),
    """
    def __init__(
        self,
        src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, mask_prob=0.15, pred_probs=None, block_size=64,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = src_sizes
        self.tgt_sizes = tgt_sizes
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle

        self.mask_prob = mask_prob
        self.pred_probs = pred_probs
        self.block_size = block_size
        
    def __getitem__(self, index):
        pkgs = {'id': index}
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        
        positions = np.arange(0, len(self.src[index]))
        masked_pos = []
        for i in range(1, len(src_item), self.block_size):
            block = positions[i: i + self.block_size]
            masked_len = int(len(block) * self.mask_prob)
            masked_block_start = np.random.choice(block[:len(block) - int(masked_len) + 1], 1)[0]
            masked_pos.extend(positions[masked_block_start : masked_block_start + masked_len])
        masked_pos = np.array(masked_pos)

        pkgs['target'] = src_item[masked_pos].clone()
        pkgs['prev_output_tokens'] = src_item[masked_pos - 1].clone()
        pkgs['positions'] = torch.LongTensor(masked_pos) + self.src_dict.pad_index
        src_item[masked_pos] = self.replace(src_item[masked_pos])
        pkgs['source'] = src_item
        return pkgs

    def collate(self, samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False):
        if len(samples) == 0:
            return {}

        def merge(x, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                x, pad_idx, eos_idx, left_pad, move_eos_to_beginning
            )
        
        id = torch.LongTensor([s['id'] for s in samples])
        source = merge([s['source'] for s in samples], left_pad=left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

        prev_output_tokens = merge([s['prev_output_tokens'] for s in samples], left_pad=left_pad_target)
        positions = merge([s['positions'] for s in samples], left_pad=left_pad_target)
        target = merge([s['target'] for s in samples], left_pad=left_pad_target)
        ntokens = target.numel()

        batch = {
            'id' : id,
            'nsentences': len(samples),
            'net_input' : {
                'src_lengths': src_lengths,
                'src_tokens' : source,
                'prev_output_tokens': prev_output_tokens,
                'positions'  : positions,
            },
            'target' : target,
            'ntokens': ntokens,
        }
        return batch

    def collater(self, samples):
        return self.collate(samples, self.src_dict.pad(), self.src_dict.eos())

    def size(self, index):
        return self.src.sizes[index]

    def replace(self, x):
        _x_real = x
        _x_rand = _x_real.clone().random_(self.src_dict.nspecial, len(self.src_dict))
        _x_mask = _x_real.clone().fill_(self.src_dict.index('[MASK]'))
        probs = torch.multinomial(self.pred_probs, len(x), replacement=True)
        _x = _x_mask * (probs == 0).long() + \
             _x_real * (probs == 1).long() + \
             _x_rand * (probs == 2).long()
        return _x
