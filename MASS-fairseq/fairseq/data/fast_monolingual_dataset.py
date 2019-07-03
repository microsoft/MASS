import numpy as np
import torch

from . import FairseqDataset, data_utils


def collate(samples, pad_idx, eos_idx, lang_id):
    if len(samples) == 0:
        return {}
    
    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
        )

    src_tokens = merge('source')
    target = merge('target')

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'nsentences': len(samples),
        'ntokens': sum(len(s['source']) for s in samples),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
            'segment_label': torch.LongTensor([lang_id])
        },
        'target': target,
    }


class FastMonolingualDataset(FairseqDataset):
    def __init__(self, dataset, sizes, vocab, shuffle, lang_id):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle
        self.lang_id = lang_id

    def __getitem__(self, index):
        target = self.dataset[index]
        source, target = target[:-1], target[1:]
        return {'id': index, 'source': source, 'target': target}

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collate(samples, self.vocab.pad(), self.vocab.eos(), self.lang_id)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.sizes[index], self.sizes[index])
    
    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = max(num_tokens // tgt_len, 1)
        target = self.vocab.dummy_sentence(tgt_len + 2)
        source, target = target[:-1], target[1:]
        return self.collater([
            {'id': i, 'source': source, 'target': target}
            for i in range(bsz)
        ])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            order = [np.arange(len(self))]
            order.append(self.sizes)
            return np.lexsort(order)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]
