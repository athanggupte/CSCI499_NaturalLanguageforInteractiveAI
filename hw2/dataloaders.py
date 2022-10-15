import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from functools import partial
from typing import Tuple
from log import get_logger

log = get_logger()

def cbow_collate_fn(batch, context_len):
    batch_x, batch_y = [], []
    # for ctx in context_len:
    ctx = context_len
    for sentence, sentence_len in batch:
        s = np.pad(sentence[:sentence_len], ctx)
        for i in range(sentence_len):
            x = np.ndarray((ctx * 2), dtype=np.int64)
            print(ctx, i, x.shape, s.shape)
            x[:ctx] = s[i:i+ctx]
            x[ctx:] = s[i+ctx+1:i+ctx+ctx+1]
            batch_x.append(x)
            batch_y.append(np.array(sentence[i]))
    batch_x, batch_y = torch.Tensor(batch_x), torch.Tensor(batch_y)
    return batch_x, batch_y

def make_cbow_collate_fn(context_len):
    return partial(cbow_collate_fn, context_len=context_len)

def skipgram_collate_fn(batch, context_len):

    pass


class LengthDataset(Dataset):
    def __init__(self, x, x_lens):
        assert(len(x) == len(x_lens))
        self.x = x
        self.x_lens = x_lens
        self.len = len(x)

    def __getitem__(self, index):
        return self.x[index], self.x_lens[index]

    def __len__(self):
        return self.len


def create_context_label_pairs(sentences, lens, context_len, pad):
    X, Y = [], []
    # for ctx in context_len:
    ctx = context_len
    for i in range(len(sentences)):
        if pad:
            s = np.pad(sentences[i][:lens[i]], ctx)
            j_range = range(lens[i])
        else:
            s = sentences[i][:lens[i]]
            j_range = range(0, lens[i] - 2 * ctx)

        for j in j_range:
            x = np.ndarray((ctx * 2), dtype=np.int64)
            x[:ctx] = s[j : j+ctx]
            x[ctx:] = s[j+ctx+1 : j+ctx+ctx+1]
            X.append(x)
            Y.append(np.array(sentences[i][j]))
    X, Y = np.array(X), np.array(Y)
    log.info("Total examples : %ld", len(X))
    log.info("Total memory : %0.2f MiB", (X.nbytes + Y.nbytes) / (1024 * 1024))
    return X, Y

def create_dataset_cbow(sentences, lens, context_len, pad=True):
    X, Y = create_context_label_pairs(sentences, lens, context_len, pad)
    X, Y = torch.LongTensor(X), torch.LongTensor(Y)
    return TensorDataset(X, Y)

def create_dataset_skipgram(sentences, lens, context_len, pad=True):
    X, Y = create_context_label_pairs(sentences, lens, context_len, True)
    X, Y = torch.LongTensor(X), torch.LongTensor(Y)
    return TensorDataset(Y, X)
        

def create_multihot_from_labels(labels, vocab_size, device):
    mhe = torch.zeros(labels.size(0), vocab_size, device=device).scatter_(1, labels, 1.)
    return mhe
