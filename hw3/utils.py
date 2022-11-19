import re
import torch
import numpy as np
from collections import Counter
from log import get_logger
import json

log = get_logger()

def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        padded_len = 2  # start/end
        for inst, _ in episode:
            inst = preprocess_string(inst)
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
        padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        # int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
        int(np.max(padded_lens)),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    max_len = 0

    for episode in train:
        max_len = max(max_len, len(episode) + 2)
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i+3 for i, a in enumerate(actions)}
    targets_to_index = {t: i+3 for i, t in enumerate(targets)}
    actions_to_index['<pad>'], targets_to_index['<pad>'] = 0, 0
    actions_to_index['<start>'], targets_to_index['<start>'] = 1, 1
    actions_to_index['<stop>'], targets_to_index['<stop>'] = 2, 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets, max_len

def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    seq_length = len(gt_labels)
    
    for i in range(seq_length):
        if torch.any(predicted_labels[i] != gt_labels[i]):
            break
    
    pm = (1.0 / seq_length) * i

    return pm


# Extra utility functions
def extract_episodes_from_json(filename):
    with open(filename, 'r') as file:
        contents = json.load(file)
        train_data, val_data = contents['train'], contents['valid_seen']
        log.info("Train #episodes: %d" % len(train_data))
        log.info("Val #episodes: %d" % len(val_data))
        return train_data, val_data

def flatten_episodes(episodes):
    data = []
    padded_lens = []
    for episode in episodes:
        instrs = []
        outputs = []
        for i, (inst, outseq) in enumerate(episode):
            inst = preprocess_string(inst).lower()
            action, target = outseq
            
            # padded_len = len([ word for word in inst.split() if len(word) > 0 ])
            # padded_lens.append(padded_len + 2)

            instrs.append(inst)
            outputs.append((action, target))
        data.append((instrs, outputs))
    return data #, int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),

def encode_data(data, v2i, a2i, t2i, seq_len=0):
    n_data = len(data)
    x = [None] * n_data
    y = [None] * n_data

    idx = 0
    n_tokens = 0
    n_unks = 0
    n_early_cutoffs = 0
    for instrs, outputs in data:
        # inst = preprocess_string(inst)
        inst = ' '.join(instrs)
        tokens = inst.split()
        tkn_idx = [ v2i['<start>'] ]
        for word in tokens:
            if len(word) > 0:
                tkn_idx.append(v2i[word] if word in v2i else v2i['<unk>'])
                n_tokens += 1
                n_unks += 1 if tkn_idx[-1] == v2i['<unk>'] else 0
                if seq_len > 0 and len(tkn_idx) == seq_len - 1:
                    n_early_cutoffs += 1
                    break
        tkn_idx.append(v2i['<end>'])
        x[idx] = np.array(tkn_idx, dtype=np.int32)
        
        out_idx = [ [a2i['<start>'], t2i['<start>']] ]
        for action, target in outputs:
            out_idx.append([a2i[action], t2i[target]])
        out_idx.append([a2i['<stop>'], t2i['<stop>']])

        y[idx] = np.array(out_idx, dtype=np.int32)
        idx += 1
    log.info(
        "Total instances: %d" % n_data
    )
    log.info(
        "UNK tokens : %d / %d (%.4f)     (vocab_size = %d)"
        % (n_unks, n_tokens, n_unks/n_tokens, len(v2i))
    )
    log.info(
        "Cut off %d instances at len %d before true ending"
        % (n_early_cutoffs, seq_len)
    )
    log.info(
        "encoded %d instances without regard to order" % idx
    )
    return x, y


# Dataloader functions
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class IndexedDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.xlen = [ len(_x_) for _x_ in x ]
        self.ylen = [ len(_y_) for _y_ in y ]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.xlen[index], self.ylen[index]
    
    def __len__(self):
        return len(self.x)

def get_dataloader(dataset:Dataset, batch_size: int, shuffle: bool=True) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate, num_workers=1)

def pad_collate(batch):
    (xx, yy, x_lens, y_lens) = zip(*batch)

    x_tensor = pad_sequence(xx, batch_first=True, padding_value=0)
    y_tensor = pad_sequence(yy, batch_first=True, padding_value=0)

    return x_tensor, x_lens, y_tensor, y_lens
