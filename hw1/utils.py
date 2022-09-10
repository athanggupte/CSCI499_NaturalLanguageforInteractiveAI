from log import get_logger
import json
import re
import torch
import numpy as np
from collections import Counter

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
            log.info("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            log.info("Using CPU")
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
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
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
        # int(np.max(padded_lens))
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


# Extra utility functions
def extract_episodes_from_json(filename):
    with open(filename, 'r') as file:
        contents = json.load(file)
        train_data, val_data = contents['train'], contents['valid_seen']
        log.info("Train #episodes: %d" % len(train_data))
        log.info("Val #episodes: %d" % len(val_data))
        return train_data, val_data

def flatten_episodes(episodes, context):
    data = []
    padded_lens = []
    for episode in episodes:
        for i, (inst, outseq) in enumerate(episode):
            inst = preprocess_string(inst).lower()
            action, target = outseq
            
            if context != "curr":
                prev = ("" if i == 0 else preprocess_string(episode[i-1][0]).lower() + " ") + "<end> <start> "
                next = " <end> <start>" + ("" if i == len(episode) - 1 else " " + preprocess_string(episode[i+1][0]).lower())

                if "prev" in context:
                    inst = prev + inst
                if "next" in context:
                    inst = inst + next
            
            padded_len = len([ word for word in inst.split() if len(word) > 0 ])
            padded_lens.append(padded_len + 2)
            data.append((inst, action, target))
    return data, int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),

def encode_data(data, v2i, a2i, t2i, seq_len=0):
    n_data = len(data)
    # x = np.zeros((n_data, seq_len), dtype=np.int32)
    y = np.zeros((n_data, 2), dtype=np.int32)
    x = [None for i,a,t in data]

    idx = 0
    n_tokens = 0
    n_unks = 0
    n_early_cutoffs = 0
    for inst, action, target in data:
        # inst = preprocess_string(inst)
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
        y[idx][0] = a2i[action]
        y[idx][1] = t2i[target]
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

def log_examples(inputs, action_preds, action_labels, target_preds, target_labels, index_to_vocab, index_to_actions, index_to_targets, num_examples=10):
    k_random_samples = np.random.choice(len(inputs), size=num_examples, replace=False)
    for idx in k_random_samples:
        log.info("input :\t%s" % ' '.join([index_to_vocab[i] for i in inputs[idx].tolist()]))
        log.info("    true action :\t%s" % index_to_actions[action_labels[idx]])
        log.info("    pred action :\t%s" % index_to_actions[action_preds[idx]])
        log.info("    true target :\t%s" % index_to_targets[target_labels[idx]])
        log.info("    pred target :\t%s" % index_to_targets[target_preds[idx]])