from log import get_logger
from random import randint
from train import setup_dataloader, setup_model
from utils import IndexedDataset, get_dataloader, get_device
import numpy as np

log = get_logger()
log.removeHandler(1)

class Args(object):
    pass

args = Args()
setattr(args, 'in_data_fn', 'lang_to_sem_data.json')
setattr(args, 'vocab_size', 1000)
setattr(args, 'batch_size', 32)
setattr(args, "emb_dim", 120)
setattr(args, "hidden_dim", 180)
setattr(args, "context", "curr")

log.info("Args:\n\t%s", "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))

# device = get_device()
train_loader, val_loader, maps = setup_dataloader(args)
loaders = {"train": train_loader, "val": val_loader}
# vocab_to_index, index_to_vocab, actions_to_index, index_to_actions, targets_to_index, index_to_targets = maps

log.debug("len(dataloader) : %s", len(train_loader))


# for inputs, labels, x_len in train_dataloader:
#     actions, targets = labels[:,0], labels[:,1]
#     log.debug("type(inputs) : %s | type(actions) : %s | type(targets) : %s", type(inputs), type(actions), type(targets))
#     log.debug("shape(inputs) : %s | shape(actions) : %s | shape(targets) : %s", inputs.size(), actions.size(), targets.size())
#     break
