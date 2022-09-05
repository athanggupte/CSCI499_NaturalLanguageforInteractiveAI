from log import get_logger
from random import randint
from train import setup_dataloader, setup_model
from utils import encode_data, extract_episodes_from_json, get_device
from datautil import IndexedDataset, get_dataloader

log = get_logger()

class Args(object):
    pass

args = Args()
setattr(args, 'in_data_fn', 'lang_to_sem_data.json')
setattr(args, 'vocab_size', 10000)
setattr(args, 'batch_size', 32)
setattr(args, "emb_dim", 120)
setattr(args, "lstm_dim", 120)


device = get_device(False)
train_loader, val_loader, maps = setup_dataloader(args)
loaders = {"train": train_loader, "val": val_loader}
log.debug("len(train_loader.dataset) : %d", len(train_loader.dataset))

exit(0)

# build model
model = setup_model(args, device, maps[0], maps[2], maps[4])
log.info(model)


# for inputs, labels, x_len in train_dataloader:
#     actions, targets = labels[:,0], labels[:,1]
#     log.debug("type(inputs) : %s | type(actions) : %s | type(targets) : %s", type(inputs), type(actions), type(targets))
#     log.debug("shape(inputs) : %s | shape(actions) : %s | shape(targets) : %s", inputs.size(), actions.size(), targets.size())
#     break
