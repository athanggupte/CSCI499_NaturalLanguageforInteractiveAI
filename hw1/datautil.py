from log import get_logger
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

log = get_logger()

class IndexedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    
    # x_len : original length of each input sequence in the batch
    x_len = [ len(x) for x in xx ]

    # log.debug("xx: %s", xx)
    # log.debug("yy: %s", yy)

    x_tensor = pad_sequence(xx, batch_first=True, padding_value=0)
    y_tensor = torch.stack(yy)

    # log.debug("x_tensor: %s", x_tensor)
    # log.debug("y_tensor: %s", y_tensor)

    return x_tensor, y_tensor, x_len