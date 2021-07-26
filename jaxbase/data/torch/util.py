from torch.utils.data import DataLoader
import numpy as np
from functools import partial

def collate_fn(batch):
    images, labels = zip(*batch)
    return np.stack(images), np.stack(labels)

NumpyLoader = partial(DataLoader, collate_fn=collate_fn)