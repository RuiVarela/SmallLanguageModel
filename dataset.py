import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class TokenDataset(Dataset):
    def __init__(self, filename, split, block_size):
        assert split in {'train', 'test'}

        self.split = split
        self.block_size = block_size
    
        try:
            data = np.fromfile(filename, dtype=np.int16)
        except Exception as e:
            logging.error(f"Unable to token data set {filename}: {e}")
            data = np.array([])

        train_size = int(0.7 * data.shape[0])
        train_data = data[:train_size]
        test_data = data[train_size:]
        #logging.info(f"Train tokens {train_data.shape[0]}")
        #logging.info(f"Test tokens {test_data.shape[0]}")

        self.data = train_data if split == "train" else test_data
        self.data = torch.tensor(self.data, dtype=torch.long)

        #logging.info(f"Data tokens {self.data.shape[0]} block_size={self.block_size} len={len(self)}")

    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        # grab the current sample with target
        d = self.data[idx:idx + self.block_size + 1]

        # return as tensors
        x = d[:-1]
        y = d[1:]
        return x, y
    
def generate_dataloader(filename, split, config):
    dataset = TokenDataset(filename, split, config["block_size"])

    loader = DataLoader(dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False, pin_memory=True,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    return loader