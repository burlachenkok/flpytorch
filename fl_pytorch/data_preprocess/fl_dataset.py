#!/usr/bin/env python3

from torch.utils.data import Dataset


class FLDataset(Dataset):
    """
    Base class for Federated Datasets with pointers to clients.
    """
    def set_client(self, index=None):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError
