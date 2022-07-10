#!/usr/bin/env python3

import os
import h5py

from .read_file_cache import cacheItemThreadUnsafe, cacheMakeKey, cacheGetItem
from .fl_dataset import FLDataset
from torchvision.datasets.utils import download_url
from utils import execution_context

TFF_DATASETS = {
    'cifar100_fl': 'https://storage.googleapis.com/tff-datasets-public/fed_cifar100.tar.bz2',
    'femnist': 'https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2',
    'shakespeare': 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2'
}

class H5TFFDataset(FLDataset):
    """
    Based FL class that loads H5 type data_preprocess.
    """
    def __init__(self, h5_path, client_id, dataset_name, data_key, download=True):
        """
        Ctor.

        Args:
            h5_path (str): path to HDF5 file with dataset. Not native for systems like TensorFlow
            client_id (int): switch dataset to work view of client client_id
            data_key(str): if h5_path is not in the filesystem and download is True then it will an attempt to download dataset from TFF_DATASETS[data_key] URL
            download(bool): allow to download dataset
        """
        self.h5_path = h5_path

        # Global lock just prevent the case when two thread simulationanously download the same file
        execution_context.torch_global_lock.acquire()

        if not os.path.isfile(h5_path):
            one_up = os.path.dirname(h5_path)
            target = os.path.basename(TFF_DATASETS[dataset_name])
            if download:
                download_url(TFF_DATASETS[dataset_name], one_up)

            def extract_bz2(filename, path="."):
                import tarfile
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(path)
            taret_file = os.path.join(one_up, target)
            if os.path.isfile(taret_file):
                extract_bz2(os.path.join(one_up, target), one_up)
            else:
                raise ValueError(f"{taret_file}: does not exists, set `download=True`.")

        execution_context.torch_global_lock.release()

        self.dataset = None
        self.clients = list()               # list of client ids
        self.clients_num_data = dict()      # number of data points for client. Key: client-id, Value: number of sample points
        self.client_and_indices = list()    # List of tuplies (first, second) where first - client_id, second - local data index

        with h5py.File(self.h5_path, 'r') as file:
            data = file['examples']
            for client in list(data.keys()):
                self.clients.append(client)
                n_data = len(data[client][data_key])
                for i in range(n_data):
                    self.client_and_indices.append((client, i))
                self.clients_num_data[client] = n_data

        self.num_clients = len(self.clients)        # Total number of clients
        self.length = len(self.client_and_indices)  # Total number of data points for all clients and for all data that belong to them

        self.set_client(client_id)

    def set_client(self, index=None):
        """
        Set pointer to client's data_preprocess corresponding to index. If index is none complete dataset as union of all datapoint will be observable by higher level

        Args:
            index(int): index of client.
        """
        if index is None:
            self.client_id = None
            self.length = len(self.client_and_indices)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.clients_num_data[self.clients[index]]

    def load_data(self):
        """
        Explicit load all need datasets from filesystem or from cache for specific dataset instance
        """
        if self.dataset is None:
            cacheKey = cacheMakeKey("examples frame", self.h5_path)
            cache = cacheGetItem(cacheKey)

            if cache != None:
                self.dataset = cache
            else:
                self.dataset = h5py.File(self.h5_path, 'r')["examples"]
                cacheItemThreadUnsafe(cacheKey, self.dataset)

    def _get_item_preprocess(self, index):
        # loading in getitem allows us to use multiple processes for data_preprocess loading
        # because hdf5 files aren't pickelable so can't transfer them across processes
        # https://discuss.pytorch.org/t/hdf5-a-data-format-for-pytorch/40379
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')["examples"]
        if self.client_id is None:
            client, i = self.client_and_indices[index]
        else:
            client, i = self.clients[self.client_id], index
        return client, i

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.length
