import os
import numpy as np

# Import PyTorch root package import torch
import torch

from torch.utils.data import DataLoader

from ..h5_tff_dataset import H5TFFDataset
from ..fl_dataset import FLDataset

SHAKESPEARE_VOCAB = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
SHAKESPEARE_EVAL_BATCH_SIZE = 4


class Shakespeare(FLDataset):
    """
    Shakespeare Dataset containing dialogs from his books.
    Clients corresponds to different characters.
    """
    # TODO: needs to be revisited, especially test loader, necessary for personalized FL
    def __init__(self, data_path, train=True, batch_size=SHAKESPEARE_EVAL_BATCH_SIZE, client_id=None):
        self.train = train
        if train:
            data_path = os.path.join(data_path, 'shakespeare/shakespeare_train.h5')
        else:
            data_path = os.path.join(data_path, 'shakespeare/shakespeare_test.h5')
        self.batch_size = batch_size
        self.dataset = ShakespeareH5(data_path)
        self.dummy_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)

        self.num_clients = self.dataset.num_clients
        self.train = train

        self.available_clients = list()
        self.data = dict()
        self.clients_num_data = dict()
        self.client_and_indices = list()

        if train:
            self._add_client_train(client_id)
        else:
            self._add_test()
        self.set_client(client_id)

    def _add_client_train(self, client_id):
        client_ids = range(self.num_clients) if client_id is None else [client_id]
        for cid in client_ids:
            if cid in self.available_clients:
                continue
            self.dataset.set_client(cid)
            x_data = torch.cat([x[0] for x, y in self.dummy_loader], dim=0)
            y_data = torch.cat([y[0] for x, y in self.dummy_loader], dim=0)
            self._update_data(cid, x_data, y_data)

    def _add_test(self):
        """
        Add test data_preprocess and reshape in a such way that subsequent batches correspond
        to the same data_preprocess because of the hidden state.
        :return:
        """
        self.dataset.set_client(None)
        x_data = torch.cat([x[0] for x, y in self.dummy_loader], dim=0)
        y_data = torch.cat([y[0] for x, y in self.dummy_loader], dim=0)
        # reorder data_preprocess  such that consequent batches follow speech order
        n_zeros = int(np.ceil(len(x_data) / self.batch_size) * self.batch_size) - len(x_data)
        # append zeros if necessary
        x_data = torch.cat([x_data, torch.zeros(n_zeros, self.dataset.seq_len).long()], dim=0)
        y_data = torch.cat([y_data, torch.zeros(n_zeros, self.dataset.seq_len).long()], dim=0)

        order = np.arange(len(x_data))
        order = order.reshape(self.batch_size, -1).T.reshape(-1)
        x_data, y_data = x_data[order], y_data[order]
        self._update_data(None, x_data, y_data)

    def _update_data(self, cid, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        if self.train:
            self.available_clients.append(cid)
            self.clients_num_data[cid] = x_data.shape[0]
        self.data[cid] = (x_data, y_data)
        self.client_and_indices.extend([(cid, i) for i in range(x_data.shape[0])])

    def _get_item_preprocess(self, index):
        if self.client_id is None:
            client, i = self.client_and_indices[index]
        else:
            client, i = self.client_id, index
        return client, i

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            if self.train and len(self.available_clients) < self.num_clients:
                self._add_client_train(index)
            self.length = len(self.client_and_indices)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            if self.train:
                if index not in self.available_clients:
                    self._add_client_train(index)
            else:
                raise ValueError('Individual clients are not supported for test set.')
            self.length = self.clients_num_data[index]

    def __getitem__(self, index):
        client, i = self._get_item_preprocess(index)
        return tuple(tensor[i] for tensor in self.data[client])

    def __len__(self):
        return self.length


class ShakespeareH5(H5TFFDataset):
    """
    Preprocessing for Shakespeare h5 Dataset.
    Text to Integer encoding.
    """
    def __init__(self, h5_path, cliend_id=None, seq_len=80):
        super(ShakespeareH5, self).__init__(h5_path, cliend_id, 'shakespeare', 'snippets')
        self.seq_len = seq_len
        # vocabulary
        self.vocab = SHAKESPEARE_VOCAB
        self.char2idx = {u: i for i, u in enumerate(self.vocab, 1)}
        self.idx2char = {i: u for i, u in enumerate(self.vocab, 1)}
        # out of vocabulary, beginning and end of speech
        self.oov = len(self.vocab) + 1
        self.bos = len(self.vocab) + 2
        self.eos = len(self.vocab) + 3

    def __getitem__(self, index):
        client, i = self._get_item_preprocess(index)
        record = self.dataset[client]['snippets'][i].decode()

        indices = np.array([self.char2idx[char] if char in self.char2idx else self.oov for char in record])
        len_chars = 1 + len(indices)  # beginning of speech
        pad_size = int(np.ceil(len_chars/self.seq_len) * self.seq_len - len_chars)
        indices = np.concatenate(([self.bos], indices, [self.eos], torch.zeros(pad_size)), axis=0)
        x = torch.from_numpy(indices[:-1]).reshape(-1, self.seq_len)
        y = torch.from_numpy(indices[1:]).reshape(-1, self.seq_len)
        return x.long(), y.long()
