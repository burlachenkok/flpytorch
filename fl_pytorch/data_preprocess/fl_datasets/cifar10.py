from torchvision.datasets import CIFAR10
from torchvision import transforms

from PIL import Image
import numpy as np

class FLCifar10(CIFAR10):
    """
    CIFAR10 Dataset.
    num_clients clients that were allocated data_preprocess uniformly at random.
    """
    def __init__(self, exec_ctx, args, root, train=True, transform=None, target_transform=None, download=False, client_id=None):

        super(FLCifar10, self).__init__(root, train=train, transform=transform,
                                        target_transform=target_transform, download=download)
        self.num_clients = 10
        self.dataset_indices = np.arange(len(self.data))
        exec_ctx.np_random.shuffle(self.dataset_indices)
        self.n_client_samples = len(self.data) // self.num_clients

        if args.sort_dataset_by_class_before_split:
            indicies = np.argsort(self.targets)
            self.targets = np.asarray(self.targets)[indicies]
            self.data = self.data[indicies, ...]

        #self.my_data = self.data
        #self.my_targets = self.targets

        #self.store_in_target_device = args.store_data_in_target_device
        # Move data to GPU maybe
        #self.my_targets = torch.tensor(self.targets, dtype=torch.long)
        #n, h, w, c = self.data.shape

        #self.my_data = torch.zeros(n, c, h, w)
        # Copy data to target device
        #for i in range(n):
        #    self.my_data[i,...] = transforms.functional.to_tensor(self.data[i])

        # Move data to target device
        #if self.store_in_target_device:
        #    self.my_targets = self.my_targets.to(device = args.device)
        #    self.my_data = self.my_data.to(device = args.device)

        # self.data = self.my_data
        # self.targets = self.my_targets

        self.set_client(client_id)

    def set_client(self, index=None):
        """ Set current client

        Args:
            index(int): index of current client. If index is None the partitioned dataset is considered as one single dataset

        Args:
            int: Numer of train points for a current client
        """
        if index is None:
            self.client_id = None
            self.length = len(self.data)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.n_client_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of item that is fetched on behalf on current setuped client

        Args:
            tuple: (image, target) where target is index of the target class.
        """
        if self.client_id is None:
            actual_index = index
        else:
            actual_index = int(self.client_id) * self.n_client_samples + index

        img, target = self.data[actual_index], self.targets[actual_index]

        # doing this so that it is consistent with all other fl_datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # TODO: If __getitem__ will always fetch object from the CPU memory. Suggestion use GPU memory or another GPU as a cache storage
        return img, target

    def __len__(self):
        """ Get length of dataset for a current client
        Returns:
            int: Numer of train points for a current client
        """
        return self.length
