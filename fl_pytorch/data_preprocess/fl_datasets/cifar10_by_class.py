from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch

class FLCifar10ByClass(CIFAR10):
    """
    CIFAR10 Dataset.
    num_clients clients that were allocated data_preprocess uniformly at random.
    """
    def __init__(self, exec_ctx, args, root, train=True, transform=None, target_transform=None, download=False, client_id=None):

        super(FLCifar10ByClass, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.num_clients = 10

        num_datapoints = len(self.targets) # Total number of datapoints in dataset
        num_datapoints_per_class = {}      # Key - class, Value - number of samples of specific class
        class_to_points_indicies = {}      # Key - class, Value - list of point indicies

        for i in range(num_datapoints):
            if self.targets[i] in num_datapoints_per_class:
                num_datapoints_per_class[self.targets[i]] += 1
                class_to_points_indicies[self.targets[i]].append(i)
            else:
                num_datapoints_per_class[self.targets[i]] = 0
                class_to_points_indicies[self.targets[i]] = list([i])

        classes = list(class_to_points_indicies.keys())
        assert len(classes) == 10

        # Spread each class across p(10%) clients
        p = 0.10
        client_id_to_classes = classes * int(self.num_clients * p)
        exec_ctx.np_random.shuffle(client_id_to_classes)

        classes_to_client_id = {}
        for c in classes:
            classes_to_client_id[c] = list()
        for i, c in enumerate(client_id_to_classes):
            classes_to_client_id[c].append(i % self.num_clients)

        # Reshuflle point indices
        for c in classes:
            exec_ctx.np_random.shuffle(class_to_points_indicies[c])

        # Finally split data across clients
        self.datapoints_per_client = {}
        for client in range(self.num_clients):
            self.datapoints_per_client[client] = list()

        for c, clients in classes_to_client_id.items():
            points = class_to_points_indicies[c]
            num_clients_for_class = len(clients)
            #assert num_clients_for_class == 10

            num_points_per_client = len(points) // num_clients_for_class

            for k in range(num_clients_for_class):
                subpoints = points[(k)*num_points_per_client : (k+1)*num_points_per_client]
                self.datapoints_per_client[clients[k]] += subpoints

        # Reshuffle data points
        for client in range(self.num_clients):
            exec_ctx.np_random.shuffle(self.datapoints_per_client[client])

        # Mapping between classes and client ids'
        self.classes_to_client_id = classes_to_client_id

        # Total number of samples
        self.total_samples = num_datapoints

        # self.store_in_target_device = args.store_data_in_target_device
        #self.targets = torch.Tensor(self.targets)
        #self.data = torch.Tensor(self.data)

        # Move data to GPU maybe
        # Move data to target device
        #if self.store_in_target_device:
        #    self.targets = self.targets.to(device = args.device)
        #    self.data = self.data.to(device = args.device)
        # ==============================================================================================================
        # test
        make_test = True
        if make_test:
            for i in range(10):
                cc1 = self.get_clients_that_stores_class_naively(2)
                cc2 = self.get_clients_that_stores_class(2)
                assert cc1 == cc2
        # ==============================================================================================================
        self.set_client(client_id)

    def get_clients_that_stores_class(self, k):
        return list(set(self.classes_to_client_id[k]))

    def get_clients_that_stores_class_naively(self, k):
        clients = []

        for i, c in enumerate(self.targets):
            if c != k:
                continue
            for client in range(self.num_clients):
                if i in self.datapoints_per_client[client]:
                    clients.append(client)
                    break

        return list(set(clients))

    def set_client(self, index=None):
        """ Set current client.

        Args:
            index(int): index of current client. If index is None the partitioned dataset is considered as
                        one single dataset

        Returns:
            int: Numer of train points for a current client
        """
        if index is None:
            self.client_id = None
            self.length = len(self.data)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = len(self.datapoints_per_client[index])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of item that is fetched on behalf on current setuped client

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.client_id is None:
            actual_index = index
        else:
            actual_index = self.datapoints_per_client[self.client_id][index]
        img, target = self.data[actual_index], self.targets[actual_index]

        # doing this so that it is consistent with all other fl_datasets to return a PIL Image
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
