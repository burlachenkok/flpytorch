#!/usr/bin/env python3

# Import PyTorch root package import torch
import torch

import torchvision

from torchvision import transforms
from .fl_datasets import FEMNIST, FLCifar100, FLCifar10, FLCifar10ByClass, Shakespeare, SHAKESPEARE_EVAL_BATCH_SIZE
from .artificial_dataset import ArificialDataset
from .libsvm_dataset import LibSVMDataset

CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def get_torch_version() -> int:
    """
    Get PyTorch library version
    """
    return int(torch.__version__.split('+')[0].replace('.', ''))


def load_data(exec_ctx, path, dataset, args, load_trainset=True, download=True, client_id=None):
    """
    Load dataset.

    Args:
      exec_ctx: Execution context that maybe required for pseudo random generations
      path: path to dataset
      args: command line arguments
      load_trainset: Load train dataset or test dataset.
      download: If dataset is not presented in filesystem download it from the web
      client_id: Specified id of the client on bhalf of which dataset will be used

    Returns:
      None
    """
    dataset = dataset.lower()
    trainset = None

    if (client_id is not None and client_id < 0) or dataset in ['emnist', 'full_shakespeare']:
        client_id = None

    if dataset.startswith("cifar"):  # CIFAR-10/100

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

        if dataset == "cifar10":
            if load_trainset:
                trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=download,
                                                        transform=transform_train)
                trainset.num_clients = get_num_clients(dataset)

            testset = torchvision.datasets.CIFAR10(root=path, train=False, download=download, transform=transform_test)

        elif dataset == "cifar10_fl":
            if load_trainset:
                trainset = FLCifar10(exec_ctx, args, root=path, train=True,
                                     download=download, transform=transform_train, client_id=client_id)
            # testset = tv.datasets.CIFAR10(root=path, train=False, download=download, transform=transform_test)
            testset = FLCifar10(exec_ctx, args, root=path, train=False,
                                download=download, transform=transform_test, client_id=client_id)

        elif dataset == "cifar10_fl_by_class":
            if load_trainset:
                trainset = FLCifar10ByClass(exec_ctx, args, root=path, train=True,
                                            download=download, transform=transform_train, client_id=client_id)
            # testset = tv.datasets.CIFAR10(root=path, train=False, download=download, transform=transform_test)
            testset = FLCifar10ByClass(exec_ctx, args, root=path, train=False,
                                       download=download, transform=transform_test, client_id=client_id)

        elif dataset == "cifar100":
            if load_trainset:
                trainset = torchvision.datasets.CIFAR100(root=path, train=True,
                                                         download=download, transform=transform_train)
                trainset.num_clients = get_num_clients(dataset)

            testset = torchvision.datasets.CIFAR100(root=path, train=False,
                                                    download=download, transform=transform_test)
        elif dataset == "cifar100_fl":
            if load_trainset:
                trainset = FLCifar100(path, train=True, transform=transform_train, client_id=client_id)
                trainset.num_clients = get_num_clients(dataset)
            testset = FLCifar100(path, train=False, transform=transform_test)
        else:
            raise NotImplementedError(f'{dataset} is not implemented.')

    elif dataset in ["femnist", 'emnist']:
        if load_trainset:
            trainset = FEMNIST(path, train=True, client_id=client_id)

        testset = FEMNIST(path, train=False)

    elif dataset in ['shakespeare', 'full_shakespeare']:
        if load_trainset:
            trainset = Shakespeare(path, train=True, client_id=client_id)
        testset = Shakespeare(path, train=False)

    elif dataset in ['generated_for_quadratic_minimization']:
        trainset = ArificialDataset(exec_ctx, args, train=True)
        testset = ArificialDataset(exec_ctx, args, train=False)

        trainset.compute_Li_for_linear_regression()
        testset.compute_Li_for_linear_regression()

    elif dataset in LibSVMDataset.allowableDatasets():
        transform_train = None  # transforms.Compose([transforms.ToTensor()])
        transform_test = None  # transforms.Compose([transforms.ToTensor()])

        trainset = LibSVMDataset(exec_ctx, args,
                                 root=path, dataset=dataset, train=True, download=download,
                                 transform=transform_train, target_transform=None, client_id=client_id,
                                 num_clients=get_num_clients(dataset))

        testset = LibSVMDataset(exec_ctx, args,
                                root=path, dataset=dataset, train=False, download=download,
                                transform=transform_test, target_transform=None, client_id=client_id,
                                num_clients=get_num_clients(dataset))

        trainset.compute_Li_for_logregression()
        testset.compute_Li_for_logregression()

    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    return trainset, testset


def get_test_batch_size(dataset, batch_size):
    dataset = dataset.lower()
    if dataset == 'shakespeare':
        return SHAKESPEARE_EVAL_BATCH_SIZE
    return batch_size


def evalute_num_classes(dataset):
    """
   Helper function for evaluate number of classes for classification via traversing all dataset samples

   Args:
       dataset: dataset object that supports __len __ and __getitem__ routines. __getitem__ should return (in., target)

   Returns:
       Number of classes in dataset
   """
    max_class = 0
    min_class = 0

    samples = len(dataset)
    for sample_idx in range(samples):
        input_sample, target = dataset[sample_idx]
        max_class = max(target, max_class)
        min_class = min(target, min_class)

    number_of_classes_in_dataset = max_class - min_class + 1
    return number_of_classes_in_dataset


def get_num_classes(dataset):
    """ Helper function for get number of classes in a well-known datasets

   Args:
       dataset(str): name of dataset

   Returns:
       Number of classes in dataset
   """
    dataset = dataset.lower()
    if dataset in ['cifar10', 'cifar10_fl', 'cifar10_fl_by_class']:
        num_classes = 10
    elif dataset in ['cifar100', 'cifar100_fl']:
        num_classes = 100
    elif dataset in ['femnist', 'emnist']:
        num_classes = 62
    elif dataset == 'fashion-mnist':
        num_classes = 10
    elif dataset in ['shakespeare', 'full_shakespeare']:
        num_classes = 90
    elif dataset in ['w9a', 'w8a', 'w7a', 'w6a', 'w5a', 'w4a', 'w3a', 'w2a', 'w1a']:
        num_classes = 2
    elif dataset in ['a9a', 'a8a', 'a7a', 'a6a', 'a5a', 'a4a', 'a3a', 'a2a', 'a1a']:
        num_classes = 2
    elif dataset in ['mushrooms', 'phishing']:
        num_classes = 2
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    return num_classes


def get_num_clients(dataset):
    """
   Get number of clients for specific dataset.

   Args:
       dataset(str): name of dataset

   Returns:
       Number of clients presented in dataset
   """
    dataset = dataset.lower()
    if dataset in ['emnist', 'cifar10', 'cifar100', 'full_shakespeare']:
        num_clients = 1
    elif dataset == 'shakespeare':
        num_clients = 715
    elif dataset == 'femnist':
        num_clients = 3400
    elif dataset == 'cifar100_fl':
        num_clients = 500
    elif dataset == 'cifar10_fl':
        num_clients = 10
    elif dataset == 'cifar10_fl_by_class':
        num_clients = 10
    elif dataset == 'w9a' or dataset == 'w8a' or dataset == 'w7a' or dataset == 'w6a' or dataset == 'w5a' or \
         dataset == 'w4a' or dataset == 'w3a' or dataset == 'w2a' or dataset == 'w1a':
        # num_clients = 100
        num_clients = 10
    elif dataset == 'a9a' or dataset == 'a8a' or dataset == 'a7a' or dataset == 'a6a' or dataset == 'a5a' or \
         dataset == 'a4a' or dataset == 'a3a' or dataset == 'a2a' or dataset == 'a1a':
        # num_clients = 100
        num_clients = 10
    elif dataset == 'mushrooms' or dataset == 'phishing':
        num_clients = 10
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

    return num_clients
