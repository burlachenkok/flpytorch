#!/usr/bin/env python3

import numpy as np

# Import PyTorch root package import torch
import torch

from copy import deepcopy
from .logger import Logger


def get_sampled_clients(num_clients, args, exec_ctx):
    # clients are pre-sampled for deterministic participation among runs
    if args.client_sampling_type == "uniform":
        sampled_clients = [exec_ctx.np_random.choice(num_clients, args.num_clients_per_round, replace=False) for _ in range(args.rounds)]
        return sampled_clients
    elif args.client_sampling_type == "poisson":
        # Poisson sampling, which allows empty sample in general
        sampled_clients = []
        for i in range(args.rounds):
            collect_clients = []
            for j in range(num_clients):
                rv = exec_ctx.np_random.uniform()
                if rv < args.client_sampling_poisson:
                    collect_clients.append(j)
            sampled_clients.append(np.asarray(collect_clients))
        return sampled_clients
    elif args.client_sampling_type == "poisson-no-empty":
        # Poisson sampling, but excluding situation with empty sample
        while True:
            sampled_clients = []
            for i in range(args.rounds):
                collect_clients = []
                while len(collect_clients) == 0:
                    for j in range(num_clients):
                        rv = exec_ctx.np_random.uniform()
                        if rv < args.client_sampling_poisson:
                            collect_clients.append(j)
                sampled_clients.append(np.asarray(collect_clients))
            return sampled_clients
    else:
        assert(not "Unknown sampling type!")
        return None


def update_train_dicts(state_dicts, weights):
    logger = Logger.get("default")

    # get dictionary structure
    model_dict = deepcopy(state_dicts[0]['model'])
    optimiser_dict = deepcopy(state_dicts[0]['optimiser'])

    # model state_dict (structure layer key: value)
    logger.info('Aggregating model state dict.')
    for layer in model_dict:
        layer_vals = torch.stack([state_dict['model'][layer] for state_dict in state_dicts])
        model_dict[layer] = weighted_sum(layer_vals, weights)

    # optimiser state dict (structure: layer key (numeric): buffers for layer: value)
    if 'state' in optimiser_dict:
        logger.info('Aggregating optimiser state dict.')
        for l_key in optimiser_dict['state']:
            layer = optimiser_dict['state'][l_key]
            for buffer in layer:
                buffer_vals = torch.stack([state_dict['optimiser']['state'][l_key][buffer]
                                           for state_dict in state_dicts])
                optimiser_dict['state'][l_key][buffer] = weighted_sum(buffer_vals, weights)
    return model_dict, optimiser_dict


def update_train_dicts_param_based(state_dicts, weights, clients):
    logger = Logger.get("default")

    # get dictionary structure
    state_dicts.waitForItem()
    first_state = state_dicts.popFront()

    optimiser_dict = deepcopy(first_state['optimiser'])
    model_dict = first_state['model'] * weights[0]

    # model state_dict (structure layer key: value)
    logger.info('Aggregating model state dict.')

    for i in range(1, clients):
        state_dicts.waitForItem()
        client_model = state_dicts.popFront()
        model_dict += client_model['model'] * weights[i]

        # optimiser state dict (structure: layer key (numeric): buffers for layer: value)
        if 'state' in optimiser_dict:
            # logger.info('Aggregating optimiser state dict.')
            for l_key in optimiser_dict['state']:
                layer = optimiser_dict['state'][l_key]
                for buffer in layer:
                    buffer_vals = client_model['optimiser']['state'][l_key][buffer]
                    optimiser_dict['state'][l_key][buffer] += buffer_vals * weights[i]
    return model_dict, optimiser_dict


def weighted_sum(tensors, weights):
    # Step-1: create view of tensor with extra artificial axis 1,1,1,1
    # Step-2: exploit broadcasting feature to form one tensor first axis # weights, other axis correspond to each tensor
    # Step-3: perform summation of all tensors with this trick

    extra_dims = (1,)*(tensors.dim()-1)
    return torch.sum(weights.view(-1, *extra_dims) * tensors, dim=0)
