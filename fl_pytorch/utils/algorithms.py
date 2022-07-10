#!/usr/bin/env python3

import random
import time
import copy
import math

# Import PyTorch root package import torch                        
import torch

import numpy as np

from utils import execution_context
from utils import  model_funcs
from utils import  compressors
from models import mutils

import utils
import argparse
from utils.logger import Logger

# ======================================================================================================================
def evaluateGradient(client_state, model, dataloader, criterion, is_rnn, update_statistics = True, evaluate_function = False, device = None, args = None):
    """
    Evalute gradient for model at current point and optionally update statistics and return loss value at current point.

    Args:
        client_state(dict): information about client. used information - statistics, used device
        model(torch.nn.Module): used model for which trainable variables we will evaluate full gradient
        dataloader: used dataloader for fetch records for evaluate local loss during training
        criterion: used criteria with setuped reduction as sum. After evalute reduction correct scaling is a part of evaluation
        is_rnn(bool): flag which specofy that what we evaluate is rnn
        update_statistics(bool): update the following statistics - full_gradient_oracles, samples_gradient_oracles, dataload_duration, inference_duration, backprop_duration
        evaluate_function(bool): if true then returned value is CPU scalar which describes loss function value

    Returns:
        If evaluate_function is True then sclar with local ERM value
    """
    model.train(True)
    if model.do_not_use_bn_and_dropout:
        mutils.turn_off_batch_normalization_and_dropout(model)

    if update_statistics:
        client_state['stats']['full_gradient_oracles'] += 1
        client_state['stats']['samples_gradient_oracles'] += len(dataloader.dataset)

    # Zero out previous gradient
    for p in model.parameters():
        p.grad = None

    if device is None:
        device = client_state["device"]

    total_number_of_samples = len(dataloader.dataset)
    function_value = None

    if evaluate_function:
        function_value = torch.Tensor([0.0]).to(device = device, dtype = model.fl_dtype)

    for i, (data, label) in enumerate(dataloader):
        start_ts = time.time()
        batch_size = data.shape[0]
        if str(data.device) != device or str(label.device) != device or data.dtype != model.fl_dtype or label.dtype != model.fl_dtype:
            data, label = data.to(device = device, dtype = model.fl_dtype), label.to(device = device, dtype = model.fl_dtype)

        if update_statistics:
            client_state["stats"]["dataload_duration"] += (time.time() - start_ts)

        input, label = model_funcs.get_train_inputs(data, label, model, batch_size, device, is_rnn)

        start_ts = time.time()
        outputs = model(*input)

        if not is_rnn:
            hidden = None
            output = outputs
        else:
            output, hidden = outputs

        loss = model_funcs.compute_loss(model, criterion, output, label)
        loss = loss * (1.0/total_number_of_samples)

        if evaluate_function:
            function_value += loss

        if update_statistics:
            client_state["stats"]["inference_duration"] += (time.time() - start_ts)

        start_ts = time.time()
        loss.backward()

        if update_statistics:
            client_state["stats"]["backprop_duration"] += (time.time() - start_ts)

    if args is None:
        args = client_state["H"]["args"]

    regulizer_global = model_funcs.get_global_regulizer(args.global_regulizer)
    R = regulizer_global(model, args.global_regulizer_alpha)

    Rvalue = 0.0
    if R is not None:
        R.backward()
        Rvalue = R.item()

    if evaluate_function:
        return function_value.item() + Rvalue
    else:
        return None
# ======================================================================================================================
def evaluateSgd(client_state, model, dataloader, criterion, is_rnn, update_statistics = True, evaluate_function = False, device = None, args = None):
    """
    Evalute gradient estimator with using global context for model at current point and optionally update statistics and return loss value at current point.

    Args:
        client_state(dict): information about client. used information - statistics, used device
        model(torch.nn.Module): used model for which trainable variables we will evaluate full gradient
        dataloader: used dataloader for fetch records for evaluate local loss during training
        criterion: used criteria with setuped reduction as sum. After evalute reduction correct scaling is a part of evaluation
        is_rnn(bool): flag which specofy that what we evaluate is rnn
        update_statistics(bool): update the following statistics - full_gradient_oracles, samples_gradient_oracles, dataload_duration, inference_duration, backprop_duration
        evaluate_function(bool): if true then returned value is CPU scalar which describes loss function value

    Returns:
        If evaluate_function is True then sclar with local ERM value
    """
    exec_ctx = client_state['H']["execution_context"]

    if "internal_sgd" not in exec_ctx.experimental_options:
        return evaluateGradient(client_state, model, dataloader, criterion, is_rnn, update_statistics=update_statistics, evaluate_function=evaluate_function, device=device)

    internal_sgd = exec_ctx.experimental_options['internal_sgd']
    if internal_sgd == 'full-gradient':
        return evaluateGradient(client_state, model, dataloader, criterion, is_rnn, update_statistics=update_statistics, evaluate_function=evaluate_function, device=device)

    model.train(True)
    if model.do_not_use_bn_and_dropout:
        mutils.turn_off_batch_normalization_and_dropout(model)

    # Zero out previous gradient
    for p in model.parameters():
        p.grad = None

    if device is None:
        device = client_state["device"]

    total_number_of_samples = len(dataloader.dataset)
    function_value = None

    if evaluate_function:
        function_value = torch.Tensor([0.0]).to(device = device, dtype = model.fl_dtype)

    # ==================================================================================================================
    indicies = None
    if internal_sgd == "sgd-nice" or internal_sgd == 'sgd-us' or internal_sgd == 'iterated-minibatch' or internal_sgd == 'sgd-multi':
        indicies = client_state['iterated-minibatch-indicies']
        indicies = torch.from_numpy(indicies)
    # ==================================================================================================================
    batch_size_ds = dataloader.batch_size
    iterations = math.ceil(len(indicies) / float(batch_size_ds))

    sampled_samples = len(indicies)

    for i in range(iterations):
        data = []
        label = []
        for j in range(batch_size_ds):
            index = i * batch_size_ds + j
            if index >= sampled_samples:
                break

            d, t = dataloader.dataset[indicies[index]]

            data.append(d.unsqueeze(0))

            if not torch.is_tensor(t):
                if type(criterion) is torch.nn.MSELoss:
                    label.append(torch.Tensor([t]))
                else:
                    label.append(torch.LongTensor([t]))
            else:
                label.append(t.unsqueeze(0))

        data = torch.cat(data)
        label = torch.cat(label)

        start_ts = time.time()
        batch_size = data.shape[0]
        if str(data.device) != device or str(label.device) != device or data.dtype != model.fl_dtype or label.dtype != model.fl_dtype:
            data, label = data.to(device = device, dtype = model.fl_dtype), label.to(device = device, dtype = model.fl_dtype)

        if update_statistics:
            client_state["stats"]["dataload_duration"] += (time.time() - start_ts)

        if update_statistics:
            client_state['stats']['full_gradient_oracles'] += float(batch_size) / total_number_of_samples
            client_state['stats']['samples_gradient_oracles'] += batch_size

        input, label = model_funcs.get_train_inputs(data, label, model, batch_size, device, is_rnn)

        start_ts = time.time()
        outputs = model(*input)

        if not is_rnn:
            hidden = None
            output = outputs
        else:
            output, hidden = outputs

        loss = model_funcs.compute_loss(model, criterion, output, label)
        loss = loss * (1.0/sampled_samples)

        if evaluate_function:
            function_value += loss

        if update_statistics:
            client_state["stats"]["inference_duration"] += (time.time() - start_ts)

        start_ts = time.time()
        loss.backward()

        if update_statistics:
            client_state["stats"]["backprop_duration"] += (time.time() - start_ts)

    if args is None:
        args = client_state["H"]["args"]

    regulizer_global = model_funcs.get_global_regulizer(args.global_regulizer)
    R = regulizer_global(model, args.global_regulizer_alpha)

    Rvalue = 0.0
    if R is not None:
        R.backward()
        Rvalue = R.item()

    if evaluate_function:
        return function_value.item() + Rvalue
    else:
        return None
# ======================================================================================================================
def evaluateFunction(client_state, model, dataloader, criterion, is_rnn, update_statistics = True, device = None, args = None):
    """
    Evalute gradient for model at current point and optionally update statistics and return loss value at current point.

    Args:
        client_state(dict): information about client. used information - statistics, used device
        model(torch.nn.Module): used model for which trainable variables we will evaluate full gradient
        dataloader: used dataloader for fetch records for evaluate local loss during training
        criterion: used criteria with setuped reduction as sum. After evalute reduction correct scaling is a part of evaluation
        is_rnn(bool): flag which specofy that what we evaluate is rnn
        update_statistics(bool): update the following statistics - dataload_duration, inference_duration

    Returns:
        Scalar with local ERM value
    """
    model.train(False)

    if device is None:
        device = client_state["device"]

    total_number_of_samples = len(dataloader.dataset)
    total_loss = torch.Tensor([0.0]).to(device = device, dtype = model.fl_dtype)

    # code wrap that stops autograd from tracking tensor 
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            start_ts = time.time()
            batch_size = data.shape[0]
            if str(data.device) != device or str(label.device) != device or data.dtype != model.fl_dtype or label.dtype != model.fl_dtype:
                data, label = data.to(device=device, dtype=model.fl_dtype), label.to(device=device,dtype=model.fl_dtype)

            if update_statistics:
                client_state["stats"]["dataload_duration"] += (time.time() - start_ts)

            input, label = model_funcs.get_train_inputs(data, label, model, batch_size, device, is_rnn)

            start_ts = time.time()
            outputs = model(*input)

            if not is_rnn:
                hidden = None
                output = outputs
            else:
                output, hidden = outputs

            loss = model_funcs.compute_loss(model, criterion, output, label)
            loss = loss * (1.0/total_number_of_samples)

            if update_statistics:
                client_state["stats"]["inference_duration"] += (time.time() - start_ts)
            total_loss += loss

    if args is None:
        args = client_state["H"]["args"]

    regulizer_global = model_funcs.get_global_regulizer(args.global_regulizer)
    R = regulizer_global(model, args.global_regulizer_alpha)

    Rvalue = 0.0
    if R is not None:
        Rvalue = R.item()

    return total_loss.item() + Rvalue

def findRecentRecord(H, client_id, field):
    """
    Find in history records recent information about query record in client_states.

    Args:
        H(dict): information about client. used information - statistics, used device
        client_id(int): integer number for client
        field(str): name of the field which we are trying to find history

    Returns:
        Return requested object if it found or None if object is not found
    """
    history = H['history']

    history_keys = [k for k in history.keys()]
    history_keys.sort(reverse=True)

    for r in history_keys:
        clients_history = history[r]
        if client_id in clients_history['client_states']:
            client_prev_state = clients_history['client_states'][client_id]['client_state']
            if field in client_prev_state:
                return_object = client_prev_state[field]
                return return_object

            else:
                # Assumption -- if client has been sampled then field have to be setuped
                return None
    return None

def findRecentRecordAndRemoveFromHistory(H, client_id, field):
    """
    Find in history records recent information about query record in client_states.
    If record has been found return it, but before that remove itself from history.

    Args:
        H(dict): information about client. used information - statistics, used device
        client_id(int): integer number for client
        field(str): name of the field which we are trying to find history

    Returns:
        Return requested object if it found or None if object is not found
    """
    history = H['history']
    history_keys = [k for k in history.keys()]
    history_keys.sort(reverse=True)

    for r in history_keys:
        clients_history = history[r]
        if client_id in clients_history['client_states']:
            client_prev_state = clients_history['client_states'][client_id]['client_state']
            if field in client_prev_state:
                return_object = client_prev_state[field]
                client_prev_state[field] = None
                return return_object
            else:
                # Assumption -- if client has been sampled then field have to be setuped
                return None
    return None

# ======================================================================================================================
def get_logger(H):
    """
    Help function to get logger.

    Args:
        H(dict): server state

    Returns:
        Reference to logger
    """

    my_logger = Logger.get(H["args"].run_id)
    return my_logger

def has_experiment_option(H, name):
    """
    Check that experimental option is presented

    Args:
        H(dict): server state
        name(str): variable name

    Returns:
        True if option is present
    """
    return name in H["execution_context"].experimental_options

def get_experiment_option_f(H, name):
    """
    Get experimental option to carry experiments with algorithms

    Args:
        H(dict): server state
        name(str): variable name

    Returns:
        Value of requested value converted to float
    """
    return float(H["execution_context"].experimental_options[name])

def get_experiment_option_int(H, name):
    """
    Get experimental option to carry experiments with algorithms

    Args:
        H(dict): server state
        name(str): variable name

    Returns:
        Value of requested value converted to int
    """
    return int(H["execution_context"].experimental_options[name])

def get_experiment_option_str(H, name):
    """
    Get experimental option to carry experiments with algorithms

    Args:
        H(dict): server state
        name(str): variable name

    Returns:
        Value of requested value converted to string
    """
    return str(H["execution_context"].experimental_options[name])

def get_initial_shift(args:argparse.Namespace, D:int, grad_start:torch.Tensor):
    """Help method to get initial shifts"""
    if args.initialize_shifts_policy == "full_gradient_at_start":
        return grad_start.detach().clone().to(device = args.device)
    else:
        return torch.zeros(D).to(device = args.device, dtype = grad_start.dtype)
# ======================================================================================================================
class MarinaAlgorithm:
    '''
    MARINA Algoritm [Gorbunov et al., 2021]: https://arxiv.org/abs/2102.07845
    '''
    @staticmethod
    def initializeServerState(args:argparse.Namespace, model:torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor)->dict:
        state = {"x_prev" : mutils.get_params(model),             # previous iterate
                 "test_ber_rv" : 0.0                              # test_ber_rv = 0.0 will force fisrt iteration be a full gradient evaluation
                 }
        return state

    @staticmethod
    def clientState(H:dict, clientId:int, client_data_samples:int, device:str)->dict:
        logger = Logger.get(H["run_id"])

        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)

        p = 1.0 / (1.0 + compressor.getW())

        state = {}
        if H["test_ber_rv"] <= p:
            state.update({"p" : p, "ck": 1, "client_compressor" : compressor})
        else:
            state.update({"p" : p, "ck": 0, "client_compressor": compressor})

        return state

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        if client_state["ck"] == 1:
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)
            client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
            return fApprox, grad_cur
        else:
            client_id = client_state["client_id"]
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            reconstruct_params = mutils.get_params(model)
            mutils.set_params(model, client_state["H"]["x_prev"])
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
            grad_prev = mutils.get_gradient(model)
            mutils.set_params(model, reconstruct_params)

            g_prev = client_state["H"]["g_prev"].to(device = client_state["device"], dtype = model.fl_dtype)
            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - grad_prev)

            # Comments: server knows g_prev
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            return fApprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer, clients: int, model: torch.nn.Module, params_current: torch.Tensor, H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H["g_prev"] = grad_server
        H["x_prev"] = mutils.get_params(model)

        H["test_ber_rv"] = H['execution_context'].np_random.random()
        return H
# ======================================================================================================================
def getLismoothForClients(H, clients_responses):
    Li = np.array(H['Li_all_clients'])

    if H["args"].global_regulizer == "none":
        pass
    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
        Li = Li + (1.0 * H["args"].global_regulizer_alpha)
    elif H["args"].global_regulizer == "noncvx_robust_linear_regression":
        Li = Li + (2.0 * H["args"].global_regulizer_alpha)

    return Li

def getLsmoothGlobal(H, clients_responses):
    L = H['L_compute']

    if H["args"].global_regulizer == "none":
        pass
    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
        L = L + (1.0 * H["args"].global_regulizer_alpha)
    elif H["args"].global_regulizer == "noncvx_robust_linear_regression":
        L = L + (2.0 * H["args"].global_regulizer_alpha)

    return L

# ======================================================================================================================
class MarinaAlgorithmPP:
    '''
    MARINA Algoritm [Gorbunov et al., 2021]: https://arxiv.org/abs/2102.07845
    '''
    @staticmethod
    def algorithmDescription():
        return { "paper" : "https://arxiv.org/abs/2102.07845" }

    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        # Step size for non-convex case
        m = 1.0
        workers_per_round = clients_in_round
        workers = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)

        # Ltask = (np.mean( (Li_all_clients) **2) )**0.5
        Ltask = (np.mean( max(Li_all_clients) **2) )**0.5 # Maybe hack by /2
        w = H["w"]
        p = (workers_per_round/workers)*1.0/(1+w)            # For RAND-K compressor
        r = workers_per_round

        step_1 = ((1 + 4*(1-p)*(1+w)/(p*workers))**0.5 - 1) / (2* (1-p) * (1+w)/(p*workers) * Ltask)
        step_2 = (-(1 + 4*(1-p)*(1+w)/(p*workers))**0.5 - 1) / (2* (1-p) * (1+w)/(p*workers) * Ltask)
        step_3 = 1.0 / (Ltask * (1 + ((1 - p) * (1 + w) / (p * workers_per_round)) ** 0.5))  # Theorem 4.1, p.37

        return step_3

    @staticmethod
    def initializeServerState(args:argparse.Namespace, model:torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor)->dict:
        compressor = compressors.initCompressor(args.client_compressor, D)

        state = {"x_prev" : mutils.get_params(model),             # previous iterate
                 "test_ber_rv" : 0.0,                             # test_ber_rv = 0.0 will force fisrt iteration be a full gradient evaluation
                 "num_clients_per_round" : args.num_clients_per_round,
                 "total_clients" : total_clients,
                 "w" : compressor.getW()
                 }

        p = 1.0 / (1.0 + compressor.getW())
        p = p * args.num_clients_per_round / total_clients
        state.update({"p" : p})

        if state["test_ber_rv"] <= p:
            state["ck"] = 1
            state["request_use_full_list_of_clients"] = True
        else:
            state["ck"] = 0
            state["request_use_full_list_of_clients"] = False

        return state

    @staticmethod
    def clientState(H:dict, clientId:int, client_data_samples:int, device:str)->dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        state = {"p" : H["p"], "ck": H["ck"], "client_compressor" : compressor}
        return state

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        if client_state["ck"] == 1:
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)
            client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
            return fApprox, grad_cur
        else:
            client_id = client_state["client_id"]
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            reconstruct_params = mutils.get_params(model)
            mutils.set_params(model, client_state["H"]["x_prev"])
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
            grad_prev = mutils.get_gradient(model)
            mutils.set_params(model, reconstruct_params)

            g_prev = client_state["H"]["g_prev"].to(device = client_state["device"], dtype = model.fl_dtype)
            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - grad_prev)

            # Comments: server knows g_prev
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            return fApprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer, clients: int, model: torch.nn.Module, params_current: torch.Tensor, H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H["g_prev"] = grad_server
        H["x_prev"] = mutils.get_params(model)

        H["test_ber_rv"] = H['execution_context'].np_random.random()
        if H["test_ber_rv"] <= H["p"]:
            H["ck"] = 1
            H["request_use_full_list_of_clients"] = True
        else:
            H["ck"] = 0
            H["request_use_full_list_of_clients"] = False

        return H
# ======================================================================================================================
class SCAFFOLD:
    '''
    SCAFFOLD Algoritm [Karimireddy et al., 2020]: https://arxiv.org/abs/1910.06378
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {"c"  : torch.zeros(D).to(device = args.device, dtype = model.fl_dtype),
                 "c0" : torch.zeros(D).to(device = args.device, dtype = model.fl_dtype)}

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        # Compressors are not part of SCAFFOLD
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_ci = findRecentRecordAndRemoveFromHistory(H, clientId, 'ci')

        if last_ci is None:
            return {"ci" : H['c0'].detach().clone().to(device = device, dtype = H["fl_dtype"]),
                    "client_compressor": compressor}
        else:
            return {"ci" : last_ci.to(device = device, dtype = H["fl_dtype"]),                  #last_ci.detach().clone().to(device, dtype = H["fl_dtype"]),
                    "client_compressor": compressor}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        c  = client_state["H"]['c'].to(device = client_state["device"], dtype = model.fl_dtype)
        ci = client_state['ci']

        if local_iteration_number[0] == 0:
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = False)
            c_plus = mutils.get_gradient(model)
            client_state['delta_c'] = client_state["client_compressor"].compressVector(c_plus - c)

            # send delta_c and delta_x for model which has the same dimension
            client_state['stats']['send_scalars_to_master'] += client_state['delta_c'].numel() # send change iterates
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance

        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        dy = grad_cur - ci + c

        return fAprox, dy

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        # x is updates as a part of general logic
        # here we will update c = c + sum(dc) * |S|/N

        obtained_model = clients_responses.get(0)
        dc = obtained_model['client_state']['delta_c'].to(device = paramsPrev.device, dtype = H["fl_dtype"])
        clients_num_in_round = len(clients_responses)

        for i in range(1, clients_num_in_round):
            client_model = clients_responses.get(i)
            dc += client_model['client_state']['delta_c'].to(device = paramsPrev.device, dtype = H["fl_dtype"])

        # Make dc is average of detla_c
        dc = dc / clients_num_in_round

        # Construct final delta step for update "c"
        dc = dc * float(clients_num_in_round)/float(H["total_clients"])
        H["c"] += dc
        return H
# ======================================================================================================================
class FRECON:
    '''
    FRECON Algoritm [Haoyu Zhao et al., 2021]: https://arxiv.org/abs/2112.13097
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        # FRECON in non-convex case
        S = clients_in_round
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        w = compressor.getW()
        a = 1 / (1.0 + w)
        n = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)
        Lmax = max(Li_all_clients)
        step_size = 1.0/(Lmax * (1 + (10*(1+w)*(1+w)*n/S/S)**0.5))
        return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start) -> dict:

        state = {"h0": get_initial_shift(args, D, grad_start),
                 "g0": grad_start.detach().clone().to(device = args.device, dtype = model.fl_dtype),
                 "h_prev": get_initial_shift(args, D, grad_start),
                 "g_server_prev": grad_start.detach().clone().to(device = args.device, dtype = model.fl_dtype),
                 "x_prev": mutils.get_params(model)
                 }

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_hi = findRecentRecordAndRemoveFromHistory(H, clientId, 'hi')

        #last_qi = findRecentRecordAndRemoveFromHistory(H, clientId, 'qi')
        # Drop qi
        #last_qi = None

        w = compressor.getW()
        alpha = 1 / (1.0 + w)

        if last_hi is None:
            return {"client_compressor" : compressor, "alpha" : alpha, "hi" : H['h0'].detach().clone().to(device = device, dtype = H["fl_dtype"])}
        else:
            return {"client_compressor" : compressor, "alpha" : alpha,
                    "hi" : last_hi.to(device = device, dtype = H["fl_dtype"])  #last_hi.detach().clone().to(device = device, dtype = H["fl_dtype"])
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:

        client_id = client_state["client_id"]
        fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)

        # Please select in GUI SGD-US or another estimator
        grad_cur = mutils.get_gradient(model)
        ui = client_state["client_compressor"].compressVector(grad_cur - client_state['hi'])
        #if client_state["H"]["current_round"] != 0:
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
        client_state['hi'] = client_state['hi'] + client_state['alpha'] * ui

        # ==============================================================================================================
        reconstruct_params = mutils.get_params(model)
        mutils.set_params(model, client_state["H"]["x_prev"])
        evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
        grad_prev = mutils.get_gradient(model)
        mutils.set_params(model, reconstruct_params)
        # ==============================================================================================================
        qi = client_state["client_compressor"].compressVector(grad_cur - grad_prev)
        #if client_state["H"]["current_round"] != 0:
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
        client_state['qi'] = qi
        # =============================================================================================================
        return fApprox, ui

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        alpha = obtained_model["client_state"]['alpha']

        gs = wi * gi
        q_avg = wi * obtained_model['client_state']['qi']
        w_total = wi

        del obtained_model['client_state']['qi']

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']
            q_avg += wi * client_model['client_state']['qi']

            w_total += wi
            gs += wi * gi

            del client_model['client_state']['qi']

        gs = gs / w_total
        q_avg = q_avg / w_total
        u = gs

        h_prev = H['h_prev']

        # ===============================================================================================================
        if has_experiment_option(H, "lambda_"):
            lambda_ = get_experiment_option_f(H, "lambda_")
        elif has_experiment_option(H, "th_stepsize_noncvx") or has_experiment_option(H, "th_stepsize_cvx"):
            S = clients
            compressor = compressors.initCompressor(H["client_compressor"], H["D"])
            w = compressor.getW()
            n = H['total_clients']
            H["lambda_th"] = S / (2 * (1 + w) * n)
            lambda_ = S/( 2*(1+w) * n )
            get_logger(H).info(f"Used lambda is {lambda_}")
        # ==============================================================================================================
        result = q_avg + (1.0 - lambda_) * H["g_server_prev"] + lambda_ * (u + h_prev)

        multipler_alpha = alpha * (clients/H['total_clients'])
        H['u_avg_update'] =  u
        H['alpha_update'] = multipler_alpha
        return result

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H['h_prev'] = H['h_prev'] + H["alpha_update"] * H['u_avg_update']
        H["x_prev"] = paramsPrev
        H["g_server_prev"] = grad_server
        return H
# ======================================================================================================================
class COFIG:
    '''
    COFIG Algoritm [Haoyu Zhao et al., 2021]: https://arxiv.org/abs/2112.13097
    Assumption: \widetilda{S} = S, i.e. they are the same sets
    '''

    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        # Step size for non-convex case
        S = clients_in_round
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])

        w = compressor.getW()
        a = 1 / (1.0 + w)

        if use_steps_size_for_non_convex_case:
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)
            step_size_1 = 1.0/( Lmax * 2 )
            step_size_2 = S / ( 5 * Lmax * (1 + w) * (H['total_clients'] ** (2.0/3.0)) )
            step_size_3 = S / ( 5 * Lmax * ((1 + w)**3.0/2.0) * (H['total_clients'] ** (0.5)) )

            return min(step_size_1, step_size_2, step_size_3)

        else:
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)

            step_size_1 = 1.0/( Lmax * (2 + 8 * (1 + w) / S) )
            step_size_2 = S / ( (1 + w) * (H['total_clients'] ** 0.5) )

            return min(step_size_1, step_size_2)

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()

        state = {"compressor_master" : cm,
                 "h0": get_initial_shift(args, D, grad_start),
                 "h_prev": get_initial_shift(args, D, grad_start),
                 }

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_hi = findRecentRecordAndRemoveFromHistory(H, clientId, 'hi')
        alpha = 1.0 / (1.0 + compressor.getW())

        if last_hi is None:
            return {"client_compressor" : compressor, "alpha" : alpha, "hi" : H['h0'].detach().clone().to(device = device, dtype = H["fl_dtype"])}
        else:
            return {"client_compressor" : compressor, "alpha" : alpha,
                    "hi" : last_hi.to(device = device, dtype = H["fl_dtype"]) #last_hi.detach().clone().to(device = device)
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:

        client_id = client_state["client_id"]
        fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)

        # Please select in GUI SGD-US or another estimator
        grad_cur = mutils.get_gradient(model)

        ui = client_state["client_compressor"].compressVector(grad_cur - client_state['hi'])
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance

        # Update hi (Experiment!)
        client_state['hi'] = client_state['hi'] + client_state['alpha'] * ui
        return fApprox, ui

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])

        alpha = obtained_model["client_state"]['alpha']

        gs = wi * gi

        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi

        gs = gs / w_total
        u = gs

        h_prev = H['h_prev']
        result =  u + h_prev

        multipler_alpha = alpha * (clients/H['total_clients'])
        H['u_avg_update'] =  gs
        H['alpha_update'] = multipler_alpha
        return result

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H['h_prev'] = H['h_prev'] + H["alpha_update"] * H['u_avg_update']
        return H
# ======================================================================================================================
class DIANA:
    '''
    DIANA Algoritm [Mishchenko et al., 2019]: https://arxiv.org/abs/1901.09269, https://arxiv.org/pdf/1904.05115.pdf
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        # DIANA for non-convex case
        if use_steps_size_for_non_convex_case:
            # For non-convex case
            m = 1.0
            workers_per_round = clients_in_round
            workers = H['total_clients']
            Ltask = getLsmoothGlobal(H, clients_responses)
            step_size = 1.0/(10*Ltask*(1 + H["w"]/workers)**0.5 * (m**(2.0/3.0) + H["w"] + 1))  # Th.4 of https://arxiv.org/pdf/1904.05115.pdf
            return step_size
        else:
            # For convex case
            compressor = compressors.initCompressor(H["client_compressor"], H["D"])
            w = compressor.getW()
            a = 1 / (1.0 + w)
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)
            step_size = 1.0 / (Lmax * (1 + 4*w/clients_in_round))                    # SGD-CTRL analysis for strongly-covnex case
            return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        c = compressors.initCompressor(args.client_compressor, D)
        w = c.getW()
        alpha = 1.0 / (1.0 + w)

        state = {"h0"    : get_initial_shift(args, D, grad_start),
                 "h"     : get_initial_shift(args, D, grad_start),
                 "alpha" : alpha,
                 "w"     :w
                }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)

        last_hi = findRecentRecordAndRemoveFromHistory(H, clientId, 'hi')

        if last_hi is None:
            return {"client_compressor" : compressor,
                    "hi" : H['h0'].detach().clone().to(device = device, dtype = H["fl_dtype"])}
        else:
            return {"client_compressor" : compressor,
                    "hi" : last_hi.to(device = device, dtype = H["fl_dtype"])          #last_hi.detach().clone().to(device = device, dtype = H["fl_dtype"])
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # In theory it's possible to perform compute without accessing "h" from master
        h = client_state['hi']
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        m_i = client_state["client_compressor"].compressVector(grad_cur - h)

        # Comments: server needs only obtain m_i
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance

        client_state['hi'] = client_state['hi'] + client_state['H']['alpha'] * m_i
        return fAprox, m_i

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])

        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total


        # Here gs is final gradient estimator without shift
        H['m'] = gs
        h = H['h']
        return h + gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        mk = H['m']
        H['h'] = H['h'] + H['alpha'] * mk
        return H
# ======================================================================================================================
class EF21:
    '''
    EF21 Algoritm: "EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback", https://arxiv.org/abs/2106.05203
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        if compressor.isContractionCompressor():
            a = compressor.getAlphaContraction() # use alpha for contraction compressor
        elif compressor.isUnbiasedCompressor():
            a = 1/(1.0 + compressor.getW())      # use w for scaled unbiased compressor

        Li = getLismoothForClients(H, clients_responses)
        Ltask = getLsmoothGlobal(H, clients_responses)
        Ltilda = np.mean(Li**2)**0.5

        theta = 1 - (1 - a)**0.5
        beta = (1.0 - a) / (1 - (1 - a)**0.5)
        gamma = 1.0 / (Ltask + Ltilda * (beta/theta)**0.5)

        if has_experiment_option(H, 'stepsize_multiplier'):
            gamma = gamma * get_experiment_option_f(H, 'stepsize_multiplier')

        return gamma # Th.1, p.40 from EF21

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()
        state = {"compressor_master" : cm,
                 "x0" : mutils.get_params(model),
                 "request_use_full_list_of_clients": True
                 }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_g_prev = findRecentRecordAndRemoveFromHistory(H, clientId, 'g_prev')

        if last_g_prev is None:
            return {"client_compressor" : compressor,
                    "g_prev" : None
                    }
        else:
            return {"client_compressor" : compressor,
                    "g_prev" : last_g_prev.to(device = device, dtype = H["fl_dtype"]) #last_g_prev.detach().clone().to(device = device)
                    }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # Compute g0 for a first iteration
        g_prev = client_state['g_prev']
        if g_prev is None:
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)
            grad_cur = mutils.get_gradient(model)
            client_state['g_prev'] = grad_cur
            # Not take into account communication at first round
            return fAprox, grad_cur
        else:
            # In theory it's possible to perform compute without accessing "h" from master
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            g_prev = client_state['g_prev']

            compressor_multiplier = 1.0
            if not client_state["client_compressor"].isContractionCompressor():
                compressor_multiplier = 1.0/(1.0 + client_state["client_compressor"].getW())

            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - g_prev) * compressor_multiplier
            # In algorithm really we need only to send compressed difference between new gradient and previous gradient estimator
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            client_state['g_prev'] = g_next
            return fAprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        # We compute it straightford.
        # In the paper the master uses g^t on server side and combine that with avg. of c_i^t

        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        H["request_use_full_list_of_clients"] = False
        return H
# ======================================================================================================================
class EF21PP:
    '''
    EF21PP: "EF21 with Bells & Whistles: Practical Algorithmic Extensions of Modern Error Feedback" https://arxiv.org/abs/2110.03294, PP with Poisson sampling
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        # Theoretical steps size for non-convex case
        p = H['args'].client_sampling_poisson
        assert p > 0.0

        pmax = p
        pmin = p

        rho = 1e-3
        s = 1e-3

        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        if compressor.isContractionCompressor():
            a = compressor.getAlphaContraction() # use alpha for contraction compressor
        elif compressor.isUnbiasedCompressor():
            a = 1/(1.0 + compressor.getW())      # use w for scaled unbiased compressor

        theta = 1 - (1 + s)*(1 - a)
        beta = (1.0 + 1.0/s) * (1 - a)
        thetap = rho * pmin + theta * pmax - rho - (pmax - pmin)
        Li = getLismoothForClients(H, clients_responses)

        B = (beta * p + (1 + 1.0/rho) * (1 - p)) * (np.mean(Li**2) )

        Ltask = getLsmoothGlobal(H, clients_responses)

        return 1.0/(Ltask + (B/thetap)**0.5) # Th.7, p.47 from EF21-PP

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()
        state = {"compressor_master" : cm,
                 "x0" : mutils.get_params(model),
                 "request_use_full_list_of_clients": True
                 }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_g_prev = findRecentRecordAndRemoveFromHistory(H, clientId, 'g_prev')

        if last_g_prev is None:
            return {"client_compressor" : compressor,
                    "g_prev" : None
                    }
        else:
            return {"client_compressor" : compressor,
                    "g_prev" : last_g_prev.to(device = device, dtype = H["fl_dtype"]) #last_g_prev.detach().clone().to(device = device, dtype = H["fl_dtype"])
                    }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # Compute g0 for a first iteration
        g_prev = client_state['g_prev']
        if g_prev is None:
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)
            grad_cur = mutils.get_gradient(model)
            client_state['g_prev'] = grad_cur
            # Not take into account communication at first round
            return fAprox, grad_cur
        else:
            # In theory it's possible to perform compute without accessing "h" from master
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            g_prev = client_state['g_prev']

            compressor_multiplier = 1.0
            if not client_state["client_compressor"].isContractionCompressor():
                compressor_multiplier = 1.0/(1.0 + client_state["client_compressor"].getW())

            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - g_prev) * compressor_multiplier
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            client_state['g_prev'] = g_next
            return fAprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        # We compute it straightford.
        # In the paper the master uses g^t on server side and combine that with avg. of c_i^t

        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        H["request_use_full_list_of_clients"] = False
        return H
# ======================================================================================================================
class DCGD:
    '''
    Distributed Compressed Gradient Descent Algoritm [Alistarh et al., 2017, Khirirat et al., 2018, Horvath et al., 2019]: https://arxiv.org/abs/1610.02132, https://arxiv.org/abs/1806.06573, https://arxiv.org/abs/1905.10988
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case):
        # Step size for convex case
        workers = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)
        L = getLsmoothGlobal(H, clients_responses)
        w = H["w"]
        wM = H["compressor_master"].getW()

        A = L + 2 * (wM + 1) * max(Li_all_clients * w/workers) + L * wM
        step_size = 1.0/A
        return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D: int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()

        c = compressors.initCompressor(args.client_compressor, D)
        state = {"compressor_master" : cm}

        if c.isUnbiasedCompressor():
            state["w"] = c.getW()

        elif c.isContractionCompressor():
            # Not applicable for DCGD
            state["alpha"] = c.getAlphaContraction()

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        return {"client_compressor" : compressor}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        grad_compress = client_state["client_compressor"].compressVector(grad_cur)
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
        return fAprox, grad_compress

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        return H

# ======================================================================================================================
class FedAvg:
    '''
    Algorithm FedAVG [McMahan et al., 2017]: https://arxiv.org/abs/1602.05629 
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {}
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        return {}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        client_state['stats']['send_scalars_to_master'] += grad_cur.numel()

        return fAprox, grad_cur

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        return H
# ======================================================================================================================
class FedProx:
    '''
    Algorithm FedProx  [Li et al., 2018]: https://arxiv.org/abs/1812.06127
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {'wt': mutils.get_params(model)}
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        return {"client_compressor": compressor}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)

        opts = client_state['H']['execution_context'].experimental_options
        mu_prox = 1.0
        if "mu_prox" in opts:
            mu_prox = float(opts['mu_prox'])

        x_cur = mutils.get_params(model)
        wt = client_state['H']['wt'].to(device = client_state["device"], dtype = model.fl_dtype)

        grad_cur += mu_prox * (x_cur - wt)

        client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
        # assume sending 'wt' from master to clients is for free

        grad_cur = client_state["client_compressor"].compressVector(grad_cur)

        return fAprox, grad_cur

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(device = params_current.device, dtype = H["fl_dtype"])
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H['wt'] = mutils.get_params(model)
        return H
# ======================================================================================================================
def getImplClassForAlgo(algorithm):
    """
    Get imlementation class for specific algorithm

    Args:
        algorithm(str): Algorithm implementation

    Returns:
        class type with need interface methods
    """
    classImpl = None
    if algorithm == "marina":
        classImpl = MarinaAlgorithm
    elif algorithm == "dcgd":
        classImpl = DCGD
    elif algorithm == "fedavg":
        classImpl = FedAvg
    elif algorithm == "diana":
        classImpl = DIANA
    elif algorithm == "scaffold":
        classImpl = SCAFFOLD
    elif algorithm == "fedprox":
        classImpl = FedProx
    elif algorithm == "ef21":
        classImpl = EF21
    elif algorithm == "cofig":
        classImpl = COFIG
    elif algorithm == "frecon":
        classImpl = FRECON
    elif algorithm == "ef21-pp":
        classImpl = EF21PP
    elif algorithm == "pp-marina":
        classImpl = MarinaAlgorithmPP
    else:
        raise ValueError(f"Please check algorithm. There is no implementation for '{algorithm}'")

    return classImpl

def getAlgorithmsList():
    """
    Get list of algorithms in order in which sorting happens in GUI
    """
    algoList = ["marina", "dcgd", "cofig", "frecon", "fedavg", "diana", "scaffold",
                "fedprox", "ef21", "ef21-pp", "pp-marina"]

    for a in algoList:
        assert getImplClassForAlgo(a) is not None

    return algoList

def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, total_clients:int, grad_start: torch.Tensor, exec_ctx) -> dict:
    """
    Initialize server state.

    Server state is a main source of information with various information, including:
     - 'x0' : start iterate. Be default it's a current position where model is
     - 'algorithm': string represenation of the used algorithms
     - 'history': history by rounds

    Args:
        args (argparse): Parsed command line for the python shell process
        model (nn.Module): Model under which server operate
        total_clients (int): Total number of clients in the experiment
        grad_start (torch.Tensor): Full gradient at starting point

    Returns:
        Returns initialize server state for specific algorithms args.algorithm
    """
    serverState = {}
    classImpl = getImplClassForAlgo(args.algorithm)

    D = mutils.number_of_params(model)
    serverState = classImpl.initializeServerState(args, model, D, total_clients, grad_start)

    serverState.update( {'algorithm' : args.algorithm,
                         'history' : {},
                         'D' : D,
                         'fl_dtype' : model.fl_dtype,
                         'D_include_frozen' : mutils.number_of_params(model, skipFrozen = False),
                         'current_round' : 0,
                         'client_compressor' : args.client_compressor,
                         'run_id' : args.run_id,
                         'start_time': time.strftime("%d %b %Y %H:%M:%S UTC%z", time.localtime()),
                         'server_state_update_time': time.strftime("%d %b %Y %H:%M:%S UTC%z", time.localtime())
                         } )

    if 'x0' not in serverState:
        serverState.update( {'x0' : mutils.get_params(model) } )

    return serverState

def clientState(H:dict, clientId:int, client_data_samples:int, round:int, device:str)->dict:
    """
    Initialize client state.

    Clientstate is initialized from the scratch at each round for each selected client which is participates in optimization.
    If you want to initilize client via using it's previous state please use findRecentRecord() to find need information in server state.

     clientstate is a second source of information which helps to operate for clients
     - 'H' : referece to server state
     - 'algorithm': string represenation of the used algorithms
     - 'client_id': id of the client
     - 'device': target device for the client
     - 'round': number of round in which this state has been used
     - 'weight': weight used in the aggregation in serverGradient
     - 'stats': different statistics for a single client
     - 'seed': custom seed for pseudo-random generator for a client

    Args:
        H (dict): Server state
        clientId(int): Id of the client
        client_data_samples(int): Number of data points for client
        round(int): Number of the round
        device(str): Target device which should be used by the client for computations and store client state

    Returns:
        Initialized client state
    """
    classImpl = getImplClassForAlgo(H["algorithm"])

    clientState = classImpl.clientState(H, clientId, client_data_samples, device)
    if 'weight' not in clientState:
        clientState.update( {'weight' : 1.0} )


    clientState.update( {'H'         : H,
                         'algorithm' : H["algorithm"],
                         'client_id' : clientId,
                         'device'    : device,
                         'weight'    : 1.0,
                         'round'     : round,
                         'approximate_f_value' : [],
                         'seed' :   H['execution_context'].np_random.randint(2**31)
                         }
                      )

    stats = {"dataload_duration"  : 0.0,
             "inference_duration" : 0.0,
             "backprop_duration"  : 0.0,
             "full_gradient_oracles"    : 0,
             "samples_gradient_oracles" : 0,
             "send_scalars_to_master"   : 0
             }

    clientState.update({"stats" : stats})

    return clientState

def localGradientEvaluation(client_state:dict, model:torch.nn.Module, dataloader:torch.utils.data.dataloader.DataLoader, criterion:torch.nn.modules.loss._Loss, is_rnn:bool, local_iteration_number:tuple)->torch.Tensor:
    """
    Evalute local gradient for client.

    This API should implement optimization schema specific SGD estimator.

    Args:
        client_state (dict): state of the client which evaluate local gradient
        model(nn.Module): Initialized computation graph(model) locating currently in the interesting "x" position
        dataloader(torch.utils.data.dataloader.DataLoader): DataLoader mechanism to fectch data
        criterion(class): Loss function for minimization, defined as a "summ" of loss over train samples.
        is_rnn(bool): boolean flag which say that model is RNN

    Returns:
        Flat 1D SGD vector
    """
    classImpl = getImplClassForAlgo(client_state["algorithm"])
    return classImpl.localGradientEvaluation(client_state, model, dataloader, criterion, is_rnn, local_iteration_number)

def serverGradient(clients_responses : utils.buffer.Buffer,
                   clients : int,
                   model   : torch.nn.Module,
                   params_current : torch.Tensor,
                   H : dict)->torch.Tensor:
    """
    Evalute server gradient via analyzing local shifts from the clients.

    Args:
            clients_responses (Buffer): client responses. Each client transfers at least:
            'model' field in their response with last iterate for local model
            'optimiser' state of the local optimizer for optimizers with state
            'client_id' id of the client
            'client_state' state of the client

        clients(int): number of clients in that communication round. Invariant len(clients_responses) == clients
        model(torch.nn.Module): model which is locating currently in need server position.
        params_current(torch.Tensor): position where currently model is locating
        H(dict): server state

    Returns:
        Flat 1D SGD vector
    """
    if clients == 0:
        return torch.zeros_like(params_current)

    classImpl = getImplClassForAlgo(H["algorithm"])
    gs = classImpl.serverGradient(clients_responses, clients, model, params_current, H)

    # Need change for global optimizer for some time was gs=-gs. Currently no need.

    return gs

def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses: utils.buffer.Buffer, use_steps_size_for_non_convex_case:bool):
    """
    Experimental method to evaluate theoretical step-size for non-convex L-smooth case

    Args:
        x_cur(torch.Tensor): current iterate
        grad_server(torch.Tensor): global gradient for which we're finding theoretical step-size
        H(dict): server state
        clients_in_round(int): number of clients particiapted in that round
        clients_responses: responses from clients

    Returns:
        Step size
    """
    classImpl = getImplClassForAlgo(H["algorithm"])
    step_size = classImpl.theoreticalStepSize(x_cur, grad_server, H, clients_in_round, clients_responses, use_steps_size_for_non_convex_case)

    logger = Logger.get(H["args"].run_id)
    logger.info(f'Computed step size for H["algorithm"] is {step_size}')

    return step_size

def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, model:torch.nn.Module, paramsPrev:torch.Tensor, round:int, grad_server:torch.Tensor, H:dict, numClients:int, sampled_clients_per_next_round)->dict:
    """
    Server global state update.

    Default update - include any states from previous round, but excluding model parameters, which maybe huge.

    Args:
        clients_responses (Buffer): client responses. Each client transfers at least:
        model(torch.nn.Module): model for sever, initialized with last position.
        paramsPrev(torch.Tensor): previous model parameters at the begininng of round
        round(int): number of the round
        grad_server(torch.Tensor): server's gradient.
        H(dict): server state
        numClients(int): number of clients in that round
        sampled_clients_per_next_round: future clients in a next round

    Returns:
        New server state.
    """
    clients = {}

    while len(clients_responses) < numClients:
        pass

    fvalues = []

    # Prune after communication round
    for item in clients_responses.items:
        assert item is not None
        # if item is None:
        #    continue

        item['client_state'].update({"optimiser" : item['optimiser']})
        item['client_state'].update({"buffers" : item['buffers']})

        del item['optimiser']
        del item['buffers']

        del item['model']

        if 'H' in item['client_state']:
            del item['client_state']['H']

        if 'client_compressor' in item['client_state']:
            del item['client_state']['client_compressor']

        clients[item['client_id']] = item
        fvalues += item['client_state']['approximate_f_value']

        # Remove sampled indicies in case of experiments with another SGD estimators inside optimization algorithms
        if 'iterated-minibatch-indicies' in item['client_state']:
            del item['client_state']['iterated-minibatch-indicies']

        #if 'iterated-minibatch-indicies-full' in item['client_state']:
        #    del item['client_state']['iterated-minibatch-indicies-full']

    # This place and serverGlobalStateUpdate() are only place where H['history'] can be updated
    assert round not in H['history']

    fRunAvg = np.nan
    if len(fvalues) > 0:
        fRunAvg = np.mean(fvalues)

    H['history'][round] = {}
    H['history'][round].update({ "client_states" : clients,
                                  "grad_sgd_server_l2" : mutils.l2_norm_of_vec(grad_server),
                                  "approximate_f_avg_value" : fRunAvg,
                                  "x_before_round" : mutils.l2_norm_of_vec(paramsPrev)
                               })

    if has_experiment_option(H, "track_distance_to_solution"):
        xSolutionFileName = H["execution_context"].experimental_options["x_solution"]
        if "used_x_solution" not in H:
            with open(xSolutionFileName, "rb") as f:
                import pickle
                obj = pickle.load(f)
                assert len(obj) == 1
                Hsol = obj[0][1]
                used_x_solution = Hsol['xfinal']
                H["used_x_solution"] = used_x_solution.to(dtype = paramsPrev.dtype)

        H['history'][round]["distance_to_solution"] = mutils.l2_norm_of_vec(paramsPrev - H["used_x_solution"].to(paramsPrev.device))

    classImpl = getImplClassForAlgo(H["algorithm"])
    Hnew = classImpl.serverGlobalStateUpdate(clients_responses, clients, model, paramsPrev, grad_server, H)

    # Update server state
    Hnew.update( {'server_state_update_time' : time.strftime("%d %b %Y %H:%M:%S UTC%z", time.localtime())} )

    # Move various shifts client information into CPU
    move_client_state_to_host = Hnew['args'].store_client_state_in_cpu

    if move_client_state_to_host:
        client_states = Hnew['history'][round]["client_states"]
        for client_id in client_states:
            # Client will be sampled in a next round. It's not worthwhile to copy state to CPU, because in next round we will need back it to GPU (TODO: Maybe still better to move to CPU)
            # if client_id in sampled_clients_per_next_round:
            #    continue

            cs = client_states[client_id]['client_state']
            for k,v in cs.items():
                if torch.is_tensor(v):
                    if v.device != 'cpu':
                        v = v.cpu()
                    client_states[client_id]['client_state'][k] = v

    return  Hnew
# ======================================================================================================================
