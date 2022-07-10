#!/usr/bin/env python3

import os

# Import PyTorch root package import torch
import torch

import shutil
import json
import copy

from utils.utils import get_model_str_from_obj
from utils.logger import Logger
from utils.utils import create_model_dir, create_metrics_dict
from utils import execution_context

import utils.model_funcs
import opts
from models import mutils
import numpy as np

def save_checkpoint(model, filename, args, is_best, metrics, metric_to_optim):
    """
    Persist checkpoint to disk.

    Args:
        model: Computation model
        filename: Filename to persist model by
        args: training setup
        is_best: Whether model with best metric
        metrics: metrics obtained from evaluation
        metric_to_optim: metric to optimize, e.g. top 1 accuracy
    """
    # Ignore saving checkpoints
    if args.do_not_save_eval_checkpoints:
        return

    logger = Logger.get(args.run_id)

    result_text = f"avg_loss={metrics['loss'].get_avg()}," \
                  f" avg_{metric_to_optim}={metrics[metric_to_optim].get_avg()}"
    metrics_dict = create_metrics_dict(metrics)

    # This is to persist the model without DataParallel wrapping. See also: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
    if get_model_str_from_obj(model) == "DataParallel":
        state = model.module
    else:
        state = model

    model_dir = create_model_dir(args)
    model_filename = os.path.join(model_dir, filename)
    result_filename = os.path.join(model_dir, 'results.txt')
    latest_filename = os.path.join(model_dir, 'latest.txt')

    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    last_filename = os.path.join(model_dir, 'model_last.pth.tar')

    best_metric_filename = os.path.join(model_dir, 'best_metrics.json')
    last_metric_filename = os.path.join(model_dir, 'last_metrics.json')

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    logger.info("Saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result_text)

    # Save entire model
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    save_dict_to_json(metrics_dict, last_metric_filename)
    if is_best:
        logger.info("Found new best.")
        shutil.copyfile(model_filename, best_filename)
        shutil.copyfile(last_metric_filename, best_metric_filename)
    logger.info("Copying to {}".format(last_filename))
    shutil.copyfile(model_filename, last_filename)

    # Remove all previous checkpoints except for the best one for storage savings
    logger.info("Removing redundant files")
    files_to_keep = ['model_last.pth.tar', 'model_best.pth.tar',
                     'results.txt', 'latest.txt', 'args.json',
                     'last_metrics.json', 'best_metrics.json']
    files_to_delete = [file for file in os.listdir(model_dir) if file not in files_to_keep]
    for f in files_to_delete:
        if not os.path.isdir(f):
            os.remove("{}/{}".format(model_dir, f))
    return

def save_checkpoint_worker(threadCtx, model, model_params, filename, args, metrics, metric_to_optim, current_round, exec_ctx):
    """
    Worker routine dedicated for check is model is best and serialize it into filesystem.

    Args:
        threadCtx(ThreadPool): thread context (can be None)
        model(torch.nn.Moduel): model instance which can be used for serialization (read, write)
        model_params(torch.Tensor): model parameters
        filename(str): filename with place where model should be serialized
        args: command linea arguments
        metrics: evaluate metrics for the model with model_params
        metric_to_optim(str): metric which is used for optimization
        current_round(int): current round

    Returns:
        True if dispatching happens fine. False if there are no threads in a thread pool or all threads have already complete their work.
    """
    logger = Logger.get(args.run_id)

    if threadCtx is not None:
        device_used = threadCtx.device
    else:
        device_used = args.device

    # Current logic for saving does not require GPU deivce for computation

    avg_metric = metrics[metric_to_optim].get_avg()
    mutils.set_params(model, model_params)

    cur_metrics = {"loss": metrics["loss"].get_avg(),
                   "top_1_acc": metrics["top_1_acc"].get_avg(),
                   "top_5_acc": metrics["top_5_acc"].get_avg(),
                   "neq_perplexity": metrics["neq_perplexity"].get_avg()
                   }

    is_best = avg_metric > exec_ctx.saver_thread.best_metric

    if is_best:
        exec_ctx.saver_thread.best_metric_lock.acquire()
        exec_ctx.saver_thread.eval_metrics.update({metrics['round']: cur_metrics})
        exec_ctx.saver_thread.best_metric = avg_metric
        exec_ctx.saver_thread.best_metric_lock.release()

        save_checkpoint(model, filename, args, is_best, metrics, metric_to_optim)
    else:
        exec_ctx.saver_thread.best_metric_lock.acquire()
        exec_ctx.saver_thread.eval_metrics.update({metrics['round']: cur_metrics})
        exec_ctx.saver_thread.best_metric_lock.release()

        save_checkpoint(model, filename, args, is_best, metrics, metric_to_optim)

def eval_and_save_checkpoint_worker(threadCtx, model, params, testloader, criterion, args, current_round, is_rnn, metric_to_optim, exec_ctx)->bool:
    logger = Logger.get(args.run_id)

    if threadCtx is not None:
        device_used = threadCtx.device
    else:
        device_used = args.device

    mutils.print_current_gpu_context(device_used, args)
    mutils.set_params(model, params)
    metrics = model_funcs.evaluate_model(model, testloader, criterion, device_used, current_round,
                                         print_freq=10, is_rnn=is_rnn, metric_to_optim=metric_to_optim)

    avg_metric = metrics[metric_to_optim].get_avg()
    # Save model checkpoint
    model_filename = '{model}_{run_id}_checkpoint_{round:0>2d}.pth.tar' \
                     .format(model=args.model, run_id=args.run_id, round=current_round)

    if exec_ctx.saver_thread.workers() > 0:
        # there are dedicated workers for save
        _model = exec_ctx.saver_thread.next_dispatched_thread().model_copy
        fargs = (_model, params, model_filename, args, metrics, metric_to_optim, current_round, exec_ctx)
        exec_ctx.saver_thread.dispatch(save_checkpoint_worker, fargs)
    else:
        # there are no workers for save threads
        fargs = (None, model, params, model_filename, args, metrics, metric_to_optim, current_round, exec_ctx)
        save_checkpoint_worker(*fargs)

    if np.isnan(metrics['loss'].get_avg()):
        logger.critical('NaN loss detected, aborting training procedure.')
        # TODO: HANDLE PROPERLY
        return True
    else:
        return True

def defered_eval_and_save_checkpoint(model, criterion, args, current_round, is_rnn, metric_to_optim, exec_ctx):
    """
    Request for evalute models and after that serialize it asyncronously.

    Args:
        model(torch.nn.Module): model which is assesed (READ ONLY)
        criterion(torch.nn.modules.loss): used criteria for target loss function
        args: command line argument which conffigured this launch
        current_round(int): number of a current communication round which has recently completed
        is_rnn(bool): Flag which describes do we handle RNN model
        metric_to_optim(str): Metric for optimize for validation

    Returns:
    """
    eval_thread = exec_ctx.eval_thread_pool.next_dispatched_thread()

    _model = eval_thread.model_copy
    _params = mutils.get_params(model)
    _testloader = eval_thread.testloader_copy

    arguments_for_call = (_model, _params, _testloader, criterion, args, current_round, is_rnn, metric_to_optim, exec_ctx)
    function_for_call = eval_and_save_checkpoint_worker
    exec_ctx.eval_thread_pool.dispatch(function_for_call, arguments_for_call)

def load_checkpoint(model_dir, load_best=True):
    """
    Load model from checkpoint.

    Args:
        model_dir: Directory to read the model from.
        load_best: Whether to read best or latest version of the model

    Returns:
        The state dictionary of the model
    """
    logger = Logger.get("default")

    if load_best:
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
        metric_filename = os.path.join(model_dir, 'best_metrics.json')
    else:
        model_filename = os.path.join(model_dir, 'model_last.pth.tar')
        metric_filename = os.path.join(model_dir, 'last_metrics.json')

    logger.info("Loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    logger.info("Loaded checkpoint '{}'".format(model_filename))
    # read checkpoint json
    with open(metric_filename, 'r') as f:
        metrics = json.load(f)
    return state, metrics['round']


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) if (isinstance(v, float) or isinstance(v, int)) else [float(e) for e in v]
             for k, v in d.items()}
        json.dump(d, f, indent=4)
