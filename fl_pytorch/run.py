#!/usr/bin/env python3

import os
import sys
import json
import socket
import math
import signal

# Import PyTorch root package import torch
import torch
from torch.utils.collect_env import get_pretty_env_info

import numpy as np
import time
import copy
import threading
import pickle

from utils import comm_socket
from utils import gpu_utils
from utils import execution_context
from utils import thread_pool
from utils import wandb_wrapper

from opts import parse_args

from utils.logger import Logger
from data_preprocess.data_loader import load_data, get_test_batch_size
from utils.model_funcs import get_training_elements, evaluate_model, get_lr_scheduler, get_optimiser, \
    run_one_communication_round, local_training, initialise_model

from utils.checkpointing import save_checkpoint
from utils.checkpointing import defered_eval_and_save_checkpoint

from utils.utils import create_model_dir, create_metrics_dict
from utils.fl_funcs import get_sampled_clients

from models import RNN_MODELS
from models import mutils

from utils import algorithms

from utils.buffer import Buffer


def main(args, raw_cmdline, extra_):
    # Init project and save in args url for the plots
    projectWB = None
    if len(args.wandb_key) > 0:
        projectWB = wandb_wrapper.initWandbProject(
            args.wandb_key,
            args.wandb_project_name,
            f"{args.algorithm}-rounds-{args.rounds}-{args.model}@{args.dataset}-runid-{args.run_id}",
            args
        )
        if projectWB is not None:
            args.wandb_url = projectWB.url

    # In case of DataParallel for .to() to
    # This a device used by master for FL algorithms
    args.device = gpu_utils.get_target_device_str(args.gpu[0]) if type(args.gpu) == list else args.gpu

    # Instantiate execution context
    exec_ctx = execution_context.initExecutionContext()
    exec_ctx.extra_ = extra_

    # Set all seeds
    if args.deterministic:
        exec_ctx.random.seed(args.manual_init_seed)
        exec_ctx.np_random.seed(args.manual_init_seed)

    # Setup extra options for experiments
    for option in args.algorithm_options.split(","):
        kv = option.split(":")
        if len(kv) == 1:
            exec_ctx.experimental_options[kv[0]] = True
        else:
            exec_ctx.experimental_options[kv[0]] = kv[1]

    # Load validation set
    trainset, testset = load_data(exec_ctx, args.data_path, args.dataset, args, load_trainset=True, download=True)

    test_batch_size = get_test_batch_size(args.dataset, args.batch_size)

    # Test set dataloader
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=test_batch_size,
                                             num_workers=args.num_workers_test,
                                             shuffle=False,
                                             pin_memory=False,
                                             drop_last=False
                                             )

    # Reset all seeds
    if args.deterministic:
        exec_ctx.random.seed(args.manual_init_seed)
        exec_ctx.np_random.seed(args.manual_init_seed)

    if not args.evaluate:  # Training mode
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers_train,
            shuffle=False,
            pin_memory=False,
            drop_last=False)

        # Is target device for master is GPU
        is_target_dev_gpu = gpu_utils.is_target_dev_gpu(args.device)

        # Initialize local training threads in case of using multithreading implementation
        exec_ctx.local_training_threads.adjust_num_workers(w=args.threadpool_for_local_opt, own_cuda_stream=True,
                                                           device_list=args.gpu)
        for th in exec_ctx.local_training_threads.threads:
            th.trainset_copy = copy.deepcopy(trainset)
            th.train_loader_copy = torch.utils.data.DataLoader(th.trainset_copy,
                                                               batch_size=args.batch_size,
                                                               num_workers=args.num_workers_train,
                                                               shuffle=False,
                                                               pin_memory=False,
                                                               drop_last=False)
            if hasattr(th.trainset_copy, "load_data"):
                th.trainset_copy.load_data()

        # Initialize remote clients - one remote connection per one remote computation device
        external_devices = [d.strip() for d in args.external_devices.split(",") if len(d.strip()) > 0]
        exec_ctx.remote_training_threads.adjust_num_workers(w=len(external_devices), own_cuda_stream=False,
                                                            device_list=["cpu"])
        for th_i in range(len(exec_ctx.remote_training_threads.threads)):
            th = exec_ctx.remote_training_threads.threads[th_i]
            dev = external_devices[th_i]
            th.external_dev_long = dev
            th.external_ip, th.external_port, th.external_dev, th.external_dev_dumber = dev.split(":")

            try:
                th.external_socket = comm_socket.CommSocket()
                th.external_socket.sock.connect((th.external_ip, int(th.external_port)))  # connect with remote side
                th.external_socket.rawSendString("execute_work")  # initiate work execution COMMAND
                th.external_socket.rawSendString(th.external_dev_long)  # provide device specification
                th.external_socket.rawSend(pickle.dumps(
                    raw_cmdline))  # provide original command line (it will be changed by the client) to be consistent with used devices
                th.external_socket_online = True
            except socket.error:
                th.external_socket_online = False

        # Initialize saver treads for information serialization (0 is acceptable)
        exec_ctx.saver_thread.adjust_num_workers(w=args.save_async_threads, own_cuda_stream=True, device_list=args.gpu)

        # Initialize local training threads in case of using multithreading implementation (0 is acceptable)
        exec_ctx.eval_thread_pool.adjust_num_workers(w=args.eval_async_threads, own_cuda_stream=True,
                                                     device_list=args.gpu)

        for th in exec_ctx.eval_thread_pool.threads:
            th.testset_copy = copy.deepcopy(testset)
            th.testloader_copy = torch.utils.data.DataLoader(th.testset_copy,
                                                             batch_size=test_batch_size,
                                                             num_workers=args.num_workers_test,
                                                             shuffle=False,
                                                             pin_memory=False,
                                                             drop_last=False
                                                             )
            if hasattr(th.testset_copy, "load_data"):
                th.testset_copy.load_data()

        if args.worker_listen_mode <= 0:
            # Path of execution for local simulation
            init_and_train_model(args, raw_cmdline, trainloader, testloader, exec_ctx)
        else:
            # Path of execution for assist simulation with a remote means
            init_and_help_with_compute(args, raw_cmdline, trainloader, testloader, exec_ctx)

    else:  # Evaluation mode
        # TODO: figure out how to exploit DataParallel. Currently - parallelism across workers
        model, criterion, round = get_training_elements(args.model, args.dataset, args, args.resume_from,
                                                        args.load_best, args.device, args.loss,
                                                        args.turn_off_batch_normalization_and_dropout)

        metrics = evaluate_model(model, testloader, criterion, args.device, round,
                                 print_freq=10, metric_to_optim=args.metric,
                                 is_rnn=args.model in RNN_MODELS)

        metrics_dict = create_metrics_dict(metrics)
        logger.info(f'Validation metrics: {metrics_dict}')
    wandb_wrapper.finishProject(projectWB)


def init_and_help_with_compute(args, raw_cmdline, trainloader, testloader, exec_ctx):
    logger = Logger.get(args.run_id)

    model_dir = create_model_dir(args)
    # don't train if setup already exists
    if os.path.isdir(model_dir):
        logger.info(f"{model_dir} already exists.")
        logger.info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    # save used args as json to experiment directory
    with open(os.path.join(create_model_dir(args), 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    is_rnn = args.model in RNN_MODELS
    # TODO: figure out how to exploit DataParallel. Currently - parallelism across workers
    model, criterion, current_round = get_training_elements(args.model, args.dataset, trainloader.dataset, args,
                                                            args.resume_from, args.load_best, args.device, args.loss,
                                                            args.turn_off_batch_normalization_and_dropout)
    # ==================================================================================================================
    # Reset execution seeds for tunable runtime behaviour
    if args.deterministic:
        exec_ctx.random.seed(args.manual_runtime_seed)
        exec_ctx.np_random.seed(args.manual_runtime_seed)
    # ==================================================================================================================
    mutils.print_models_info(model, args)
    logger.info(f"Number of parameters in the model: {mutils.number_of_params(model):,d}\n")
    gpu_utils.print_info_about_used_gpu(args.device, args.run_id)
    # ==================================================================================================================
    # Initialize server state
    # ==================================================================================================================
    trainloader.dataset.set_client(None)
    algorithms.evaluateGradient(None, model, trainloader, criterion, is_rnn, update_statistics=False,
                                evaluate_function=False, device=args.device, args=args)
    gradient_at_start = mutils.get_gradient(model)
    for p in model.parameters():
        p.grad = None
    H = algorithms.initializeServerState(args, model, trainloader.dataset.num_clients, gradient_at_start, exec_ctx)
    logger.info(f'D: {H["D"]} / D_include_frozen : {H["D_include_frozen"]}')

    # Append all launch arguments (setup and default)
    H["args"] = args
    H["raw_cmdline"] = raw_cmdline
    H["execution_context"] = exec_ctx
    H["total_clients"] = trainloader.dataset.num_clients
    H["comment"] = args.comment
    H["group-name"] = args.group_name
    # ==================================================================================================================
    local_optimiser = get_optimiser(model.parameters(), args.local_optimiser, args.local_lr, args.local_momentum,
                                    args.local_weight_decay)
    socket = exec_ctx.extra_
    state_dicts_thread_safe = Buffer()

    while True:
        # Main loop for obtaining commands from master
        cmd = socket.rawRecvString()
        if cmd != "finish_work" and cmd != "non_local_training":
            print("Unknown command: ", cmd)
            break

        if cmd == "finish_work":
            # Work termination
            break

        if cmd == "non_local_training":
            # Remote training. Param# 1- all params with local training
            msg = socket.rawRecv()
            serialized_args = pickle.loads(msg)

            (client_state, client_id, msg, model_dict_original, optimiser_dict_original,
             model_, train_loader_, criterion, local_optimiser_, device_, round, run_local_iters,
             number_of_local_steps, is_rnn, print_freq) = serialized_args

            # 1. Update to client specific parameters - compute device, execution context
            # 2. Move all tensors into device
            device = args.device
            client_state['device'] = args.device

            client_state['H']["execution_context"] = H["execution_context"]

            for k, v in client_state['H'].items():
                if torch.is_tensor(v):
                    client_state['H'][k] = v.to(device=device)

            model_dict_original = model_dict_original.to(device=device)

            res = local_training(None, client_state, client_id, msg, model_dict_original, optimiser_dict_original,
                                 model, trainloader, criterion, local_optimiser, device, round, run_local_iters,
                                 number_of_local_steps, is_rnn, print_freq, state_dicts_thread_safe,
                                 client_state['H']["run_id"])
            # ==========================================================================================================
            # Training is finished - response
            socket.rawSendString("result_of_local_training")
            msg = state_dicts_thread_safe.popFront()
            # Temporary remove reference to server state from client
            h_ctx = msg['client_state']['H']
            # Remove server state from sending
            del msg['client_state']['H']
            for k, v in msg.items():
                if torch.is_tensor(v):
                    msg[k] = v.cpu()
            socket.rawSend(pickle.dumps(msg))
            # Revert reference to server state from client
            msg['client_state']['H'] = h_ctx
            # ==========================================================================================================


def makeBackupOfServerState(H, round):
    args = H["args"]
    job_id = args.run_id
    fname = args.out

    if fname == "":
        fname = os.path.join(args.checkpoint_dir, f"{args.run_id}_{args.algorithm}_{args.global_lr}_backup.bin")
    else:
        fname = fname.replace(".bin", f"{args.run_id}_{args.algorithm}_{args.global_lr}_backup.bin")

    execution_context = H["execution_context"]
    H["execution_context"] = None
    exec_ctx_in_arg = ('exec_ctx' in H["args"])
    exec_ctx = None
    if exec_ctx_in_arg:
        exec_ctx = H["args"]['exec_ctx']
        H["execution_context"] = None

    with open(fname, "wb") as f:
        pickle.dump([(job_id, H)], f)

    H["execution_context"] = execution_context
    if exec_ctx_in_arg:
        H["args"]['exec_ctx'] = exec_ctx


def init_and_train_model(args, raw_cmdline, trainloader, testloader, exec_ctx):
    logger = Logger.get(args.run_id)
    model_dir = create_model_dir(args)
    # don't train if setup already exists
    if os.path.isdir(model_dir):
        logger.info(f"{model_dir} already exists.")
        logger.info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    # save used args as json to experiment directory
    with open(os.path.join(create_model_dir(args), 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    is_rnn = args.model in RNN_MODELS
    # TODO: figure out how to exploit DataParallel. Currently - parallelism across workers
    model, criterion, current_round = get_training_elements(args.model, args.dataset, trainloader.dataset, args,
                                                            args.resume_from, args.load_best, args.device, args.loss,
                                                            args.turn_off_batch_normalization_and_dropout)
    # ==================================================================================================================
    # Reset execution seeds for tunable runtime behaviour
    if args.deterministic:
        exec_ctx.random.seed(args.manual_runtime_seed)
        exec_ctx.np_random.seed(args.manual_runtime_seed)
    # ==================================================================================================================

    local_optimiser = get_optimiser(model.parameters(), args.local_optimiser, args.local_lr,
                                    args.local_momentum, args.local_weight_decay)

    local_scheduler = get_lr_scheduler(local_optimiser, args.rounds, args.local_lr_type)

    global_optimiser = get_optimiser(model.parameters(), args.global_optimiser, args.global_lr,
                                     args.global_momentum, args.global_weight_decay)

    global_scheduler = get_lr_scheduler(global_optimiser, args.rounds, args.global_lr_type)

    metric_to_optim = args.metric
    train_time_meter = 0

    best_metric = -np.inf
    eval_metrics = {}

    mutils.print_models_info(model, args)
    logger.info(f"Number of parameters in the model: {mutils.number_of_params(model):,d}\n")

    gpu_utils.print_info_about_used_gpu(args.device, args.run_id)

    sampled_clients = get_sampled_clients(trainloader.dataset.num_clients, args, exec_ctx)

    # ============= Initialize worker threads===========================================================================
    for th in exec_ctx.local_training_threads.threads:
        th.model_copy = copy.deepcopy(model)
        # th.model_copy, _ = initialise_model(args.model, args.dataset, trainloader.dataset, args, args.resume_from,
        #                                     args.load_best)
        # th.model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        th.model_copy = th.model_copy.to(th.device)

        th.local_optimiser_copy = get_optimiser(th.model_copy.parameters(), args.local_optimiser, args.local_lr,
                                                args.local_momentum, args.local_weight_decay)
        th.local_scheduler = get_lr_scheduler(th.local_optimiser_copy, args.rounds, args.local_lr_type)

    for th in exec_ctx.eval_thread_pool.threads:
        th.model_copy = copy.deepcopy(model)
        # th.model_copy, _ = initialise_model(args.model, args.dataset, trainloader.dataset, args, args.resume_from,
        #                                     args.load_best)
        # th.model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        th.model_copy = th.model_copy.to(th.device)

    for th in exec_ctx.saver_thread.threads:
        th.model_copy = copy.deepcopy(model)
        # th.model_copy, _ = initialise_model(args.model, args.dataset, trainloader.dataset, args, args.resume_from,
        #                                     args.load_best)
        # th.model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        th.model_copy = th.model_copy.to(th.device)

    exec_ctx.saver_thread.best_metric = -np.inf
    exec_ctx.saver_thread.eval_metrics = {}
    exec_ctx.saver_thread.best_metric_lock = threading.Lock()
    # ==================================================================================================================
    optimiser_dict_original = copy.deepcopy(local_optimiser.state_dict())
    # ==================================================================================================================

    logger.info(f"Use separate {exec_ctx.local_training_threads.workers()} worker threads for make local optimization")

    # Initialize server state

    if args.initialize_shifts_policy == "full_gradient_at_start":
        trainloader.dataset.set_client(None)
        algorithms.evaluateGradient(None, model, trainloader, criterion, is_rnn, update_statistics=False,
                                    evaluate_function=False, device=args.device, args=args)
        gradient_at_start = mutils.get_gradient(model)
        for p in model.parameters():
            p.grad = None
    else:
        D = mutils.number_of_params(model)
        gradient_at_start = torch.zeros(D).to(args.device)

    H = algorithms.initializeServerState(args, model, trainloader.dataset.num_clients, gradient_at_start, exec_ctx)
    logger.info(f'D: {H["D"]} / D_include_frozen : {H["D_include_frozen"]}')

    # ==================================================================================================================
    # Append all launch arguments (setup and default)
    H["args"] = args
    H["raw_cmdline"] = raw_cmdline
    H["execution_context"] = exec_ctx
    H["total_clients"] = trainloader.dataset.num_clients
    H["comment"] = args.comment
    H["group-name"] = args.group_name
    # H["sampled_clients"] = sampled_clients

    # Probable pre-scale x0
    if algorithms.has_experiment_option(H, "x0_norm"):
        H["x0"] = (H["x0"] / H["x0"].norm()) * algorithms.get_experiment_option_f(H, "x0_norm")

    # For task where we can obtain good estimation of L
    if "L" in dir(trainloader.dataset):
        H["L_compute"] = trainloader.dataset.L
    if "Li_all_clients" in dir(trainloader.dataset):
        H["Li_all_clients"] = trainloader.dataset.Li_all_clients
        H["Li_max"] = max(H["Li_all_clients"])

    # Initialize metrics
    H['best_metric'] = best_metric
    H['eval_metrics'] = copy.deepcopy(eval_metrics)

    # Initialize starting point of optimization
    mutils.set_params(model, H['x0'])

    is_target_gpu = gpu_utils.is_target_dev_gpu(args.device)

    if execution_context.simulation_start_fn:
        execution_context.simulation_start_fn(H)

    extra_opts = [a.strip() for a in args.extra_track.split(",")]

    fed_dataset_all_clients = np.arange(H['total_clients'])

    # Main Algorithm Optimization loop
    for i in range(args.rounds):
        # Check requirement to force stop simulation
        if execution_context.is_simulation_need_earlystop_fn is not None:
            if execution_context.is_simulation_need_earlystop_fn(H):
                break

        # Check that all is ok with current run
        if i > 0:
            prev_history_stats = H['history'][i - 1]
            forceTermination = False

            for elem in ["x_before_round", "grad_sgd_server_l2"]:
                if elem in prev_history_stats:
                    if math.isnan(prev_history_stats[elem]) or math.isinf(prev_history_stats[elem]):
                        logger.error(f"Force early stop due to numerical problems with {elem} at round {i}.")
                        forceTermination = True
                        break

            if forceTermination:
                break

        # Update information about current round
        H['current_round'] = i

        start = time.time()
        # Generate client state
        # TODO: metrics meter to be used for Tensorboard/WandB
        metrics_meter, fed_dataset_clients = run_one_communication_round(H, model, trainloader, criterion,
                                                                         local_optimiser,
                                                                         optimiser_dict_original, global_optimiser,
                                                                         args.device,
                                                                         current_round, args.run_local_steps,
                                                                         args.number_of_local_iters,
                                                                         is_rnn, sampled_clients)
        train_time = time.time() - start
        if H['algorithm'] == 'gradskip':
            train_time = H['time']
        train_time_meter += train_time  # Track timings for across epochs average
        logger.debug(f'Epoch train time: {train_time}')
        # H['history'][current_round]["xi_after"] = mutils.get_params(model)
        H['history'][current_round]["train_time"] = train_time
        H['history'][current_round]["time"] = time.time() - exec_ctx.context_init_time
        if H['algorithm'] == 'gradskip':
            if current_round == 0:
                H['history'][current_round]["time"] = train_time
            else:
                H['history'][current_round]["time"] = H['history'][current_round - 1]["time"] + train_time
        H['last_round_elapsed_sec'] = train_time

        if (i % args.eval_every == 0 or i == (args.rounds - 1)) and metric_to_optim != "none":
            # Save results obtained so far into backup file
            # ==========================================================================================================
            # Serialize server state into filesystem"
            makeBackupOfServerState(H, i)
            # ==========================================================================================================
            # Evaluate model
            if args.eval_async_threads > 0:
                defered_eval_and_save_checkpoint(model, criterion, args, current_round, is_rnn=is_rnn,
                                                 metric_to_optim=metric_to_optim, exec_ctx=exec_ctx)

                # Update recent information about eval metrics
                exec_ctx.saver_thread.best_metric_lock.acquire()
                H['best_metric'] = max(H['best_metric'], exec_ctx.saver_thread.best_metric)
                H['eval_metrics'] = copy.deepcopy(exec_ctx.saver_thread.eval_metrics)
                exec_ctx.saver_thread.best_metric_lock.release()

            else:
                metrics = evaluate_model(model, testloader, criterion, args.device, current_round, print_freq=10,
                                         is_rnn=is_rnn, metric_to_optim=metric_to_optim)

                avg_metric = metrics[metric_to_optim].get_avg()

                cur_metrics = {"loss": metrics["loss"].get_avg(),
                               "top_1_acc": metrics["top_1_acc"].get_avg(),
                               "top_5_acc": metrics["top_5_acc"].get_avg(),
                               "neq_perplexity": metrics["neq_perplexity"].get_avg()
                               }

                eval_metrics.update({metrics['round']: cur_metrics})

                # Save model checkpoint
                model_filename = '{model}_{run_id}_checkpoint_{round:0>2d}.pth.tar'.format(model=args.model,
                                                                                           run_id=args.run_id,
                                                                                           round=current_round)

                is_best = avg_metric > best_metric
                save_checkpoint(model, model_filename, is_best=is_best, args=args, metrics=metrics,
                                metric_to_optim=metric_to_optim)

                if is_best:
                    best_metric = avg_metric

                # Update recent information about eval metrics
                H['best_metric'] = max(H['best_metric'], best_metric)
                H['eval_metrics'] = copy.deepcopy(eval_metrics)

                if np.isnan(metrics['loss'].get_avg()):
                    logger.critical('NaN loss detected, aborting training procedure.')
                #   return

                logger.info(f'Current lrs global:{global_scheduler.get_last_lr()}')
        # ==============================================================================================================
        # Add number of clients in that round
        H['history'][current_round]["number_of_client_in_round"] = len(fed_dataset_clients)

        if args.log_gpu_usage:
            H['history'][current_round]["memory_gpu_used"] = 0
            for dev in args.gpu:
                if gpu_utils.is_target_dev_gpu(dev):
                    memory_gpu_used = torch.cuda.memory_stats(args.device)['reserved_bytes.all.current']
                    logger.info(
                        f"GPU: Before round {i} we have used {memory_gpu_used / (1024.0 ** 2):.2f} MB from device {dev}")
                    H['history'][current_round]["memory_gpu_used"] += memory_gpu_used / (1024.0 ** 2)

        if current_round % args.eval_every == 0 or i == (args.rounds - 1):
            if "full_gradient_norm_train" in extra_opts and "full_objective_value_train" in extra_opts:
                fed_dataset = trainloader.dataset

                gradient_avg = mutils.get_zero_gradient_compatible_with_model(model)
                fvalue_avg = 0.0

                for c in fed_dataset_all_clients:
                    fed_dataset.set_client(c)
                    fvalue = algorithms.evaluateGradient(None, model, trainloader, criterion, is_rnn,
                                                         update_statistics=False, evaluate_function=True,
                                                         device=args.device, args=args)
                    g = mutils.get_gradient(model)

                    gradient_avg = (gradient_avg * c + g) / (c + 1)
                    fvalue_avg = (fvalue_avg * c + fvalue) / (c + 1)

                H['history'][current_round]["full_gradient_norm_train"] = mutils.l2_norm_of_vec(gradient_avg)
                H['history'][current_round]["full_objective_value_train"] = fvalue_avg

            if "full_gradient_norm_train" in extra_opts and "full_objective_value_train" not in extra_opts:
                fed_dataset = trainloader.dataset
                gradient_avg = mutils.get_zero_gradient_compatible_with_model(model)

                for c in fed_dataset_all_clients:
                    fed_dataset.set_client(c)
                    algorithms.evaluateGradient(None, model, trainloader, criterion, is_rnn, update_statistics=False,
                                                evaluate_function=False, device=args.device, args=args)
                    g = mutils.get_gradient(model)
                    gradient_avg = (gradient_avg * c + g) / (c + 1)

                H['history'][current_round]["full_gradient_norm_train"] = mutils.l2_norm_of_vec(gradient_avg)

            if "full_objective_value_train" in extra_opts and "full_gradient_norm_train" not in extra_opts:
                fed_dataset = trainloader.dataset
                fvalue_avg = 0.0
                for c in fed_dataset_all_clients:
                    fed_dataset.set_client(c)
                    fvalue = algorithms.evaluateFunction(None, model, trainloader, criterion, is_rnn,
                                                         update_statistics=False, device=args.device, args=args)
                    fvalue_avg = (fvalue_avg * c + fvalue) / (c + 1)
                H['history'][current_round]["full_objective_value_train"] = fvalue_avg
            # ==========================================================================================================
            if "full_gradient_norm_val" in extra_opts and "full_objective_value_val" in extra_opts:
                fvalue = algorithms.evaluateGradient(None, model, testloader, criterion, is_rnn,
                                                     update_statistics=False, evaluate_function=True,
                                                     device=args.device, args=args)
                H['history'][current_round]["full_gradient_norm_val"] = mutils.l2_norm_of_gradient_m(model)
                H['history'][current_round]["full_objective_value_val"] = fvalue

            if "full_gradient_norm_val" in extra_opts and "full_objective_value_val" not in extra_opts:
                algorithms.evaluateGradient(None, model, testloader, criterion, is_rnn, update_statistics=False,
                                            evaluate_function=False, device=args.device, args=args)
                H['history'][current_round]["full_gradient_norm_val"] = mutils.l2_norm_of_gradient_m(model)

            if "full_objective_value_val" in extra_opts and "full_gradient_norm_val" not in extra_opts:
                fvalue = algorithms.evaluateFunction(None, model, testloader, criterion, is_rnn,
                                                     update_statistics=False, device=args.device, args=args)
                H['history'][current_round]["full_objective_value_val"] = fvalue
            # ==========================================================================================================
            # Interpolate
            # ==========================================================================================================
            if current_round > 0:
                look_behind = args.eval_every

                # ======================================================================================================
                if i % args.eval_every == 0:
                    look_behind = args.eval_every  # Exactly look_behind previous evaluation had place to be
                elif i == args.rounds - 1:
                    look_behind = i % args.eval_every  # We are in case that we in last round "R-1".
                    pass  # Exactly i%args.eval_every round before previous eval had place to be
                else:
                    assert (not "Impossible case")
                # ======================================================================================================

                for s in range(1, look_behind):
                    alpha = s / float(look_behind)
                    prev_eval_round = current_round - look_behind
                    inter_round = prev_eval_round + s

                    # Linearly interpolate between a (alpha=0.0) and b(alpha=1.0)
                    def lerp(a, b, alpha):
                        return (1.0 - alpha) * a + alpha * b

                    if "full_gradient_norm_train" in extra_opts:
                        H['history'][inter_round]["full_gradient_norm_train"] = lerp(
                            H['history'][prev_eval_round]["full_gradient_norm_train"],
                            H['history'][current_round]["full_gradient_norm_train"], alpha)
                    if "full_objective_value_train" in extra_opts:
                        H['history'][inter_round]["full_objective_value_train"] = lerp(
                            H['history'][prev_eval_round]["full_objective_value_train"],
                            H['history'][current_round]["full_objective_value_train"], alpha)
                    if "full_gradient_norm_val" in extra_opts:
                        H['history'][inter_round]["full_gradient_norm_val"] = lerp(
                            H['history'][prev_eval_round]["full_gradient_norm_val"],
                            H['history'][current_round]["full_gradient_norm_val"], alpha)
                    if "full_objective_value_val" in extra_opts:
                        H['history'][inter_round]["full_objective_value_val"] = lerp(
                            H['history'][prev_eval_round]["full_objective_value_val"],
                            H['history'][current_round]["full_objective_value_val"], alpha)

        if current_round % args.eval_every == 0 or i == (args.rounds - 1):
            if execution_context.simulation_progress_steps_fn is not None:
                execution_context.simulation_progress_steps_fn(i / float(args.rounds) * 0.75, H)
            wandb_wrapper.logStatistics(H, current_round)

        # ==============================================================================================================
        # Save parameters of the last model
        # if i == args.rounds - 1:
        #    xfinal = mutils.get_params(model)
        #    H['xfinal'] = xfinal
        # ==============================================================================================================
        # Update schedulers
        if exec_ctx.local_training_threads.workers() > 0:
            # In this mode local optimizers for worker threads are used
            for th in exec_ctx.local_training_threads.threads:
                th.local_scheduler.step()
        else:
            # In this mode local optimizer is used
            local_scheduler.step()

        global_scheduler.step()

        # Increment rounder counter
        current_round += 1

        # Empty caches. Force cleanup the cache after round (good to fix fragmentation issues)
        if args.per_round_clean_torch_cache:
            execution_context.torch_global_lock.acquire()
            torch.cuda.empty_cache()
            execution_context.torch_global_lock.release()

    xfinal = mutils.get_params(model)
    H['xfinal'] = xfinal

    logger.debug(f'Average epoch train time: {train_time_meter / args.rounds}')
    # ==================================================================================================================
    logger.info("Wait for threads from a training threadpool")
    exec_ctx.local_training_threads.stop()
    if execution_context.simulation_progress_steps_fn is not None:
        execution_context.simulation_progress_steps_fn(0.8, H)

    logger.info("Wait for threads from a evaluate threadpool")
    exec_ctx.eval_thread_pool.stop()
    if execution_context.simulation_progress_steps_fn is not None:
        execution_context.simulation_progress_steps_fn(0.95, H)

    logger.info("Wait for threads from a serialization threadpool")
    exec_ctx.saver_thread.stop()

    # Not need for use in fact, but we do it for purpose of excluding any bugs inside PyTorch

    logger.info("Synchronize all GPU streams")
    if is_target_gpu:
        torch.cuda.synchronize(args.device)

    # Update metrics based on eval and save results
    H['best_metric'] = max(H['best_metric'], exec_ctx.saver_thread.best_metric)
    H['best_metric'] = max(H['best_metric'], best_metric)

    if len(eval_metrics) > 0:
        H['eval_metrics'] = copy.deepcopy(eval_metrics)
    else:
        H['eval_metrics'] = copy.deepcopy(exec_ctx.saver_thread.eval_metrics)

    if execution_context.simulation_progress_steps_fn is not None:
        execution_context.simulation_progress_steps_fn(1.0, H)
        # ==================================================================================================================
    # Final prune
    HKeys = list(H.keys())

    # Remove any tensor field huger then 1 MBytes from final solution
    tensor_prune_mb_threshold = 0.0

    for field in HKeys:
        if torch.is_tensor(H[field]):
            size_in_mbytes = H[field].numel() * H[field].element_size() / (1024.0 ** 2)
            if size_in_mbytes > tensor_prune_mb_threshold:
                del H[field]
                continue
            else:
                # Convert any not removed tensor to CPU
                H[field] = H[field].cpu()

        if type(H[field]) == list:
            sz = len(H[field])
            i = 0

            while i < sz:
                if torch.is_tensor(H[field][i]):
                    size_in_mbytes = H[field][i].numel() * H[field][i].element_size() / (1024.0 ** 2)
                    if size_in_mbytes >= tensor_prune_mb_threshold:
                        del H[field][i]
                        sz = len(H[field])
                    else:
                        # Convert any not removed tensor to CPU
                        H[field][i] = H[field][i].cpu()
                        i += 1
                else:
                    i += 1

    # Remove local optimiser state, client_compressors, reference to server state
    not_to_remove_from_client_state = ["error"]

    for round, history_state in H['history'].items():
        clients_history = history_state['client_states']
        for client_id, client_state in clients_history.items():
            client_state = client_state['client_state']

            if 'H' in client_state:
                del client_state['H']
            if 'optimiser' in client_state:
                del client_state['optimiser']
            if 'buffers' in client_state:
                del client_state['buffers']
            if 'client_compressor' in client_state:
                del client_state['client_compressor']

            client_state_keys = list(client_state.keys())
            for field in client_state_keys:

                if field in not_to_remove_from_client_state:
                    continue

                if torch.is_tensor(client_state[field]):
                    size_in_mbytes = client_state[field].numel() * client_state[field].element_size() / (1024.0 ** 2)
                    if size_in_mbytes > tensor_prune_mb_threshold:
                        del client_state[field]
                        continue
                    else:
                        # Convert any not removed tensor to CPU
                        client_state[field] = client_state[field].cpu()
                        continue

                if type(client_state[field]) == list:
                    sz = len(client_state[field])
                    i = 0
                    while i < sz:
                        if torch.is_tensor(client_state[field][i]):
                            size_in_mbytes = client_state[field][i].numel() * client_state[field][i].element_size() / (
                                    1024.0 ** 2)
                            if size_in_mbytes >= tensor_prune_mb_threshold:
                                del client_state[field][i]
                                sz = len(client_state[field])
                            else:
                                client_state[field][i] = client_state[field][i].cpu()
                                i += 1
                        else:
                            i += 1

    # Remove reference for execution context
    del H["execution_context"]
    if 'exec_ctx' in H["args"]:
        del H["args"]

    # ==================================================================================================================
    # Cleanup execution context resources
    # ==================================================================================================================
    execution_context.resetExecutionContext(exec_ctx)
    # ==================================================================================================================
    if execution_context.simulation_finish_fn is not None:
        execution_context.simulation_finish_fn(H)


def runSimulation(cmdline, extra_=None):
    # ==================================================================================================================
    # ENTRY POINT FOR LAUNCH SIMULATION FROM GUI
    # ==================================================================================================================
    global CUDA_SUPPORT

    args = parse_args(cmdline)
    if args.hostname == "":
        args.hostname = socket.gethostname()

    Logger.setup_logging(args.loglevel, args.logfilter, logfile=args.logfile)
    logger = Logger.get(args.run_id)

    logger.info(f"CLI args original: {cmdline}")

    if torch.cuda.device_count():
        CUDA_SUPPORT = True
        # torch.backends.cudnn.benchmark = True
    else:
        logger.warning('CUDA unsupported!!')
        CUDA_SUPPORT = False

    if not CUDA_SUPPORT:
        args.gpu = "cpu"

    if args.deterministic:
        import torch.backends.cudnn as cudnn
        import os
        import random

        # TODO: This settings are not thread safe in case of using cuda backend from several threads
        # Project use execution context random generators for random and numpy in thread safe way
        if CUDA_SUPPORT:
            cudnn.deterministic = args.deterministic
            cudnn.benchmark = not args.deterministic
            torch.cuda.manual_seed(args.manual_init_seed)
            torch.cuda.manual_seed_all(args.manual_init_seed)

            # Turn off Tensor Cores if have been requested
            torch.backends.cudnn.allow_tf32 = args.allow_use_nv_tensorcores
            torch.backends.cuda.matmul.allow_tf32 = args.allow_use_nv_tensorcores

        torch.manual_seed(args.manual_init_seed)
        random.seed(args.manual_init_seed)
        np.random.seed(args.manual_init_seed)

        os.environ['PYTHONHASHSEED'] = str(args.manual_init_seed)
        torch.backends.cudnn.deterministic = True

    main(args, cmdline, extra_)


# ======================================================================================================================
g_Terminate = False


# ======================================================================================================================
def terminationWithSignal(*args):
    global g_Terminate
    g_Terminate = True
    print("Request for terminate process has been obtained. Save results and terminate.")


def isSimulationNeedEarlyStopCmdLine(H):
    global g_Terminate
    return g_Terminate


# ======================================================================================================================

def saveResult(H):
    """Serialize server state into filesystem"""
    args = H["args"]

    job_id = args.run_id
    fname = args.out
    if fname == "":
        fname = os.path.join(args.checkpoint_dir, f"{args.algorithm}_simulation_history.bin")

    # Save formally list of tuples (job_id, server_state)
    with open(fname, "wb") as f:
        pickle.dump([(job_id, H)], f)

    print(f"Final server state for {job_id} is serialized into: '{fname}'")
    print(f"Program which uses '{args.algorithm}' algorithm for {args.rounds} rounds was finished")


class ClientThread(threading.Thread):
    def __init__(self, clientSocket, listenPort):
        threading.Thread.__init__(self)
        self.clientSocket = clientSocket  # OS-like socket to communicate with information
        self.comSocket = comm_socket.CommSocket(clientSocket)  # Light construction on top of socket to communicate
        self.listenPort = listenPort  # Listening port

    def run(self):
        cmd = self.comSocket.rawRecvString()  # Obtain command

        t = time.localtime()
        cur_time = time.strftime("%H:%M:%S", t)
        print(f"-- {cur_time}: The received command: {cmd}")

        if cmd == "list_of_gpus":  # List of resources in the system
            gpus = len(gpu_utils.get_available_gpus())
            self.comSocket.rawSendString(f"{gpus}")
        elif cmd == "execute_work":  # Execute work
            devConfig = self.comSocket.rawRecvString()  # 1-st param: device configuration
            cmdLine = self.comSocket.rawRecv()  # 2-nd param: command line
            cmdLine = pickle.loads(cmdLine)  # unpack cmdline

            ip, port, dev_name, dev_number = devConfig.split(":")

            # Remove unnecessary flags for worker
            i = len(cmdLine) - 1
            while i >= 0:
                if cmdLine[i] == "--metric":
                    cmdLine[i + 1] = "none"

                if cmdLine[i] == "--gpu":
                    cmdLine[i + 1] = dev_number  # update device number

                if cmdLine[i] in ['--wandb-key', '--eval-async-threads', '--save-async-threads',
                                  '--threadpool-for-local-opt', '--external-devices', '--evaluate']:
                    del cmdLine[i]  # remove flag
                    del cmdLine[i]  # remove value for flag (originally at index i+1)
                i -= 1

            # Specify listen mode in which worker is
            cmdLine.append("--worker-listen-mode")
            cmdLine.append(str(self.listenPort))

            runSimulation(cmdLine, extra_=self.comSocket)
        else:
            print(f"The received command {cmd} is not valid")


def executeRemoteWorkerSupportMode(listenPort):
    """
    Listen for a specified port and wait for incoming connection. Maximum number of pending connection is 5.
    """
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('0.0.0.0', listenPort))
    MAX_CONNECTIONS = 5
    serversocket.listen(MAX_CONNECTIONS)

    while True:
        t = time.localtime()
        cur_time = time.strftime("%H:%M:%S", t)
        print(f"-- {cur_time}: Waiting for commands from the master by worker {socket.gethostname()}:{listenPort}")
        # accept connections from outside and process it in separate threads
        (clientsocket, address) = serversocket.accept()
        ct = ClientThread(clientsocket, listenPort)
        ct.run()


if __name__ == "__main__":
    # ==================================================================================================================
    # ENTRY POINT FOR CUI
    # ==================================================================================================================
    # Worker listener mode
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--worker-listen-mode":
            print(get_pretty_env_info())
            print("")
            executeRemoteWorkerSupportMode(int(sys.argv[i + 1]))
            sys.exit(0)
    # ==================================================================================================================
    # Usual mode

    # Setup signal for CTRL+C(SIGINT) or SIGTERM
    signal.signal(signal.SIGINT, terminationWithSignal)
    signal.signal(signal.SIGTERM, terminationWithSignal)

    execution_context.is_simulation_need_earlystop_fn = isSimulationNeedEarlyStopCmdLine
    execution_context.simulation_finish_fn = saveResult
    runSimulation(sys.argv[1:])
    sys.exit(0)
    # ==================================================================================================================
