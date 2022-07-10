#!/usr/bin/env python3

import argparse
import time
from datetime import datetime
import os
import utils.gpu_utils as gpu_utils
import random

# Global simulation counter
gSimulationCounter = 0


def parse_args(args):
    parser = initialise_arg_parser(args, 'FLPyTorch, running arguments.')

    # SERVER OPTIMIZATION PARAMS
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning",
    )
    parser.add_argument(
        '--client-sampling-type',
        type=str,
        choices=['uniform', 'poisson', 'poisson-no-empty'],
        default='uniform',
        help='Sampling strategy to sample clients'
    )
    parser.add_argument(
        "--num-clients-per-round",
        type=int,
        default=0,
        help="Number of available clients used for one communication round",
    )
    parser.add_argument(
        "--client-sampling-poisson",
        type=float,
        default=0.0,
        help="Sampling probability for Poisson sampling." 
             "Coincident with probability for clients to be selected in communication round",
    )
    parser.add_argument(
        '--global-lr',
        type=float,
        default=1,
        help='Global initial local learning rate (default: 1)'
    )
    parser.add_argument(
        '--global-lr-type',
        type=str,
        choices=['cosine', 'cifar_1', 'cifar_2', 'static'],
        default='static',
        help='Global learning rate strategy (default: static)'
    )
    parser.add_argument(
        "--global-optimiser",
        type=str,
        choices=['sgd', 'adam', 'rmsprop'],
        default='sgd',
        help='Global optimiser to use (default: SGD)'
    )
    parser.add_argument(
        '--global-momentum',
        type=float,
        default=0.,
        help='Global momentum (default: 0.)'
    )
    parser.add_argument(
        '--global-weight-decay',
        type=float,
        default=0.,
        help='Global weight decay (default: 0.)'
    )

    # LOCAL OPTIMISATION PARAMETERS
    parser.add_argument(
        "--run-local-steps",
        action="store_true",
        default=False,
        help="Run local epochs or local iterations, "
             "if 'True' then each worker runs '--number-of-local-iters' steps in batches"
             "          else each worker runs '--number-of-local-iters' local epochs."
    )
    parser.add_argument(
        "-li", "--number-of-local-iters",
        type=int,
        default=None,
        help="Static number of local steps to run training for defined by the server training configuration"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Static batch size for data loading"
    )
    parser.add_argument(
        '--local-lr',
        type=float,
        default=0.1,
        help='initial local learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--local-lr-type',
        type=str,
        choices=['cosine', 'cifar_1', 'cifar_2', 'static'],
        default='static',
        help='Local learning rate strategy (default: static)'
    )
    parser.add_argument(
        "--local-optimiser",
        type=str,
        choices=['sgd', 'adam', 'rmsprop'],
        default='sgd',
        help='Local optimiser to use (default: SGD)'
    )
    parser.add_argument(
        '--local-momentum',
        type=float,
        default=0.0,
        help='Momentum (default: 0.0)'
    )
    parser.add_argument(
        '--local-weight-decay',
        type=float,
        default=0.0,
        help='Local weight decay (default: 1e-4)'
    )

    # MODEL and DATA PARAMETERS
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cifar10", "cifar100", "emnist", "full_shakespeare",
            "cifar10_fl", "cifar10_fl_by_class",
            "cifar100_fl", "shakespeare", "femnist",
            # Artificial problem
            "generated_for_quadratic_minimization",
            # For Logistic Regression
            "w9a", "w8a", "w7a", "w6a", "w5a", "w4a", "w3a", "w2a", "w1a",
            "a9a", "a8a", "a7a", "a6a", "a5a", "a4a", "a3a", "a2a", "a1a",
            "mushrooms", "phishing"
        ],
        help="Define which dataset to load"
    )
    parser.add_argument(
        "--dataset-generation-spec",
        type=str,
        default='',
        help="Specification when data is artificially generated"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='top_1_acc',
        choices=["top_1_acc", "top_5_acc", "neg_perplexity", "loss", "none"],
        help="Define which metric to optimize. None ignores validation step."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Define which model to load"
    )

    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        default=False,
        help="Define should be modeled be pretrained or not"
    )

    parser.add_argument(
        "--train-last-layer",
        action="store_true",
        default=False,
        help="Train only last linear layer"
    )

    parser.add_argument(
        "--turn-off-batch-normalization-and-dropout",
        action="store_true",
        default=False,
        help="During train time do not use Batch Normalization and Dropout if it is inside compute graph f(x;ai,bi)."
    )

    # SETUP ARGUMENTS
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default='../check_points',
        help="Directory to persist run meta data_preprocess, e.g. best/last models."
    )
    parser.add_argument(
        "--do-not-save-eval-checkpoints",
        action="store_true",
        default=False,
        help="Turn off save evaluate checkpoints."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from checkpoint."
    )
    parser.add_argument(
        "--load-best",
        default=False,
        action='store_true',
        help="Load best from checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default='fp32',
        choices=["bfp16", "fp16", "fp32", "fp64"],
        help="Define the type for trainable parameters and for buffers used e.g. in Batch Normalization."
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help="Define on which GPU to run the model (comma-separated for multiple). If -1, use CPU."
    )

    parser.add_argument(
        "--log-gpu-usage",
        default=False,
        action='store_true',
        help="Log GPU usage"
    )

    parser.add_argument(
        "-n", "--num-workers-train",
        type=int,
        default=0,
        help="Num workers for train dataset loading"
    )
    parser.add_argument(
        "-nt", "--num-workers-test",
        type=int,
        default=0,
        help="Num workers for test dataset loading"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically for reproducibility."
    )

    parser.add_argument(
        "--manual-init-seed",
        type=int,
        default=123,
        help="Random seed to use for model initialization and data generation"
    )

    parser.add_argument(
        "--manual-runtime-seed",
        type=int,
        default=123,
        help="Random seed to use during runtime mostly for stochastic optimization algorithms"
    )

    parser.add_argument(
        "--group-name",
        type=str,
        default="",
        help="Name of the group which allow group experiments for statistics"
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Extra arbitrarily comment during transferring experimental results"
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default="",
        help="Name of the machine. If empty string is passed the name is determinate automatically."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="How often to do validation."
    )
    parser.add_argument(
        "--eval-async-threads",
        type=int,
        default=0,
        help="Thread pool size for perform evaluation checkpoint asynchronously"
    )
    parser.add_argument(
        "--save-async-threads",
        type=int,
        default=0,
        help="Thread pool size for perform serialization into filesystem asynchronously"
    )
    parser.add_argument(
        "--threadpool-for-local-opt",
        type=int,
        default=0,
        help="Perform local training within a pool of threads." 
             "Each one execute sequence of optimization for assigned clients in a serialized way."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=str(time.time()),
        help="Identifier for the current job"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dcgd",
        help="Algorithm (marina, diana, dcgd, fedavg, scaffold, fedprox, ef21, cofig, frecon, ef21-pp, pp-marina)"
    )
    parser.add_argument(
        "--algorithm-options",
        type=str,
        default="",
        help="Extra option for pass into algorithm implementation to carry experiments (sgd:name-of-sgd)"
    )
    parser.add_argument(
        "--client-compressor",
        type=str,
        default="ident",
        help="Client compressor. ident, randk:k|<percentage-of-D>%, bernulli:p, natural, qsgd:levels, "
             "nat.dithering:levels:norm, std.dithering:levels:norm, topk:k|topk:<percentage-of-D>%),"
             "rank_k:k|rank_k:<percentage-of-D>%, terngrad"
    )
    parser.add_argument(
        "--extra-track",
        type=str,
        default="",
        help="Extra expensive comma separated tracking characterstics that should be collected"
             "(full_gradient_norm_train,full_objective_value_train,full_gradient_norm_val,full_objective_value_val)"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="crossentropy",
        help="Name of the use loss function (crossentropy, mse, logistic)"
    )
    parser.add_argument(
        "--global-regulizer",
        type=str,
        choices=["none", "noncvx_robust_linear_regression", "cvx_l2norm_square_div_2"],
        default="none",
        help="Name of the extra global regulizer (default none)"
    )
    parser.add_argument(
        "--global-regulizer-alpha",
        type=float,
        default=0.0,
        help="Global regulizer scalar multiple (default value 0.0)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Name of the file for serialize results into filesystem"
    )

    parser.add_argument(
        "--wandb-key",
        type=str,
        default='',
        help="Personal Wandb key to wandb.ai online plotting tool"
    )

    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='fl_pytorch_simulation',
        help="Project name for wandb.ai online plotting tool in which plots will be generated"
    )

    parser.add_argument(
        "--external-devices",
        type=str,
        default='',
        help="List of external devices ip:port:device that can be used during simulation process"
    )

    parser.add_argument(
        "--worker-listen-mode",
        type=int,
        default=-1,
        help="This flag activate worker in local listener mode. For local worker mode only this flag is needed"
    )

    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["debug", "info", "warn", "error", "critical"],
        default="INFO"
    )
    parser.add_argument(
        "--logfilter",
        type=str,
        default='.*',
        help="Regular expression to filter logging strings in which we're interesting in during debugging"
    )
    parser.add_argument(
        "--store-client-state-in-cpu",
        action="store_true",
        default=False,
        help="Store client state in CPU DRAM memory. Useful when number of clients is relatively big"
    )
    parser.add_argument(
        "--per-round-clean-torch-cache",
        action="store_true",
        default=False,
        help="Force clean GPU PyTorch cache at the end of each communication round used internally by PyTorch."
             "Good to fix fragmentation issues."
    )
    parser.add_argument(
        "--allow-use-nv-tensorcores",
        action="store_true",
        default=False,
        help="Allow to use NVIDIA Tensor Cores available from NVIDIA Ampere architecture"
    )
    parser.add_argument(
        "--initialize-shifts-policy",
        type=str,
        choices=["zero", "full_gradient_at_start"],
        default="zero",
        help="Policy for initial shifts for opt. algorithms that contains notion of shifts"
    )
    parser.add_argument(
        "--sort-dataset-by-class-before-split",
        action="store_true",
        default=False,
        help="Sort train and test dataset by class label"
    )

    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )

    # Evaluation mode, do not run training
    parser.add_argument(
        "--evaluate",
        action='store_true', 
        default=False, 
        help="Evaluation or Training mode"
    )

    # Parse cmdline arguments
    args_dict = {}
    i = 0
    while True:
        if i == len(args):
            # Terminating the loop
            break
        if args[i].find('--') == 0 and (i + 1 == len(args) or args[i + 1].find('--') == 0):
            # Flag
            key = args[i][2:]
            args_dict[key] = True
            i += 1
        else:
            # Key-value pair
            key = args[i][2:]
            value = args[i+1]
            args_dict[key] = value
            i += 2

    # Perform substitution
    global gSimulationCounter
    gSimulationCounter += 1

    args_dict["simcounter"] = gSimulationCounter     # Simulation counter
    args_dict["now"] = int(time.time())              # Time
    args_dict["rnd-salt"] = random.randint(0, 1024*1024)  # Any seeds are setting after parameters initialization

    # Make substitution into arguments
    for kNested in range(5):
        for i in range(len(args)):
            args[i] = str(args[i]).format(**args_dict).lower().strip()

    args = parser.parse_args(args)

    # Make args.gpu as a list of target devices
    transform_gpu_args(args)
    return args


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser


def transform_gpu_args(args):
    if args.gpu == "-1":
        args.gpu = ["cpu"]
    else:
        gpu_str_arg = args.gpu.split(',')
        if len(gpu_str_arg) > 1:
            # args.gpu = sorted([int(card) for card in gpu_str_arg])
            # Create a list of cuda(gpu) devices, followed at the end by cpu devices
            args.gpu = sorted([gpu_utils.get_target_device_str(int(card)) for card in gpu_str_arg if int(card) >= 0])
            args.gpu += sorted([gpu_utils.get_target_device_str(int(card)) for card in gpu_str_arg if int(card) < 0])
        else:
            args.gpu = [f"cuda:{args.gpu}"]
