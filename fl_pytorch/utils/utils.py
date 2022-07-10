#!/usr/bin/env python3

import numpy as np
import os
import glob
import json


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else None

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg


def init_metrics_meter(round):
    metrics_meter = {
        'round': round,
        'loss': AverageMeter(),
        'top_1_acc': AverageMeter(),
        'top_5_acc': AverageMeter(),
        'neq_perplexity': AverageMeter(),
    }
    return metrics_meter


def neg_perplexity_from_loss(loss):
    return - np.exp(loss)


def get_model_str_from_obj(model):
    return str(list(model.modules())[0]).split("\n")[0][:-1]


def create_metrics_dict(metrics):
    metrics_dict = {}

    for k in metrics:
        if k == 'round' or k == 'regression':
            metrics_dict[k] = metrics[k]
            continue
        metrics_dict[k] = metrics[k].get_avg()
    return metrics_dict


def create_model_dir(args, lr=True):
    model_dataset = '_'.join([args.model, args.dataset])
    run_id = f'id={args.run_id}'
    model_dir = os.path.join(args.checkpoint_dir, model_dataset, run_id)
    if lr:
        run_hp = os.path.join(f"lr=l_{str(args.local_lr)}_g_{str(args.global_lr)}",
                              f"seed_init={str(args.manual_init_seed)}",
                              f"seed_runtime={str(args.manual_runtime_seed)}"
                              )
        model_dir = os.path.join(model_dir, run_hp)

    return model_dir


# TODO: Update, to work for dual master and client stepsize (applies to 2 functions below)
def get_best_lr_and_metric(args, last=True):
    best_lookup = np.argmin if args.metric in ['loss'] else np.argmax
    model_dir_no_lr = create_model_dir(args, lr=False)
    lr_dirs = [lr_dir for lr_dir in os.listdir(model_dir_no_lr)
               if os.path.isdir(os.path.join(model_dir_no_lr, lr_dir))
               and not lr_dir.startswith('.')]

    lrs = np.array([float(lr_dir.split('=')[-1]) for lr_dir in lr_dirs])

    best_runs_metric = list()
    best_runs_dir = list()
    for lr_dir in lr_dirs:
        json_dir = 'best_metrics.json' if last else 'best_metrics.json'
        lr_metric_dirs = glob.glob(model_dir_no_lr + '/' + lr_dir + '/*/' + json_dir)
        lr_metric = list()
        for lr_metric_dir in lr_metric_dirs:
            with open(lr_metric_dir) as json_file:
                metric = json.load(json_file)
            lr_metric.append(metric[get_metric_key_name(args.metric)])
        i_best_run = best_lookup(lr_metric)
        best_runs_metric.append(lr_metric[i_best_run])
        best_runs_dir.append(lr_metric_dirs[i_best_run])

    i_best_lr = best_lookup(best_runs_metric)
    best_metric = best_runs_metric[i_best_lr]
    best_lr = lrs[i_best_lr]
    return best_lr, best_metric, np.sort(lrs)


def get_best_runs(args_exp, metric_name, last=True):
    model_dir_no_lr = create_model_dir(args_exp, lr=False)
    best_lr, _, _ = get_best_lr_and_metric(args_exp, last=last)
    model_dir_lr = os.path.join(model_dir_no_lr, f"lr={str(best_lr)}")
    json_dir = 'last_metrics.json' if last else 'best_metrics.json'
    metric_dirs = glob.glob(model_dir_lr + '/*/' + json_dir)

    with open(metric_dirs[0]) as json_file:
        metric = json.load(json_file)
    x = {
        'eval_p': metric['eval_p'],
        'params_p': metric['params_p'],
        'macs_p': metric['macs_p']
    }
    runs = [metric[metric_name]]

    for metric_dir in metric_dirs[1:]:
        with open(metric_dir) as json_file:
            metric = json.load(json_file)
        # ignores failed runs
        #if not np.isnan(metric['avg_loss']):
        runs.append(metric[metric_name])

    return x, runs


def log_epoch_info(logger, i, loader, metrics_meter, dataload_duration,
                   inference_duration, backprop_duration, train=True):
    mode_str = 'Train' if train else 'Test'
    msg = "{mode_str} [{round}][{current_batch}/{total_batches}]\t" \
          "DataLoad time {dataload_duration:.3f}\t" \
          "F/W time {inference_duration:.3f}\t" \
          "B/W time {backprop_duration:.3f}\t" "Loss {loss:.4f}\t" \
          "Prec@1 {prec1:.3f}\t" "Prec@5 {prec5:.3f}\t".format(
                mode_str=mode_str,
                round=metrics_meter['round'],
                current_batch=i,
                total_batches=len(loader),
                dataload_duration=dataload_duration,
                inference_duration=inference_duration,
                backprop_duration=backprop_duration,
                loss=metrics_meter['loss'].get_avg(),
                prec1=metrics_meter['top_1_acc'].get_avg(),
                prec5=metrics_meter['top_5_acc'].get_avg())
    logger.info(msg)
