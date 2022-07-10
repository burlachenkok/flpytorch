#!/usr/bin/env python3

# Import PyTorch root package import torch
import torch

from . import logger


def is_target_dev_gpu(device):
    """ Check that target device is gpu.

    Args:
        device: integer or string. If it's integer -1 stands for CPU, and value greater then or equal to
        0 is a GPU number in the system. If it's a string then it's a string in device PyTorch format.

    Returns:
        True if device is a specification of GPU device, and False otherwise.
    """
    if type(device) is int:
        return device >= 0

    if device.find(":") == -1:
        return device.lower() == "cuda"
    else:
        return device.split(":")[0].lower() == "cuda"


def get_target_dev_number(device):
    """ Get target device number.

    Args:
        device: integer or string in format <device_type:index> or <device>. For the last case device index will be zero

    Returns:
        Integer with device index.
    """
    if type(device) is int:
        return device

    if device.find(":") == -1:
        return 0
    else:
        return int(device.split(":")[1])


def get_target_device_str(device):
    """Get string for target device

    Args:
        device: device specification

    Returns:
        Explicit PyTorch device string to specify device
    """
    if is_target_dev_gpu(device):
        return f"cuda:{get_target_dev_number(device)}"
    else:
        return "cpu"


def get_available_gpus():
    """Get list of available gpus in the system.
    Returns:
        List of string with device properties
    """
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpus.append(torch.cuda.get_device_properties(i))
    return gpus


def print_gpu_usage(args):
    """Print info about current GPU usage into logger as information message"""
    log = logger.Logger.get(args.run_id)
    memory_gpu_used = torch.cuda.memory_stats(args.device)['reserved_bytes.all.current']
    log.info(f"GPU usage: We are using {memory_gpu_used/(1024.0**2):.2f} MB from device {args.device}")


def print_info_about_used_gpu(target_device, run_id):
    """Print info about GPU installed in the system into standard output"""
    log = logger.Logger.get(run_id)

    log.info("-------------------------------------------------------------------------------------------------")
    if not is_target_dev_gpu(target_device):
        gpus_properties = get_available_gpus()
        for i in range(len(gpus_properties)):
            log.info(" {0} {1:g} GBytes of GDDR".format(gpus_properties[i].name,
                                                        gpus_properties[i].total_memory/(1024.0**3)))
    else:
        gpu_id = get_target_dev_number(target_device)
        gpus_properties = get_available_gpus()
        for i in range(len(gpus_properties)):
            if i == gpu_id:
                log.info(" {0} {1:g} GBytes of GDDR *".format(gpus_properties[i].name,
                                                              gpus_properties[i].total_memory / (1024.0 ** 3)))
            else:
                log.info(" {0} {1:g} GBytes of GDDR".format(gpus_properties[i].name,
                                                            gpus_properties[i].total_memory / (1024.0 ** 3)))
    log.info("-------------------------------------------------------------------------------------------------")


# ======================================================================================================================
# Unittests for launch please use: "pytest -v gpu_utils.py" 
def test_device_naming():
    assert is_target_dev_gpu("cuda")
    assert not is_target_dev_gpu("cpu")
    assert is_target_dev_gpu("cuda:1")
    assert get_target_dev_number("cuda") == 0
    assert get_target_dev_number("cuda:1") == 1
    assert get_target_dev_number(2) == 2
    assert is_target_dev_gpu(0)
    assert is_target_dev_gpu(1)

    if torch.cuda.is_available():
        assert len(get_available_gpus()) == torch.cuda.device_count()

# ======================================================================================================================
