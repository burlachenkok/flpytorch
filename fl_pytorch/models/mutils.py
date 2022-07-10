#!/usr/bin/env python3

# Import PyTorch root package
import torch                        

# Import PyTorch layers, activations and more
# import torch.nn.functional as F

from utils.logger import Logger


def torch_layers_info(model: torch.nn.Module):
    """
        Args:
            model (nn.Module): Neural Network model

        Returns:
            dict: Statistics about used standard torch modules inside. key: <module name> value: <used number> 
    """
    max_string_length = 0
    basic_modules = {}

    for module in model.modules():
        class_name = str(type(module)).replace("class ", "").replace("<'", "").replace("'>", "")

        # Skip Sequential models
        if class_name.find("torch.nn.modules.container.Sequential") == 0:
            continue

        max_string_length = max(max_string_length, len(class_name))
        if class_name not in basic_modules:
            basic_modules[class_name] = dict()
            basic_modules[class_name]["count"] = 0
            basic_modules[class_name]["parameters_size_bytes_cpu_train"] = 0
            basic_modules[class_name]["parameters_size_bytes_gpu_train"] = 0
            basic_modules[class_name]["parameters_size_bytes_cpu_frozen"] = 0
            basic_modules[class_name]["parameters_size_bytes_gpu_frozen"] = 0

            basic_modules[class_name]["parameters_size_numel_cpu_train"] = 0
            basic_modules[class_name]["parameters_size_numel_gpu_train"] = 0
            basic_modules[class_name]["parameters_size_numel_cpu_frozen"] = 0
            basic_modules[class_name]["parameters_size_numel_gpu_frozen"] = 0

        basic_modules[class_name]["count"] += 1
        size_in_bytes_cpu_train = 0
        size_in_bytes_gpu_train = 0
        size_in_bytes_cpu_frozen = 0
        size_in_bytes_gpu_frozen = 0

        size_in_numel_cpu_train = 0
        size_in_numel_gpu_train = 0
        size_in_numel_cpu_frozen = 0
        size_in_numel_gpu_frozen = 0

        for param in module.parameters():
            size_in_bytes = param.numel() * param.element_size()
            size_in_elements = param.numel()

            if param.device.type == "cpu":
                if param.requires_grad:
                    size_in_bytes_cpu_train += size_in_bytes
                    size_in_numel_cpu_train += size_in_elements
                else:
                    size_in_bytes_cpu_frozen += size_in_bytes
                    size_in_numel_cpu_frozen += size_in_elements

            else:
                if param.requires_grad:
                    size_in_bytes_gpu_train += size_in_bytes
                    size_in_numel_gpu_train += size_in_elements
                else:
                    size_in_bytes_gpu_frozen += size_in_bytes
                    size_in_numel_gpu_frozen += size_in_elements

        basic_modules[class_name]["parameters_size_bytes_cpu_train"] += size_in_bytes_cpu_train
        basic_modules[class_name]["parameters_size_bytes_gpu_train"] += size_in_bytes_gpu_train
        basic_modules[class_name]["parameters_size_bytes_cpu_frozen"] += size_in_bytes_cpu_frozen
        basic_modules[class_name]["parameters_size_bytes_gpu_frozen"] += size_in_bytes_gpu_frozen

        basic_modules[class_name]["parameters_size_numel_cpu_train"] += size_in_numel_cpu_train
        basic_modules[class_name]["parameters_size_numel_gpu_train"] += size_in_numel_gpu_train
        basic_modules[class_name]["parameters_size_numel_cpu_frozen"] += size_in_numel_cpu_frozen
        basic_modules[class_name]["parameters_size_numel_gpu_frozen"] += size_in_numel_gpu_frozen

    return basic_modules


def print_current_gpu_context(device, args):
    """Print current stream and blas handle for specific device"""
    if device == "cpu":
        return

    logger = Logger.get(args.run_id)
    current_stream = torch.cuda.current_stream(device)
    blas_handle = torch.cuda.current_blas_handle()
    logger.info(f"Current Steam: {current_stream}, BLAS handle: {hex(blas_handle)}")


def print_models_info(model: torch.nn.Module, args):
    """
        Args:
            model (nn.Module): Neural Network model
            args: Command line arguments

        Returns:
            None. All information is printed into stdout
    """
    logger = Logger.get(args.run_id)

    logger.info("----------------- Information about the model start ---------------------------------------------")
    logger.info('{0:44s} | {1:3s} | {2:s} | {3:s}'.format("Name",
                                                          "Layers",
                                                          "Learnable Parameters(Frozen)",
                                                          "Learnable Parameters(Train)"))
    logger.info("-------------------------------------------------------------------------------------------------")

    layers_info = torch_layers_info(model)
    for layer, info in layers_info.items():
        logger.info(f'{layer:44s} | {info["count"]:6d} | {( (info["parameters_size_bytes_cpu_frozen"] + info["parameters_size_bytes_gpu_frozen"])/1024.0):8g} KBytes / {(info["parameters_size_numel_cpu_frozen"] + info["parameters_size_numel_gpu_frozen"])} elements'
                    +
                    f'| {((info["parameters_size_bytes_cpu_train"] + info["parameters_size_bytes_gpu_train"])/1024.0):8g} KBytes / {(info["parameters_size_numel_cpu_train"] + info["parameters_size_numel_gpu_train"])} elements')

    logger.info("-------------------------------------------------------------------------------------------------")
    logger.info("    Model class:" + str(type(model)).replace("class ", "").replace("<'", "").replace("'>", ""))
    logger.info("-------------------------------------------------------------------------------------------------")


def number_of_params(model: torch.nn.Module, skipFrozen: bool = True) -> int:
    """
        Args:
            model (torch.nn.Module): Neural Network model

        Returns:
            integer: number of scalar parameters in the network to learn
    """
    total_number_of_scalar_parameters = 0
    for p in model.parameters():
        if skipFrozen and not p.requires_grad:
            continue

        total_items_in_param = 1
        for i in range(p.dim()):
            total_items_in_param = total_items_in_param * p.size(i)
        total_number_of_scalar_parameters += total_items_in_param
    return total_number_of_scalar_parameters


def set_params_to_zero(model, skipFrozen: bool = True, param_predicate=None):
    """
    Setup all model parameter to zero without tracking by autograd.

    This setup process is not tracking by autograd.

    Args:
        model (torch.nn.Module): Neural Network model
        param_predicate(function(i,param)): If none this function is used to understand should be setup this parameter or not
    """
    with torch.no_grad():
        if param_predicate is None:
            for p in model.parameters():

                if skipFrozen and not p.requires_grad:
                    continue

                p.zero_()
        else:
            for i, p in enumerate(model.parameters()):

                if skipFrozen and not p.requires_grad:
                    continue

                if param_predicate(i, p):
                    p.zero_()


# Currently used. Once it will be, please be carefully with random generators states
def set_params_uniform_random(model, a=0.0, b=1.0, skipFrozen: bool = True, param_predicate=None):
    """
    Setup all model parameter independently uniformly at random U(a,b).

    This setup process is not tracking by autograd.

    Args:
        model (torch.nn.Module): Neural Network model
        a(float): 'a' parameter of distribution
        b(float): 'b' parameter of distribution
        param_predicate(function(i,param)): If none this function is used to understand should be setup this parameter or not
    """
    with torch.no_grad():
        if param_predicate is None:
            for p in model.parameters():
                if skipFrozen and not p.requires_grad:
                    continue
                p[:] = a + (b-a) * torch.rand_like(p)
        else:
            for i, p in enumerate(model.parameters()):
                if skipFrozen and not p.requires_grad:
                    continue

                if param_predicate(i, p) == True:
                    p[:] = a + (b-a) * torch.rand_like(p)


def get_buffers(model: torch.nn.Module):
    local_model_buffers = list()
    for buf in model.buffers(): 
        local_model_buffers.append(buf.detach().clone())
    return local_model_buffers


def set_buffers(model: torch.nn.Module, buffer_list: list):
    with torch.no_grad():
        local_model_buffers = list()
        for index, buf in enumerate(model.buffers()):
            buf.flatten(0)[:] = buffer_list[index].flatten(0)[:]


def get_params(model: torch.nn.Module, skipFrozen: bool=True, param_predicate=None):
    """
    Get all model parameters as a single dense vector.

    Get all model parameters as a single dense vector, 
    if you are interesting only on a subset of parameter use param_predicate.

    Args:
        model (torch.nn.Module): Neural Network model
        param_predicate(function(i,param)): If none this function is used to understand 
                                            should be setup this parameter or not

    Returns:
        torch.Tensor: all parameters in a form of a tensor
    """
    params = []

    # For optimization do not perform copy on each tensor, intead of it make light copy of data. 
    # torch.cat(...) will produce new tensors

    if param_predicate is None:
        for p in model.parameters():
            if skipFrozen and not p.requires_grad:
                continue

            params.append(p.flatten(0).detach())      # Remove clone()
    else:
        for i, p in enumerate(model.parameters()):
            if skipFrozen and not p.requires_grad:
                continue

            if param_predicate(i, p) == True:
                params.append(p.flatten(0).detach())  # Remove clone()

    # Concatenates tensors along dim=0
    params_vector = torch.cat(tuple(params))

    return params_vector


def set_params(model: torch.nn.Module, parameters, skipFrozen: bool = True, param_predicate=None):
    """
    Set model parameters from a single dense vector.

    Set all model parameters from a single dense vector, 
    if you are interesting only on a subset of parameter use param_predicate. 
    This setup process is not tracking by autograd.

    Args:
        model (torch.nn.Module): Neural Network model
        parameters(torch.Tensor): Dense vector with parameters
        param_predicate(function(i,param)): If none this function is used to understand should be setup this parameter or not
    """
    with torch.no_grad():
        offset = 0
        if param_predicate is None:
            for i, p in enumerate(model.parameters()):
                if skipFrozen and not p.requires_grad:
                    continue

                sz = p.numel()
                p.flatten(0)[:] = parameters[offset:(offset+sz)]
                offset += sz
        else:
            for i, p in enumerate(model.parameters()):
                if skipFrozen and not p.requires_grad:
                    continue

                if param_predicate(i, p) == True:
                    sz = p.numel()
                    p.flatten(0)[:] = parameters[offset:(offset+sz)]
                    offset += sz


def get_gradient(model: torch.nn.Module, skipFrozen: bool = True):
    """
    Get all model gradient data as a single dense vector.

    Args:
        model (torch.nn.Module): Neural Network model

    Returns:
        torch.Tensor: all parameters in a form of a tensor
    """
    grads = []
    for p in model.parameters():
        if skipFrozen and not p.requires_grad:
            continue

        if p.grad is not None:
            grads.append(p.grad.flatten(0).detach())     # Remove clone()
        else:
            grads.append(torch.zeros_like(p).flatten(0))

    # Concatenates tensors along dim = 0
    grad_vec = torch.cat(tuple(grads))

    return grad_vec


def get_zero_gradient_compatible_with_model(model: torch.nn.Module, skipFrozen: bool = True):
    """
    Get zero vector with shape compatible with gradient data in a form of a single dense vector.

    Args:
        model (torch.nn.Module): Neural Network model

    Returns:
        torch.Tensor: zero vector with compatible shape
    """
    grads = []
    for p in model.parameters():
        if skipFrozen and not p.requires_grad:
            continue

        grads.append(torch.zeros_like(p).flatten(0))

    grad_vec = torch.cat(tuple(grads))
    return grad_vec


def add_to_gradient(model: torch.nn.Module, extra_grad, skipFrozen: bool = True):
    """
    Add to model gradient extra vector

    Args:
        model (nn.Module): Neural Network model
        extra_grad (torch.Tensor): Dense vector with gradients for all components
    """
    with torch.no_grad():
        offset = 0
        for i, p in enumerate(model.parameters()):

            if skipFrozen and not p.requires_grad:
                continue

            if p.grad is None:
                p.grad = torch.zeros_like(p)

            sz = p.grad.numel()
            p.grad.flatten(0)[:] += extra_grad[offset:(offset+sz)]
            offset += sz


def set_gradient(model: torch.nn.Module, grad, skipFrozen: bool = True):
    """
    Set model gradient

    Args:
        model (nn.Module): Neural Network model
        grad (torch.Tensor): Dense vector with gradients for all components
    """
    with torch.no_grad():
        offset = 0
        for i, p in enumerate(model.parameters()):
            if skipFrozen and not p.requires_grad:
                continue

            if p.grad is None:
                p.grad = torch.empty_like(p)

            sz = p.grad.numel()
            p.grad.flatten(0)[:] = grad[offset:(offset+sz)]
            offset += sz


def l2_norm_of_gradient_m(model: torch.nn.Module, skipFrozen: bool = True):
    value = 0.0
    for p in model.parameters():
        if skipFrozen and not p.requires_grad:
            continue

        if p.grad is not None:
            value += (p.grad**2).sum().item()
    return value**0.5


def l2_norm_of_vec(grad):
    return ((grad**2).sum().item())**0.5


def turn_off_batch_normalization_and_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d) or \
           isinstance(m, torch.nn.BatchNorm2d) or \
           isinstance(m, torch.nn.BatchNorm3d) or \
           isinstance(m, torch.nn.Dropout):
            m.eval()
# ======================================================================================================================
# Unittests for launch please use: "pytest -v mutils.py" 
# https://docs.pytest.org/en/stable/getting-started.html


def test_get_set_for_model():
    hidden_features_layer_1 = 20
    hidden_features_layer_2 = 22
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=hidden_features_layer_1, bias=False),
        torch.nn.Linear(in_features=hidden_features_layer_1, out_features=hidden_features_layer_2, bias=False),
        torch.nn.Linear(in_features=hidden_features_layer_2, out_features=1, bias=False),
    )

    assert number_of_params(model) == hidden_features_layer_1 + \
                                      hidden_features_layer_1 * hidden_features_layer_2 + \
                                      hidden_features_layer_2
    
    class Empty: 
        pass
    
    args = Empty()
    args.run_id = "test"
    Logger.setup_logging()

    print_models_info(model, args)

    model.train(True)
    for p in model.parameters():
        assert p.grad is None

    z = (10.0 - model(torch.Tensor([[3]])))**2
    z.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape

    g = get_gradient(model)
    assert abs(l2_norm_of_gradient_m(model) - l2_norm_of_vec(g)) < 1.0e-4
    assert g.numel() == number_of_params(model)
    set_gradient(model, 2.0 * g)

    assert abs(l2_norm_of_gradient_m(model) - 2.0 * l2_norm_of_vec(g)) < 1.0e-4


def test_grad_addition_for_model():
    hidden_features_layer_1 = 20
    hidden_features_layer_2 = 22
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=hidden_features_layer_1, bias=False),
        torch.nn.Linear(in_features=hidden_features_layer_1, out_features=hidden_features_layer_2, bias=True),
        torch.nn.Linear(in_features=hidden_features_layer_2, out_features=1, bias=True),
    )
    n1 = number_of_params(model, skipFrozen=False)
    model[0].requires_grad_(False)
    n2 = number_of_params(model, skipFrozen=False)
    assert n1 == n2
    z = (10.0 - model(torch.Tensor([[3]])))**2
    z.backward()
    g = get_gradient(model, skipFrozen=False)
    # Verify that requires_grad_(False) is correctly working with get_gradient() functionality
    assert l2_norm_of_vec(g[0:hidden_features_layer_1]) < 1.0e-4

    # Verify that gradient has correct number of items
    assert g.numel() == 1*hidden_features_layer_1 + \
                        hidden_features_layer_1 * hidden_features_layer_2 + \
                        hidden_features_layer_2 * 1 + \
                        hidden_features_layer_2 + 1
    
    assert g.dim() == 1
    get_zero_gradient_compatible_with_model(model, skipFrozen=False)

    assert abs(l2_norm_of_gradient_m(model, skipFrozen=False) - l2_norm_of_vec(g)) < 1.0e-4
    set_gradient(model, g, skipFrozen=False)
    add_to_gradient(model, torch.ones(n1), skipFrozen=False)
    g1 = get_gradient(model, skipFrozen=False)
    assert g1[0].item() == 1.0
    add_to_gradient(model, -torch.ones(n1),skipFrozen=False)
    g2 = get_gradient(model,skipFrozen=False)
    assert l2_norm_of_vec(g2 - g) < 1.0e-5
    assert l2_norm_of_vec(g2 - g1) > 1.0e-5

    assert len(torch_layers_info(model)) == 1

def test_get_set_params_and_grad_inf():
    hidden_features_layer_1 = 20
    hidden_features_layer_2 = 22
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=hidden_features_layer_1, bias=False),
        torch.nn.Linear(in_features=hidden_features_layer_1, out_features=hidden_features_layer_2, bias=True),
        torch.nn.Linear(in_features=hidden_features_layer_2, out_features=1, bias=True),
    )
    model[0].requires_grad_(False)
    z = (10.0 - model(torch.Tensor([[3]])))**2
    z.backward()
    g1 = get_gradient(model, skipFrozen=False)
    add_to_gradient(model, g1, skipFrozen=False)
    g2 = get_gradient(model, skipFrozen=False)
    assert l2_norm_of_vec(g2 - 2*g1) < 1.0e-5
    
    g3 = get_zero_gradient_compatible_with_model(model, skipFrozen=False)
    assert l2_norm_of_vec(g3) < 1.0e-5
    assert g3.size() == g2.size()
    set_params_to_zero(model, skipFrozen=False)
    assert l2_norm_of_vec(get_params(model, skipFrozen=False)) < 1.0e-5
    set_params_uniform_random(model, skipFrozen=False)
    assert l2_norm_of_vec(get_params(model, skipFrozen=False)) > 1.0
    set_params_to_zero(model, skipFrozen=False)
    z = (10.0 - model(torch.Tensor([[0]])))**2
    z.backward()
    assert z.item() == 100.0

# ======================================================================================================================
