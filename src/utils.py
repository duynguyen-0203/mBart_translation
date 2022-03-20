import numpy as np
import random
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'


def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape
    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)
    max_shape = [max([tensor.shape[d] for tensor in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for tensor in tensors:
        e = extend_tensor(tensor, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)

    return stacked
