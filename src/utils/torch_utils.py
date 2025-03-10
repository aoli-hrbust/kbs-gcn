# MIT License

# Copyright (c) 2023 Ao Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam


def get_device():
    return 0 if torch.cuda.is_available() else 'cpu'


def EPS_like(x: Tensor):
    """
    产生一个EPS数值，放在x相同的设备上。
    """
    return torch.tensor(1e-10, dtype=x.dtype, device=x.device)


def EPS_max(x: Tensor):
    """
    小于EPS的值统一设为EPS，提升数值稳定性。
    """
    return torch.max(x, EPS_like(x))


def convert_tensor(thing, dtype=torch.float, dev="cpu"):
    """
    Convert a np.ndarray or list of them to tensor.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_tensor(x, dtype, dev) for x in thing]
    elif isinstance(thing, dict):
        return {key: convert_tensor(val) for key, val in thing.items()}
    elif isinstance(thing, np.ndarray):
        return torch.tensor(thing, dtype=dtype, device=dev)
    elif isinstance(thing, torch.Tensor):
        return thing
    elif thing is None:
        return None
    else:
        raise ValueError(f"{type(thing)}")


def convert_numpy(thing):
    """
    Convert a tensor or list of them to numpy.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_numpy(x) for x in thing]
    elif isinstance(thing, dict):
        return {key: convert_numpy(val) for key, val in thing.items()}
    elif isinstance(thing, torch.Tensor):
        return thing.detach().cpu().numpy()
    else:
        return thing


def convert_cpu(thing):
    """
    Convert a tensor or list of them to numpy.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_cpu(x) for x in thing]
    elif isinstance(thing, dict):
        return {key: convert_cpu(val) for key, val in thing.items()}
    elif isinstance(thing, torch.Tensor):
        return thing.detach().cpu()
    else:
        return thing

# class DeviceRun:
#
#     def __init__(self, fn):
#         self.fn = fn
#         self.device = get_device()
#
#     def __call__(self, *args, **kwargs):
#         kwargs.update(device=self.device)
#         if self.device == 'cpu':
#             return self.fn(*args, **kwargs)
#         try:
#             return self.fn(*args, **kwargs)
#         except torch.cuda.CudaError
