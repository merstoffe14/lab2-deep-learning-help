import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    # There are ways to do this in less lines of code but I write it like this for me, to understand every step involved.
    #difference between two tensors
    sub_tensor = torch.subtract(target, input_tensor)
    #element wise squaring of this difference tensor
    squared_sub_tensor = torch.square(sub_tensor)
    #summing all the elements in this squared difference tensor
    sum_squared_sub_tensor = torch.sum(squared_sub_tensor)
    #deviding this by amount of elements in tensor
    mse = sum_squared_sub_tensor/torch.numel(target)

    return mse