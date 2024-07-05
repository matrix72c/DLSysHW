from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_z = array_api.max(Z, self.axes, keepdims=True)
        max_zn = array_api.max(Z, self.axes, keepdims=False)
        Z = array_api.log(array_api.sum(array_api.exp(Z - max_z), self.axes)) + max_zn
        return Z

    def gradient(self, out_grad, node):
        Z = node.inputs[0].cached_data
        max_z = array_api.max(Z, self.axes, keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, self.axes)

        log_grad = out_grad.realize_cached_data() / sum_exp
        broadcast_shape = list(Z.shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                broadcast_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    broadcast_shape[axis] = 1
        else:
            broadcast_shape = [1 for _ in range(len(broadcast_shape))]
        grad = array_api.reshape(log_grad, tuple(broadcast_shape))
        grad = exp_z * grad
        return Tensor(grad)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

