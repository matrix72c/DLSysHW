from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

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
        self.mZ = Z.max(self.axes, keepdims=True)
        mZ = self.mZ.broadcast_to(Z.shape)
        Z_ = (Z - mZ).exp()
        Z_ = Z_.sum(self.axes)
        Z_ = Z_.log()
        return Z_ + self.mZ.reshape(Z_.shape)

    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        input_shape = inp.shape
        mZ = Tensor(self.mZ.broadcast_to(input_shape), device=inp.device)
        base_shape = list(input_shape)
        if isinstance(self.axes, int): self.axes = (self.axes,)
        axes = list(range(len(base_shape))) \
            if self.axes is None else self.axes
        for ax in axes:
            base_shape[ax] = 1
        out_grad = out_grad / summation(exp((inp - mZ)), self.axes)
        out_grad = out_grad.reshape(base_shape)
        out_grad = out_grad.broadcast_to(input_shape)
        out_grad = out_grad * exp(inp - mZ)
        return (out_grad,)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

