"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad_with_penalty = param.grad.detach() + self.weight_decay * param.detach()
            u = self.u.get(id(param), 0) * self.momentum + (1 - self.momentum) * grad_with_penalty
            u = ndl.Tensor(u, dtype=param.dtype)
            self.u[id(param)] = u
            param.data -= self.lr * u

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            grad_with_penalty = param.grad.detach() + self.weight_decay * param.detach()
            grad_with_penalty = ndl.Tensor(grad_with_penalty, dtype=param.dtype) # Must convert dtype which cause memory check failed

            m = self.beta1 * self.m.get(id(param), 0) + (1 - self.beta1) * grad_with_penalty
            v = self.beta2 * self.v.get(id(param), 0) + (1 - self.beta2) * grad_with_penalty ** 2
            self.m[id(param)] = m.detach()
            self.v[id(param)] = v.detach()
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
