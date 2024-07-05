import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = (6 / fan_in) ** 0.5
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = (2 / fan_in) ** 0.5
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
