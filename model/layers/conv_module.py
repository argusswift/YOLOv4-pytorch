import math
import torch
import numpy as np
import torch.nn as nn
from .activate import *
from torch.nn import init
from itertools import repeat
import torch.nn.functional as F
from torch._six import container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import config.yolov4_config as cfg

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {"relu": nn.ReLU, "leaky": nn.LeakyReLU, "mish": Mish}


class Convolutional(nn.Module):
    def __init__(
        self,
        filters_in,
        filters_out,
        kernel_size,
        stride,
        pad,
        norm=None,
        activate=None,
    ):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        if cfg.CONV_TYPE["TYPE"] == "DO_CONV":
            self.__conv = DOConv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=not norm,
            )
        else:
            self.__conv = nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=not norm,
            )
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](
                    negative_slope=0.1, inplace=True
                )
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x


class DOConv2d(Module):
    """
    DOConv2d can be used as an alternative for torch.nn.Conv2d.
    The interface is similar to that of Conv2d, with one exception:
         1. D_mul: the depth multiplier for the over-parameterization.
    Note that the groups parameter switchs between DO-Conv (groups=1),
    DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "D_mul",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        D_mul=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(
            x for x in self.padding for _ in range(2)
        )

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.D_mul)
        )
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros(
                [in_channels, M * N, self.D_mul], dtype=np.float32
            )
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(
                torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N)
            )
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(
                    torch.cat([D_diag, zeros], dim=2), requires_grad=False
                )
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def _conv_forward(self, input, weight):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.D_diag
            W = torch.reshape(
                self.W,
                (
                    self.out_channels // self.groups,
                    self.in_channels,
                    self.D_mul,
                ),
            )

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum("ims,ois->oim", D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M * N
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)
