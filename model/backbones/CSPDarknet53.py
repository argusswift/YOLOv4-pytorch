import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.layers.attention_layers import SEModule, CBAM
import config.yolov4_config as cfg


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "linear": nn.Identity(),
    "mish": Mish(),
}


class Convolutional(nn.Module):
    def __init__(
        self,
        filters_in,
        filters_out,
        kernel_size,
        stride=1,
        norm="bn",
        activate="mish",
    ):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        self.__conv = nn.Conv2d(
            in_channels=filters_in,
            out_channels=filters_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
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
            if activate == "mish":
                self.__activate = activate_name[activate]

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x


class CSPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        residual_activation="linear",
    ):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.block = nn.Sequential(
            Convolutional(in_channels, hidden_channels, 1),
            Convolutional(hidden_channels, out_channels, 3),
        )

        self.activation = activate_name[residual_activation]
        self.attention = cfg.ATTENTION["TYPE"]
        if self.attention == "SEnet":
            self.attention_module = SEModule(out_channels)
        elif self.attention == "CBAM":
            self.attention_module = CBAM(out_channels)
        else:
            self.attention = None

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.attention is not None:
            out = self.attention_module(out)
        out += residual
        return out


class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = Convolutional(
            in_channels, out_channels, 3, stride=2
        )

        self.split_conv0 = Convolutional(out_channels, out_channels, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            Convolutional(out_channels, out_channels, 1),
        )

        self.concat_conv = Convolutional(out_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x


class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = Convolutional(
            in_channels, out_channels, 3, stride=2
        )

        self.split_conv0 = Convolutional(out_channels, out_channels // 2, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels // 2, 1)

        self.blocks_conv = nn.Sequential(
            *[
                CSPBlock(out_channels // 2, out_channels // 2)
                for _ in range(num_blocks)
            ],
            Convolutional(out_channels // 2, out_channels // 2, 1)
        )

        self.concat_conv = Convolutional(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x


class CSPDarknet53(nn.Module):
    def __init__(
        self,
        stem_channels=32,
        feature_channels=[64, 128, 256, 512, 1024],
        num_features=3,
        weight_path=None,
        resume=False,
    ):
        super(CSPDarknet53, self).__init__()

        self.stem_conv = Convolutional(3, stem_channels, 3)

        self.stages = nn.ModuleList(
            [
                CSPFirstStage(stem_channels, feature_channels[0]),
                CSPStage(feature_channels[0], feature_channels[1], 2),
                CSPStage(feature_channels[1], feature_channels[2], 8),
                CSPStage(feature_channels[2], feature_channels[3], 8),
                CSPStage(feature_channels[3], feature_channels[4], 4),
            ]
        )

        self.feature_channels = feature_channels
        self.num_features = num_features

        if weight_path and not resume:
            self.load_CSPdarknet_weights(weight_path)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features[-self.num_features :]

    def _initialize_weights(self):
        print("**" * 10, "Initing CSPDarknet53 weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))

    def load_CSPdarknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                # if count == cutoff:
                #     break
                # count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias.data
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight.data
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight.data
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))


def _BuildCSPDarknet53(weight_path, resume):
    model = CSPDarknet53(weight_path=weight_path, resume=resume)

    return model, model.feature_channels[-3:]


if __name__ == "__main__":
    model = CSPDarknet53()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
