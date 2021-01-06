"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch
import torch.nn as nn
import math


__all__ = ["mobilenetv3"]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish(),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(
        self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
    ):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _MobileNetV3(nn.Module):
    def __init__(self, width_mult=1.0):
        super(_MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t, c, SE, HS, s
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                block(
                    input_channel,
                    exp_size,
                    output_channel,
                    k,
                    s,
                    use_se,
                    use_hs,
                )
            )
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers

        output_channel = (
            _make_divisible(1024 * width_mult, 8) if width_mult > 1.0 else 1024
        )
        self.conv = conv_1x1_bn(input_channel, output_channel)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        print("**" * 10, "Initing MobilenetV3 weights", "**" * 10)

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
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

                print("initing {}".format(m))


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "features":
                for f_name, f_module in module._modules.items():
                    x = f_module(x)
                    if f_name in self.extracted_layers:
                        outputs.append(x)
            if name is "conv":
                x = module(x)
                if name in self.extracted_layers:
                    outputs.append(x)
        return outputs


class MobilenetV3(nn.Module):
    def __init__(
        self,
        extract_list=["3", "8", "conv"],
        weight_path=None,
        resume=False,
        width_mult=1.0,
        feature_channels=[24, 48, 1024]
    ):
        super(MobilenetV3, self).__init__()
        self.feature_channels = feature_channels
        self.__submodule = _MobileNetV3(width_mult=width_mult)
        if weight_path and not resume:
            print(
                "*" * 40,
                "\nLoading weight of MobilenetV3 : {}".format(weight_path),
            )

            pretrained_dict = torch.load(
                weight_path, map_location=torch.device("cpu")
            )
            model_dict = self.__submodule.state_dict()
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                if "features" in k:
                    new_state_dict[k] = v
            # pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(new_state_dict)
            self.__submodule.load_state_dict(model_dict)
            del pretrained_dict

            print("Loaded weight of MobilenetV3 : {}".format(weight_path))

        self.__extractor = FeatureExtractor(self.__submodule, extract_list)

    def forward(self, x):
        return self.__extractor(x)


def _BuildMobilenetV3(weight_path, resume):
    model = MobilenetV3(weight_path=weight_path, resume=resume)

    return model, model.feature_channels[-3:]


if __name__ == "__main__":
    # path = 'mobilenetv3.pth'
    model = MobilenetV3(weight_path=None)
    print(model)
    in_img = torch.randn(2, 3, 224, 224)
    p = model(in_img)

    for i in range(3):
        print(p[i].shape)
