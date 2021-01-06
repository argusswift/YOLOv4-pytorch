import torch
from torch import nn
from mmcv.cnn import constant_init, kaiming_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):
    def __init__(self, inplanes, planes):
        super(ContextBlock2d, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode="fan_in")
        self.conv_mask.inited = True
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        beta1 = context_mask
        beta2 = torch.transpose(beta1, 1, 2)
        atten = torch.matmul(beta2, beta1)

        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context, atten

    def forward(self, x):
        # [N, C, 1, 1]
        context, atten = self.spatial_pool(x)
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term

        return out, atten
