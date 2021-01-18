# Model memory occupancy monitoring function
# model：Input model
# input：Input tensor variables
# default type_size:  4
# default type: float32
import numpy as np
import torch.nn as nn


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print(
        "Model {} : params: {:4f}M".format(
            model._get_name(), para * type_size / 1000 / 1000
        )
    )

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)[0]
        if len(out):
            for j in range(len(out)):
                out_sizes.append(np.array(out[j].size()))
            break

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print(
        "Model {} : intermedite variables: {:3f} M (without backward)".format(
            model._get_name(), total_nums * type_size / 1000 / 1000
        )
    )
    print(
        "Model {} : intermedite variables: {:3f} M (with backward)".format(
            model._get_name(), total_nums * type_size * 2 / 1000 / 1000
        )
    )


if __name__ == "__main__":
    from model.build_model import Build_Model
    import torch

    net = Build_Model()
    print(net)

    in_img = torch.randn(1, 3, 320, 320)
    modelsize(net, in_img)
