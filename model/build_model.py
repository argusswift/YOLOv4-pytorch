import sys

sys.path.append("..")

import torch.nn as nn
import torch
from model.head.yolo_head import Yolo_head
from model.YOLOv4 import YOLOv4
import config.yolov4_config as cfg


class Build_Model(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """

    def __init__(self, weight_path=None, resume=False, showatt=False):
        super(Build_Model, self).__init__()
        self.__showatt = showatt
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        if cfg.TRAIN["DATA_TYPE"] == "VOC":
            self.__nC = cfg.VOC_DATA["NUM"]
        elif cfg.TRAIN["DATA_TYPE"] == "COCO":
            self.__nC = cfg.COCO_DATA["NUM"]
        else:
            self.__nC = cfg.Customer_DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__yolov4 = YOLOv4(
            weight_path=weight_path,
            out_channels=self.__out_channel,
            resume=resume,
            showatt=showatt
        )
        # small
        self.__head_s = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0]
        )
        # medium
        self.__head_m = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1]
        )
        # large
        self.__head_l = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2]
        )

    def forward(self, x):
        out = []
        [x_s, x_m, x_l], atten = self.__yolov4(x)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            if self.__showatt:
                return p, torch.cat(p_d, 0), atten
            return p, torch.cat(p_d, 0)


if __name__ == "__main__":
    from utils.flops_counter import get_model_complexity_info

    net = Build_Model()
    print(net)

    in_img = torch.randn(1, 3, 416, 416)
    p, p_d = net(in_img)
    flops, params = get_model_complexity_info(
        net, (224, 224), as_strings=False, print_per_layer_stat=False
    )
    print("GFlops: %.3fG" % (flops / 1e9))
    print("Params: %.2fM" % (params / 1e6))
    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
