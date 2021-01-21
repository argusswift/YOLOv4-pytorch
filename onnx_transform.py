import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
from utils.tools import *
# from tool.utils import *
from model.build_model import Build_Model
from eval.evaluator import *
import config.yolov4_config as cfg


def convert_predbox(pred_bbox, test_input_size, org_img_shape, valid_scale):
    """
    预测框进行过滤，去除尺度不合理的框
    """
    pred_coor = xywh2xyxy(pred_bbox[:, :4])
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (1)
    # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
    # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
    # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
    # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (2)将预测的bbox中超出原图的部分裁掉
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    # (3)将无效bbox的coor置为0
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4)去掉不在有效范围内的bbox
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5)将score低于score_threshold的bbox去掉
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > cfg.VAL["CONF_THRESH"]

    mask = np.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]

    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    return bboxes


def detect(session, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})
    bboxes = convert_predbox(outputs[-1], IN_IMAGE_H, (IN_IMAGE_H, IN_IMAGE_W), (0, np.inf))
    bboxes_prd = nms(bboxes, cfg.VAL["CONF_THRESH"], cfg.VAL["NMS_THRESH"])
    if bboxes_prd.shape[0] != 0:
        boxes = bboxes_prd[..., :4]
        class_inds = bboxes_prd[..., 5].astype(np.int32)
        scores = bboxes_prd[..., 4]
        visualize_boxes(image=image_src, boxes=boxes, labels=class_inds, probs=scores, class_labels=cfg.VOC_DATA["CLASSES"])
        path = os.path.join(cfg.PROJECT_PATH, "save.jpg")
        cv2.imwrite(path, image_src)
        print("saved images : {}".format(path))


def transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    model = Build_Model()
    pretrained_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          )

        print('Onnx model exporting done')
        return onnx_file_name


def main(weight_file=None, image_path= None, batch_size= 1, IN_IMAGE_H= 416, IN_IMAGE_W= 416):
    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Transform to onnx for demo
        onnx_path_demo = transform_to_onnx(weight_file, 1, IN_IMAGE_H, IN_IMAGE_W)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    image_src = cv2.imread(image_path)
    detect(session, image_src)


if __name__ == '__main__':
    import os.path as osp
    print("Converting to onnx and running demo ...")
    PROJECT_PATH = osp.abspath(osp.dirname(__file__))
    weight_file = osp.join(PROJECT_PATH, 'weight/best.pt')
    image_path = osp.join(PROJECT_PATH, '000001.jpg')
    main(weight_file=weight_file,image_path=image_path)