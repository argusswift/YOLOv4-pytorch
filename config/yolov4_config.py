# coding=utf-8
# project
import os.path as osp
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

DATA_PATH = osp.join(PROJECT_PATH, 'data')


MODEL_TYPE = {
    "TYPE": "Mobilenet-YOLOv4"
}  # YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4

CONV_TYPE = {"TYPE": "DO_CONV"}  # conv type:DO_CONV or GENERAL

ATTENTION = {"TYPE": "NONE"}  # attention type:SEnet、CBAM or NONE

# train
TRAIN = {
    "DATA_TYPE": "VOC",  # DATA_TYPE: VOC ,COCO or Customer
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 1,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "YOLO_EPOCHS": 50,
    "Mobilenet_YOLO_EPOCHS": 120,
    "NUMBER_WORKERS": 0,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2,  # or None
    "showatt": False
}


# val
VAL = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.005,
    "NMS_THRESH": 0.45,
    "MULTI_SCALE_VAL": False,
    "FLIP_VAL": False,
    "Visual": False,
    "showatt": False
}

Customer_DATA = {
    "NUM": 3,  # your dataset number
    "CLASSES": ["unknown", "person", "car"],  # your dataset class
}

VOC_DATA = {
    "NUM": 20,
    "CLASSES": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
}

COCO_DATA = {
    "NUM": 80,
    "CLASSES": [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
}


# model
MODEL = {
    "ANCHORS": [
        [
            (1.25, 1.625),
            (2.0, 3.75),
            (4.125, 2.875),
        ],  # Anchors for small obj(12,16),(19,36),(40,28)
        [
            (1.875, 3.8125),
            (3.875, 2.8125),
            (3.6875, 7.4375),
        ],  # Anchors for medium obj(36,75),(76,55),(72,146)
        [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],
    ],  # Anchors for big obj(142,110),(192,243),(459,401)
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
}
