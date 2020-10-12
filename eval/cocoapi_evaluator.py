import json
import tempfile
import torch
from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable
import config.yolov4_config as cfg
from utils.cocodataset import *
from utils.utils import *


class COCOAPIEvaluator:
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """

    def __init__(self, model_type, data_dir, img_size, confthre, nmsthre):
        """
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """

        augmentation = {
            "LRFLIP": False,
            "JITTER": 0,
            "RANDOM_PLACING": False,
            "HUE": 0,
            "SATURATION": 0,
            "EXPOSURE": 0,
            "RANDOM_DISTORT": False,
        }

        self.dataset = COCODataset(
            model_type=model_type,
            data_dir=data_dir,
            img_size=img_size,
            augmentation=augmentation,
            json_file="instances_val2017.json",
            name="val2017",
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.VAL["BATCH_SIZE"],
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.VAL["NUMBER_WORKERS"],
        )
        self.img_size = img_size
        self.confthre = confthre  # from darknet
        self.nmsthre = nmsthre  # 0.45 (darknet)

    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        ids = []
        data_dict = []
        dataiterator = iter(self.dataloader)
        while True:  # all the data in val2017
            try:
                img, _, info_img, id_ = next(dataiterator)  # load a batch
            except StopIteration:
                break
            info_img = [float(info) for info in info_img]
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                img = Variable(img.type(Tensor))
                _, outputs = model(img)
                outputs = outputs.unsqueeze(0)
                outputs = postprocess(outputs, 80, self.confthre, self.nmsthre)
                if outputs[0] is None:
                    continue
                outputs = outputs[0].cpu().data

            for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                label = self.dataset.class_ids[int(output[6])]
                box = yolobox2label((y1, x1, y2, x2), info_img)
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                score = float(
                    output[4].data.item() * output[5].data.item()
                )  # object score * class score
                A = {
                    "image_id": id_,
                    "category_id": label,
                    "bbox": bbox,
                    "score": score,
                    "segmentation": [],
                }  # COCO json format
                data_dict.append(A)

        annType = ["segm", "bbox", "keypoints"]

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            return cocoEval.stats[0], cocoEval.stats[1]
        else:
            return 0, 0
