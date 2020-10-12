import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [
        0,
        5,
        2,
        16,
        9,
        44,
        6,
        3,
        17,
        62,
        21,
        67,
        18,
        19,
        4,
        1,
        64,
        20,
        63,
        7,
        72,
    ]

    def __init__(self, args, base_dir="", split="train", year="2014"):
        super().__init__()
        ann_file = os.path.join(
            base_dir, "annotations/instances_{}{}.json".format(split, year)
        )
        ids_file = os.path.join(
            base_dir, "annotations/{}_ids_{}.pth".format(split, year)
        )
        self.img_dir = os.path.join(base_dir, "images/{}{}".format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {"image": _img, "label": _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]
        _img = Image.open(os.path.join(self.img_dir, path)).convert("RGB")
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(
            self._gen_seg_mask(
                cocotarget, img_metadata["height"], img_metadata["width"]
            )
        )

        return _img, _target

    def _preprocess(self, ids, ids_file):
        print(
            "Preprocessing mask, this will take a while. "
            + "But don't worry, it only run once for each split."
        )
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(
                cocotarget, img_metadata["height"], img_metadata["width"]
            )
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description(
                "Doing: {}/{}, got {} qualified images".format(
                    i, len(ids), len(new_ids)
                )
            )
        print("Found number of qualified images: ", len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (
                    ((np.sum(m, axis=2)) > 0) * c
                ).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    # from dataloaders import custom_transforms as tr
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    coco_val = COCOSegmentation(
        args, base_dir="cocodata", split="val", year="2014"
    )

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample["image"].numpy()
            gt = sample["label"].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            # segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title("display")
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            # plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

####################################################################################################
# import torch.utils.data as data
# import os
# import cv2
# import random
# import numpy as np
# import torch
# from pycocotools.coco import COCO
# import config.yolov3_config_coco as cfg
# from torch.utils.data import DataLoader
# import utils.data_augment as dataAug
# import utils.tools as tools
#
# class ImageFolder_COCO(data.Dataset):
#     def __init__(self, root, dataType, transform=None, img_size=416):
#         self.img_size = img_size  # For Multi-training
#         self.root = root
#         self.coco = 0
#         annFile = '{}/annotations/instances_{}.json'.format(root, dataType)
#         self.coco = COCO(annFile)
#
#         self.image_set_index = sorted(self.coco.getImgIds())
#         for image_id in self.image_set_index:
#             anns = self.coco.imgToAnns[image_id]
#             if len(anns)==0:
#                 self.image_set_index.remove(image_id)
#
#         cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
#         self.classes = cats
#         self.num_classes = len(self.classes)
#         self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
#
#
#         self.transform = transform
#         self._imgInd_to_coco_imgId = dict(zip(range(len(self.image_set_index)),self.image_set_index))
#         self._imgpath = os.path.join(root, 'images',dataType, '%s')
#         self.classToInd = dict(zip(self.classes, range(len(self.classes))))
#
#
#     def load_gt(self, anns):
#         res = []
#         for obj in range(len(anns)):
#             cat_id = anns[obj]['category_id']
#             img_label_name= str(self.coco.cats[cat_id]['name'])
#             bb = anns[obj]['bbox'] # x1, y1, w, h
#             x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
#
#             # x1 = float(x1) / width
#             # x2 = float(x2) / width
#             # y1 = float(y1) / height
#             # y2 = float(y2) / height
#             if abs(y2-y1)<=1e-15 or abs(x2-x1)<=1e-15:
#                 continue
#
#             res.append([y1,x1,y2,x2,self.classToInd[img_label_name]])
#         res = np.array(res)
#         return res
#
#     def __parse_annotation(self, img, bboxes):
#
#         img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
#         img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
#         img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
#         img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
#
#         return img, bboxes
#
#     def __getitem__(self, index):
#         image_id = self._imgInd_to_coco_imgId[index]
#         img = cv2.imread(self._imgpath % self.coco.imgs[image_id]['file_name'])
#         anns = self.coco.imgToAnns[image_id]
#         target = self.load_gt(anns)
#         img_org, bboxes_org = self.__parse_annotation(img, target)
#         img_org = img_org.transpose(2, 0, 1)  # HWC->CHW
#
#         item_mix = random.randint(0, len(self.image_set_index) - 1)
#         image_id = self._imgInd_to_coco_imgId[item_mix]
#         img = cv2.imread(self._imgpath % self.coco.imgs[image_id]['file_name'])
#         anns = self.coco.imgToAnns[image_id]
#         target = self.load_gt(anns)
#         img_mix, bboxes_mix = self.__parse_annotation(img, target)
#         img_mix = img_mix.transpose(2, 0, 1)
#
#         img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
#         del img_org, bboxes_org, img_mix, bboxes_mix
#
#         label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)
#
#         img = torch.from_numpy(img).float()
#         label_sbbox = torch.from_numpy(label_sbbox).float()
#         label_mbbox = torch.from_numpy(label_mbbox).float()
#         label_lbbox = torch.from_numpy(label_lbbox).float()
#         sbboxes = torch.from_numpy(sbboxes).float()
#         mbboxes = torch.from_numpy(mbboxes).float()
#         lbboxes = torch.from_numpy(lbboxes).float()
#
#         return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
#
#     def __len__(self):
#         return len(self._imgInd_to_coco_imgId)
#
#     def __creat_label(self, bboxes):
#
#         anchors = np.array(cfg.MODEL["ANCHORS"])
#         strides = np.array(cfg.MODEL["STRIDES"])
#         train_output_size = self.img_size / strides
#         anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]
#
#         label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6+self.num_classes))
#                                                                       for i in range(3)]
#         for i in range(3):
#             label[i][..., 5] = 1.0
#
#         bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]   # Darknet the max_num is 30
#         bbox_count = np.zeros((3,))
#
#         for bbox in bboxes:
#             bbox_coor = bbox[:4]
#             bbox_class_ind = int(bbox[4])
#             bbox_mix = bbox[5]
#
#             # onehot
#             one_hot = np.zeros(self.num_classes, dtype=np.float32)
#             one_hot[bbox_class_ind] = 1.0
#             one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)
#
#             # convert "xyxy" to "xywh"
#             bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
#                                         bbox_coor[2:] - bbox_coor[:2]], axis=-1)
#             # print("bbox_xywh: ", bbox_xywh)
#
#             bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
#
#             iou = []
#             exist_positive = False
#             a = 0
#             for i in range(3):
#                 anchors_xywh = np.zeros((anchors_per_scale, 4))
#                 anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
#                 anchors_xywh[:, 2:4] = anchors[i]
#
#                 iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
#                 iou.append(iou_scale)
#                 iou_mask = iou_scale > 0.3
#
#                 if np.any(iou_mask):
#                     xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
#
#                     # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
#                     print('i {} yind {} xind {} iou_mask {} bbox_xywh {}'.format(i, yind, xind, iou_mask, bbox_xywh))
#                     label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
#                     label[i][yind, xind, iou_mask, 4:5] = 1.0
#                     label[i][yind, xind, iou_mask, 5:6] = bbox_mix
#                     label[i][yind, xind, iou_mask, 6:] = one_hot_smooth
#
#
#                     bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
#                     bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
#                     bbox_count[i] += 1
#
#                     exist_positive = True
#
#             if not exist_positive:
#                 best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
#                 best_detect = int(best_anchor_ind / anchors_per_scale)
#                 best_anchor = int(best_anchor_ind % anchors_per_scale)
#
#                 xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
#
#                 label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
#                 label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
#                 label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
#                 label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth
#
#                 bbox_ind = int(bbox_count[best_detect] % 150)
#                 bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
#                 bbox_count[best_detect] += 1
#
#         label_sbbox, label_mbbox, label_lbbox = label
#         sbboxes, mbboxes, lbboxes = bboxes_xywh
#
#         return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
#
#
# if __name__ == "__main__":
#
#     voc_dataset = ImageFolder_COCO(root='E:\code\object_detection\yolov3-other-master\YOLOV3-master\cocodata', dataType='train2014')
#     dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)
#
#     for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
#     # for i, (img, target) in enumerate(dataloader):
#         if i==0:
#             # a, b = img, target
#             print(img.shape)
#             print(label_sbbox.shape)
#             print(label_mbbox.shape)
#             print(label_lbbox.shape)
#             print(sbboxes.shape)
#             print(mbboxes.shape)
#             print(lbboxes.shape)
#
#             if img.shape[0] == 1:
#                 labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
#                                          label_lbbox.reshape(-1, 26)], axis=0)
#                 labels_mask = labels[..., 4]>0
#                 labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
#                                         axis=-1).reshape(-1, 1)], axis=-1)
#
#                 print(labels.shape)
#                 tools.plot_box(labels, img, id=1)
################################################################################################################
# class ImageFolder_COCO_eval(data.Dataset):
#     def __init__(self, root, dataType, transform=None):
#         self.root = root
#         self.coco = 0
#         annFile = '{}/annotations/instances_{}.json'.format(root, dataType)
#         self.coco = COCO(annFile)
#
#         self.image_set_index = sorted(self.coco.getImgIds())
#         cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
#         self.classes = ['__background__'] + cats
#         self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
#
#         self.transform = transform
#         self._imgInd_to_coco_imgId = dict(zip(range(len(self.image_set_index)),self.image_set_index))
#         self._imgpath = os.path.join(root, dataType, '%s')
#         self.classToInd = dict(zip(self.classes, range(len(self.classes))))
#
#     def __getitem__(self, index):
#         image_id = self._imgInd_to_coco_imgId[index]
#         img = cv2.imread(self._imgpath % self.coco.imgs[image_id]['file_name'])
#         img = img.astype(np.float32)
#         img = cv2.resize(img, (cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["TRAIN_IMG_SIZE"]))
#         img = img / 255
#         means = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         means = np.array(means, dtype=np.float32)
#         std = np.array(std, dtype=np.float32)
#         img[:, :, 0] -= means[0]
#         img[:, :, 1] -= means[1]
#         img[:, :, 2] -= means[2]
#         img[:, :, 0] /= std[0]
#         img[:, :, 1] /= std[1]
#         img[:, :, 2] /= std[2]
#
#         if isinstance(img, np.ndarray):
#             img = torch.from_numpy(img.transpose((2,0,1)).copy())
#
#
#         return img
#
#     def __len__(self):
#         return len(self._imgInd_to_coco_imgId)
