import utils.gpu as gpu
from model.build_model import Build_Model
from eval.evaluator import Evaluator
import argparse
import time
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.log import Logger
import cv2
from eval.cocoapi_evaluator import COCOAPIEvaluator


class Evaluation(object):
    def __init__(self, gpu_id=0, weight_path=None, visiual=None, heatmap=False):
        self.__num_class = cfg.COCO_DATA["NUM"]
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_val = cfg.VAL["MULTI_SCALE_VAL"]
        self.__flip_val = cfg.VAL["FLIP_VAL"]

        self.__visiual = visiual
        self.__eval = eval
        self.__classes = cfg.COCO_DATA["CLASSES"]

        self.__model = Build_Model().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, showatt=heatmap)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def reset(self):
        path1 = os.path.join(cfg.DETECTION_PATH, "detection_result/")
        path2 = os.path.join(cfg.DETECTION_PATH, "ShowAtt/")
        for i in os.listdir(path1):
            path_file = os.path.join(path1, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)
        for i in os.listdir(path2):
            path_file = os.path.join(path2, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)

    def study(self):
        # Parameter study
        y = []
        for i in [0.08, 0.07, 0.06]:
            t = time.time()
            evaluator = COCOAPIEvaluator(
                model_type="YOLOv3",
                data_dir=cfg.DATA_PATH,
                img_size=cfg.VAL["TEST_IMG_SIZE"],
                confthre=i,
                nmsthre=cfg.VAL["NMS_THRESH"],
            )
            _, r = evaluator.evaluate(self.__model)
            y.append(
                str(i)
                + str("  ")
                + str(r)
                + str("  ")
                + str(
                    time.time() - t,
                )
            )
            np.savetxt("study.txt", y, fmt="%s")  # y = np.loadtxt('study.txt')

    def val(self):
        global logger
        logger.info("***********Start Evaluation****************")
        start = time.time()

        evaluator = COCOAPIEvaluator(
            model_type="YOLOv4",
            data_dir=cfg.DATA_PATH,
            img_size=cfg.VAL["TEST_IMG_SIZE"],
            confthre=cfg.VAL["CONF_THRESH"],
            nmsthre=cfg.VAL["NMS_THRESH"],
        )
        ap50_95, ap50 = evaluator.evaluate(self.__model)
        logger.info("ap50_95:{}|ap50:{}".format(ap50_95, ap50))
        end = time.time()
        logger.info("  ===val cost time:{:.4f}s".format(end - start))

    def Inference(self):
        global logger
        # clear cache
        self.reset()

        logger.info("***********Start Inference****************")
        imgs = os.listdir(self.__visiual)
        logger.info("images path: {}".format(self.__visiual))
        path = os.path.join(cfg.DETECTION_PATH, "detection_result")
        logger.info("saved images at: {}".format(path))
        inference_times = []
        for v in imgs:
            start_time = time.time()
            path = os.path.join(self.__visiual, v)
            img = cv2.imread(path)
            assert img is not None

            bboxes_prd = self.__evalter.get_bbox(img, v)
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(
                    image=img,
                    boxes=boxes,
                    labels=class_inds,
                    probs=scores,
                    class_labels=self.__classes,
                )
                path = os.path.join(
                    cfg.DETECTION_PATH, "detection_result/{}".format(v)
                )
                cv2.imwrite(path, img)
            end_time = time.time()
            inference_times.append(end_time - start_time)
        inference_time = sum(inference_times) / len(inference_times)
        fps = 1.0 / inference_time
        logging.info(
            "Inference_Time: {:.5f} s/image, FPS: {}".format(
                inference_time, fps
            )
        )


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/best.pt",
        help="weight file path",
    )
    parser.add_argument(
        "--log_val_path", type=str, default="log_val", help="weight file path"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)",
    )
    parser.add_argument(
        "--visiual", type=str, default="test_pic", help="val data path or None"
    )
    parser.add_argument(
        "--mode", type=str, default="val", help="val or det or study"
    )
    parser.add_argument(
        "--heatmap", type=str, default=False, help="whither show attention map"
    )
    opt = parser.parse_args()
    logger = Logger(
        log_file_name=opt.log_val_path + "/log_coco_val.txt",
        log_level=logging.DEBUG,
        logger_name="YOLOv4",
    ).get_log()

    if opt.mode == "val":
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            heatmap=opt.heatmap,
        ).val()
    if opt.mode == "det":
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            heatmap=opt.heatmap,
        ).Inference()
    else:
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            heatmap=opt.heatmap,
        ).study()
