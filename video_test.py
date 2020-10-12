import utils.gpu as gpu
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
from timeit import default_timer as timer
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.log import Logger
from tensorboardX import SummaryWriter


class Detection(object):
    def __init__(
        self,
        gpu_id=0,
        weight_path=None,
        video_path=None,
        output_dir=None,
    ):
        self.__num_class = cfg.VOC_DATA["NUM"]
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_val = cfg.VAL["MULTI_SCALE_VAL"]
        self.__flip_val = cfg.VAL["FLIP_VAL"]
        self.__classes = cfg.VOC_DATA["CLASSES"]

        self.__video_path = video_path
        self.__output_dir = output_dir
        self.__model = Build_Model().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, showatt=False)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def Video_detection(self):
        import cv2

        vid = cv2.VideoCapture(self.__video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (
            int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        isOutput = True if self.__output_dir != "" else False
        if isOutput:
            print(
                "!!! TYPE:",
                type(self.__output_dir),
                type(video_FourCC),
                type(video_fps),
                type(video_size),
            )
            out = cv2.VideoWriter(
                self.__output_dir, video_FourCC, video_fps, video_size
            )
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            bboxes_prd = self.__evalter.get_bbox(frame)
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                visualize_boxes(
                    image=frame,
                    boxes=boxes,
                    labels=class_inds,
                    probs=scores,
                    class_labels=self.__classes,
                )

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(
                frame,
                text=fps,
                org=(3, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)
            if isOutput:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        default="E:\YOLOV4\weight/best.pt",
        help="weight file path",
    )
    parser.add_argument(
        "--video_path", type=str, default="bag.avi", help="video file path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="output file path"
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
    parser.add_argument("--mode", type=str, default="det", help="val or det")
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_val_path + "/event")
    logger = Logger(
        log_file_name=opt.log_val_path + "/log_video_detection.txt",
        log_level=logging.DEBUG,
        logger_name="CIFAR",
    ).get_log()

    Detection(
        gpu_id=opt.gpu_id,
        weight_path=opt.weight_path,
        video_path=opt.video_path,
        output_dir=opt.output_dir,
    ).Video_detection()
