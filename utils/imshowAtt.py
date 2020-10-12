import cv2
import math
import random
import numpy as np
import os


def imshowAtt(beta, img=None):
    cv2.namedWindow("img")
    cv2.namedWindow("img1")
    if img is None:
        img = cv2.imread(
            os.path.join("VOCdevkit\VOC2007\JPEGImages/000001.jpg"), 1
        )  # the same input image

    h, w, c = img.shape
    img1 = img.copy()
    img = np.float32(img) / 255

    (height, width) = beta.shape[1:]
    h1 = int(math.sqrt(height))
    w1 = int(math.sqrt(width))

    for i in range(height):
        img_show = img1.copy()
        h2 = int(i / w1)
        w2 = int(i % h1)

        mask = np.zeros((h1, w1), dtype=np.float32)
        mask[h2, w2] = 1
        mask = cv2.resize(mask, (w, h))
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img_show * mask
        color = (random.random(), random.random(), random.random())
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img_show = img_show + 0.8 * clmsk - 0.8 * mskd

        cam = beta[0, i, :]
        cam = cam.view(h1, w1).data.cpu().numpy()
        cam = cv2.resize(cam, (w, h))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # cam = 1 / (1 + np.exp(-cam))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * (cam))
        cv2.imwrite("att.jpg", cam)
        cv2.imwrite("img.jpg", np.uint8(img_show))
        cv2.imshow("img", cam)
        cv2.imshow("img1", np.uint8(img_show))
        k = cv2.waitKey(0)
        if k & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit(0)
