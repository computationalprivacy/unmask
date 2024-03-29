# -*- coding: utf-8 -*-

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

class Detection:
    def __init__(self):
        caffemodel = "./checkpoint/Widerface-RetinaFace.caffemodel"
        deploy = "./checkpoint/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.999

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        return out

class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        if device_id == 'cpu':
            self.device = torch.device(device_id)
        else:
            self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

