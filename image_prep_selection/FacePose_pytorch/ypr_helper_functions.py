import warnings
import numpy as np
import torch
import math
from torchvision import transforms
import cv2

from pfld.pfld import PFLDInference, AuxiliaryNet
warnings.filterwarnings('ignore')

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu") #torch.device("cpu")
LANDMARK_MODEL_PATH = "./checkpoint/snapshot/checkpoint.pth.tar"
checkpoint = torch.load(LANDMARK_MODEL_PATH, map_location=device)
PLFD_BACKBONE = PFLDInference().to(device)
PLFD_BACKBONE.load_state_dict(checkpoint['plfd_backbone'])
PLFD_BACKBONE.eval()
PLFD_BACKBONE = PLFD_BACKBONE.to(device)
TRANSFORM = transforms.Compose([transforms.ToTensor()])

def get_num(point_dict,name,axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1)
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance

def get_ypr(image_bbox, img, transform = TRANSFORM, plfd_backbone = PLFD_BACKBONE):

    height, width = img.shape[:2]

    x1 = image_bbox[0]
    y1 = image_bbox[1]
    x2 = image_bbox[0] + image_bbox[2]
    y2 = image_bbox[1] + image_bbox[3]

    w = x2 - x1
    h = y2 - y1

    size = int(max([w, h]))
    cx = x1 + w / 2
    cy = y1 + h / 2
    x1 = cx - size / 2
    x2 = x1 + size
    y1 = cy - size / 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    cropped = img[int(y1):int(y2), int(x1):int(x2)]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

    cropped = cv2.resize(cropped, (112, 112))

    input = cv2.resize(cropped, (112, 112))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = transform(input).unsqueeze(0).to(device)
    _, landmarks = plfd_backbone(input)
    pre_landmark = landmarks[0]
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
    point_dict = {}
    i = 0
    for (x, y) in pre_landmark.astype(np.float32):
        point_dict[f'{i}'] = [x, y]
        i += 1

    # yaw
    point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
    point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
    point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
    crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
    yaw_mean = point_point(point1, point31) / 2
    yaw_right = point_point(point1, crossover51)
    yaw = (yaw_mean - yaw_right) / yaw_mean
    yaw = int(yaw * 71.58 + 0.7037)

    # pitch
    pitch_dis = point_point(point51, crossover51)
    if point51[1] < crossover51[1]:
        pitch_dis = -pitch_dis
    pitch = int(1.497 * pitch_dis + 18.97)

    # roll
    roll_tan = abs(get_num(point_dict, 60, 1) - get_num(point_dict, 72, 1)) / abs(
        get_num(point_dict, 60, 0) - get_num(point_dict, 72, 0))
    roll = math.atan(roll_tan)
    roll = math.degrees(roll)
    if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
        roll = -roll
    roll = int(roll)

    return yaw, pitch, roll, (x1, x2, y1, y2)
