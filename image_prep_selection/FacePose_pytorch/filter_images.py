import numpy as np
import pickle
import os
import cv2
from PIL import Image
from tqdm import tqdm
import configparser

from detect import AntiSpoofPredict
from ypr_helper_functions import get_ypr

config = configparser.ConfigParser()
config.read('filter_images_config.ini')

# we need to be sure for the face of interest
MIN_CONFIDENCE_FACE_DETECTOR = config.getfloat('DEFAULT', 'MIN_CONFIDENCE_FACE_DETECTOR')
# and make sure there is no other faces
MAX_CONFIDENCE_OTHER_FACE = config.getfloat('DEFAULT', 'MAX_CONFIDENCE_OTHER_FACE')

MAX_YAW = config.getint('DEFAULT', 'MAX_YAW')
MAX_ROLL = config.getint('DEFAULT', 'MAX_ROLL')
MAX_PITCH = config.getint('DEFAULT', 'MAX_PITCH')

CROP_MARGIN_SCALE = config.getfloat('DEFAULT', 'CROP_MARGIN_SCALE')

SOURCE_DIR = config['DEFAULT']['SOURCE_DIR']

def filter_image(img_path, face_detect_model):

    # (1) How many faces are detected? Only proceed if exactly 1
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    face_detect_output = face_detect_model.get_bbox(img)

    # take the number of faces detected with certain accuracy
    confidences = face_detect_output[:, 2]

    max_conf_index = np.argmax(confidences)  # should be 0
    left = face_detect_output[max_conf_index, 3] * width
    top = face_detect_output[max_conf_index, 4] * height
    right = face_detect_output[max_conf_index, 5] * width
    bottom = face_detect_output[max_conf_index, 6] * height

    image_bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]

    # (2) What is the estimation of yaw, pitch, roll?
    yaw, pitch, roll, coords = get_ypr(image_bbox, img)

    # (3) determine validity of image
    valid_image = True
    if max(confidences) <= MIN_CONFIDENCE_FACE_DETECTOR:
        valid_image = False

    n_faces = [k for k in confidences if k >= MAX_CONFIDENCE_OTHER_FACE]
    if len(n_faces) != 1:
        valid_image = False

    if (abs(yaw) >= MAX_YAW) or (abs(pitch) >= MAX_PITCH):
        valid_image = False

    # (3) if still valid image, return cropped
    if valid_image:
        x1, x2, y1, y2 = coords
        x_margin = CROP_MARGIN_SCALE * (x2 - x1) / 2
        y_margin = CROP_MARGIN_SCALE * (y2 - y1) / 2

        x1_new = max(0, int(x1 - x_margin))
        x2_new = min(width, int(x2 + x_margin))
        y1_new = max(0, int(y1 - y_margin))
        y2_new = min(height, int(y2 + y_margin))

        cropped = img[y1_new:y2_new, x1_new:x2_new, ::-1]
        cropped = cv2.resize(cropped, (224, 224))
    else:
        cropped = None

    return confidences, (yaw, pitch, roll), coords, cropped

if __name__ == "__main__":

    # load models
    FACE_DETECT_MODEL = AntiSpoofPredict(device_id=0)

    identities = [k for k in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, k)) and k != 'ready_for_masking']

    for identity in tqdm(identities):

        # create a dict to save all
        # we want {image_path:  {'confidences': confidences, 'ypr': (y, p, r), 'coords': coords}}
        dict_of_results_identity = dict()

        identity_dir = os.path.join(SOURCE_DIR, identity)
        if os.path.exists(os.path.join(identity_dir, 'dict_of_results.pkl')):
            continue

        results_path = os.path.join(identity_dir, 'filtered_images')
        if not os.path.exists(results_path):
            # Create a new directory because it does not exist
            os.makedirs(results_path)

        movies = [k for k in os.listdir(identity_dir) if os.path.isdir(os.path.join(identity_dir, k))]

        for movie in movies:
            movie_dir = os.path.join(identity_dir, movie)
            images = [k for k in os.listdir(movie_dir) if k[-3:] == 'jpg']

            for image in images:
                image_path = os.path.join(movie_dir, image)
                try:
                    confidences, ypr, coords, cropped = filter_image(image_path, FACE_DETECT_MODEL)
                    dict_of_results_identity[image_path] = {'confidences': confidences, 'ypr': ypr,
                                                            'coords': coords}

                    if cropped is not None:
                        result_image = Image.fromarray(cropped)
                        result_image.save(os.path.join(results_path, image[:-4] + '_prep.png'))
                        print('Image saved for ' + identity)
                except Exception as e:
                    print(e)

        with open(os.path.join(identity_dir, 'dict_of_results.pkl'), 'wb') as f:
            pickle.dump(dict_of_results_identity, f)
