import dlib
import numpy as np
from scipy.spatial import ConvexHull
import skimage

PREDICTOR_PATH = 'shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat'

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

def predict_landmarks(og_image, detector = DETECTOR, predictor = PREDICTOR):

    # find detected faces(s)
    dets = detector(og_image)

    if len(dets) == 1:
        for k, d in enumerate(dets):
            shape = predictor(og_image, d)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            break
    else:
        landmarks = None

    return landmarks

def get_mask(og_image, mesh_image):

    landmarks = predict_landmarks(og_image)

    if landmarks is not None:
        # not that landmark indices can be (very rarely) larger than image size
        # so replace by image bound if the case
        landmarks = np.clip(landmarks, 0, mesh_image.shape[0] - 1)
        vertices = ConvexHull(landmarks).vertices
        Y, X = skimage.draw.polygon(landmarks[vertices, 1], landmarks[vertices, 0])
        overlay = og_image.copy()
        overlay[Y, X] = mesh_image[Y, X]

        mask_only = 255 * np.ones(mesh_image.shape, dtype=np.uint8)
        mask_only[Y, X] = mesh_image[Y, X]
    else:
        print('Landmark detection did not find any face. No mask cropping will be executed.')
        overlay, mask_only = og_image, mesh_image

    return overlay, mask_only
