import os
import cv2
from tqdm import tqdm
import numpy as np
import dill
import configparser
from insightface.app import FaceAnalysis

def read_and_convert_rgb(img_path, as_float=True, black_background = False):
    """Read and convert image to grayscale."""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        img_rgb = img_rgb.astype(np.float32) / 255.0
    if black_background:
        img_rgb[img_rgb == 255] = 0
    return img_rgb

config = configparser.ConfigParser()
config.read('compute_embeddings_config.ini')

SOURCE_FOLDER = config['DEFAULT']['SOURCE_FOLDER']
IMAGE_SUFFIX = config['DEFAULT']['IMAGE_SUFFIX']
EMBEDDING_DICT_NAME = config['DEFAULT']['EMBEDDING_DICT_NAME']

img_path2embeddings = dict()

if __name__ == '__main__':

    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    identities = [k for k in os.listdir(SOURCE_FOLDER) if os.path.isdir(os.path.join(SOURCE_FOLDER, k))]
    failed_identities = 0
    for identity in tqdm(identities):

        identity_dir = os.path.join(SOURCE_FOLDER, identity)
        movies_w_valid_results = [k for k in os.listdir(identity_dir) if os.path.isdir(os.path.join(identity_dir, k))]
        for movie in movies_w_valid_results:
            movie_dir = os.path.join(identity_dir, movie)
            image_paths = [k for k in os.listdir(movie_dir) if IMAGE_SUFFIX in k]
            if len(image_paths) == 1:
                image_path = image_paths[0]
                image_path = os.path.join(movie_dir, image_path)
        
                img = read_and_convert_rgb(image_path, as_float=False, black_background = True)
                faces = app.get(img)
                if len(faces) > 0:
                    img_path2embeddings[image_path] = faces[0]['embedding']

                else:
                    print('Embedding computation failed')
                    failed_identities += 1

    print('Failed identities for ', SOURCE_FOLDER, ' : ', failed_identities)
    
    with open(os.path.join(SOURCE_FOLDER, EMBEDDING_DICT_NAME), 'wb') as f:
            dill.dump(img_path2embeddings, f)
