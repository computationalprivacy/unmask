from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from tqdm import tqdm
from psbody.mesh import Mesh
import configparser

from util import renderer as vis_util
from config_test import get_config

from batch_helper import main

config = configparser.ConfigParser()
config.read('batch_process_config.ini')
INPUT_FOLDER = config['DEFAULT']['INPUT_FOLDER']
OUTPUT_FOLDER = config['DEFAULT']['OUTPUT_FOLDER']

IMAGES = [os.path.join(INPUT_FOLDER, k) for k in os.listdir(INPUT_FOLDER) if k[-3:] == 'png']

if __name__ == '__main__':

    config = get_config()
    config.out_folder = OUTPUT_FOLDER

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    if not os.path.exists(config.out_folder + '/images'):
        os.mkdir(config.out_folder + '/images')

    config.save_obj_file = True
    config.save_flame_parameters = True

    template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')

    identities = [k for k in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, k))]

    for identity in tqdm(identities[:1]):

        identity_dir = os.path.join(INPUT_FOLDER, identity)
        movies_w_valid_results = [k for k in os.listdir(identity_dir) if os.path.isdir(os.path.join(identity_dir, k))]

        for movie in movies_w_valid_results:
            movie_dir = os.path.join(identity_dir, movie)
            prep_images = [k for k in os.listdir(movie_dir) if '_prep' in k]
            if len(prep_images) == 1:
                prep_image = prep_images[0]
                image_path = os.path.join(movie_dir, prep_image)

                # run ringnet
                config.img_path = image_path
                renderer = vis_util.SMPLRenderer(faces=template_mesh.f)
                main(config, template_mesh, renderer)

                # all results are in OUTPUT folder
                # but let's also copy the images to the INPUT FOLDER
                shutil.copy2(os.path.join(OUTPUT_FOLDER, 'images', prep_image.replace('prep', 'prep_overlay')),
                             os.path.join(movie_dir, prep_image.replace('prep', 'prep_overlay')))

                shutil.copy2(os.path.join(OUTPUT_FOLDER, 'images', prep_image.replace('prep', 'prep_mask_only')),
                             os.path.join(movie_dir, prep_image.replace('prep', 'prep_mask_only')))

                print(image_path + ' is done!')



