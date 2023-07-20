# Note that we have a problem when the landmark cropping does not detect a face
# then we still generate the masked face, but the entire mask
# so let's get rid of this: just delete the images
# we can detect so if the overlay image is the same as the og prep one

"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about RingNet is available at https://ringnet.is.tue.mpg.de.

based on github.com/akanazawa/hmr
"""
## Demo of RingNet.
## Note that RingNet requires a loose crop of the face in the image.
## Sample usage:
## Run the following command to generate check the RingNet predictions on loosely cropped face images
# python -m demo --img_path *.jpg --out_folder ./RingNet_output
## To output the meshes run the following command
# python -m demo --img_path *.jpg --out_folder ./RingNet_output --save_obj_file=True
## To output both meshes and flame parameters run the following command
# python -m demo --img_path *.jpg --out_folder ./RingNet_output --save_obj_file=True --save_flame_parameters=True
## To output both meshes and flame parameters and generate a neutralized mesh run the following command
# python -m demo --img_path *.jpg --out_folder ./RingNet_output --save_obj_file=True --save_flame_parameters=True --neutralize_expression=True

import os
import shutil
from tqdm import tqdm
import cv2

INPUT_FOLDER = 'ALL_YT_ready_for_masking/query'

if __name__ == '__main__':

    identities = [k for k in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, k))]
    problems = 0
    for identity in tqdm(identities):

        identity_dir = os.path.join(INPUT_FOLDER, identity)
        movies_w_valid_results = [k for k in os.listdir(identity_dir) if os.path.isdir(os.path.join(identity_dir, k))]

        for movie in movies_w_valid_results:
            movie_dir = os.path.join(identity_dir, movie)

            og_image_path = [k for k in os.listdir(movie_dir) if '_prep.' in k][0]
            og_image = cv2.imread(os.path.join(movie_dir, og_image_path))
            overlay_image_path = [k for k in os.listdir(movie_dir) if '_prep_overlay' in k][0]
            overlay_image = cv2.imread(os.path.join(movie_dir, overlay_image_path))

            if og_image.sum() == overlay_image.sum():
                print(og_image_path)
                print(overlay_image_path)
                print('Found one where it is exactly the same')
                problems += 1
                # what to do now?
                # change name to overlay_failed?
                os.rename(os.path.join(movie_dir, overlay_image_path),
                          os.path.join(movie_dir, overlay_image_path.replace('_overlay', '_failed_overlay')))

    print('Number of images that have an issue: ', problems)



