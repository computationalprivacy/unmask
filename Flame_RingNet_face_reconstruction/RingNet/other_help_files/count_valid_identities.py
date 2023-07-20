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

import os
import shutil
from tqdm import tqdm

QUERY_FOLDER = 'ALL_YT_ready_for_FR/query'
DATABASE_FOLDER = 'ALL_YT_ready_for_FR/database'

if __name__ == '__main__':

    identities = [k for k in os.listdir(QUERY_FOLDER) if os.path.isdir(os.path.join(QUERY_FOLDER, k))]
    valid_identities = 0

    for identity in tqdm(identities):

        # do query
        identity_dir = os.path.join(QUERY_FOLDER, identity)
        identity_is_valid = True
        movies_w_valid_results = [k for k in os.listdir(identity_dir) if
                                  os.path.isdir(os.path.join(identity_dir, k))]

        valid_movies = 0
        for movie in movies_w_valid_results:
            movie_dir = os.path.join(identity_dir, movie)

            results = [k for k in os.listdir(movie_dir) if '_prep_overlay.' in k]
            if len(results) == 1:
                valid_movies += 1

        if valid_movies == 0:
            identity_is_valid = False

        # do database
        identity_dir = os.path.join(DATABASE_FOLDER, identity)
        movies_w_valid_results = [k for k in os.listdir(identity_dir) if
                                  os.path.isdir(os.path.join(identity_dir, k))]

        valid_movies = 0
        for movie in movies_w_valid_results:
            movie_dir = os.path.join(identity_dir, movie)

            results = [k for k in os.listdir(movie_dir) if '_prep_overlay.' in k]
            if len(results) == 1:
                valid_movies += 1

        if valid_movies == 0:
            identity_is_valid = False

        if identity_is_valid:
            valid_identities += 1
        else:
            shutil.rmtree(os.path.join(QUERY_FOLDER, identity))
            shutil.rmtree(os.path.join(DATABASE_FOLDER, identity))
            print('Removed all files for ', identity)

    print('Number of valid identities: ', valid_identities)



