# find sample for masking
import shutil
import os
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('filter_images_config.ini')

# We want to create the following structure:
# Database
# -- Identity
#    -- Video number
#       -- Image path(s) [original image, prepped image, overlay prep image]
# Query
# -- Identity
#    -- Video number
#       -- Image path(s) [original image, prepped image, overlay prep image]

SOURCE_DIR = config['DEFAULT']['SOURCE_DIR']

if __name__ == "__main__":

    identities = [k for k in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, k))
                  and k != 'ready_for_masking']

    masking_dir = os.path.join(SOURCE_DIR, 'ready_for_masking')
    if not os.path.exists(masking_dir):
        os.makedirs(masking_dir)

    database_dir = os.path.join(masking_dir, 'database')
    if not os.path.exists(database_dir):
        os.makedirs(database_dir)

    query_dir = os.path.join(masking_dir, 'query')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    identities_w_valid_results = 0

    rng = np.random.default_rng(seed = 42)

    for identity in identities:

        identity_dir = os.path.join(SOURCE_DIR, identity)
        results_path = os.path.join(identity_dir, 'filtered_images')
        if not os.path.exists(results_path):
            print('{} has not been processed yet.'.format(identity))
            continue

        # create a folder w the identity in both database and query
        identity_dir_database = os.path.join(database_dir, identity)
        if not os.path.exists(identity_dir_database):
            os.makedirs(identity_dir_database)
        identity_dir_query = os.path.join(query_dir, identity)
        if not os.path.exists(identity_dir_query):
            os.makedirs(identity_dir_query)

        result_images = [k for k in os.listdir(results_path) if k[-3:] == 'png']
        movies_w_valid_results = np.unique([k[0] for k in result_images])

        valid_movies = 0
        for movie in movies_w_valid_results:
            movie_valid_images = [k for k in result_images if k[0] == movie]
            # check if we have at least 2 images for each video
            if len(movie_valid_images) >= 2:
                # create a folder for the movie on both database and query
                movie_dir_database = os.path.join(identity_dir_database, movie)
                if not os.path.exists(movie_dir_database):
                    os.makedirs(movie_dir_database)
                movie_dir_query = os.path.join(identity_dir_query, movie)
                if not os.path.exists(movie_dir_query):
                    os.makedirs(movie_dir_query)

                # randomly sample 2 of these
                images_selected = rng.choice(movie_valid_images, size=2, replace = False)

                # add the first one to database
                database_image = images_selected[0]
                database_og_image_name = database_image.split('_')[0] + '.jpg'
                shutil.copy2(os.path.join(identity_dir, movie, database_og_image_name),
                             os.path.join(movie_dir_database, identity + '_' + database_og_image_name))
                shutil.copy2(os.path.join(results_path, database_image),
                             os.path.join(movie_dir_database, identity + '_' + database_image))

                # and the second to the query dir
                query_image = images_selected[1]
                query_og_image_name = query_image.split('_')[0] + '.jpg'
                shutil.copy2(os.path.join(identity_dir, movie, query_og_image_name),
                             os.path.join(movie_dir_query, identity + '_' + query_og_image_name))
                shutil.copy2(os.path.join(results_path, query_image),
                             os.path.join(movie_dir_query, identity + '_' + query_image))

                valid_movies += 1
        if valid_movies == 0:
            print('Did not find at least 2 valid images per video for {}.'.format(identity))
        else:
            print('Sampled 2 images per movie for {} and {} video(s).'.format(identity, valid_movies))

        if valid_movies >= 1:
            identities_w_valid_results += 1

    print('Found {} identities with valid results.'.format(identities_w_valid_results))