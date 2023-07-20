
## we want all identities for which we have at least 2 videos with valid mask and embedding

import os
from tqdm import tqdm
import dill
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--database_source_folder", type = str,
                    help = "path to the directory with the database input")
parser.add_argument("--query_source_folder", type = str,
                    help = "path to the directory with the query input")
parser.add_argument("--output_folder_valid_identities", type = str,
                    help = "path to the directory where to save the valid identities")

args = parser.parse_args()

DATABASE_SOURCE_FOLDER = args.database_source_folder
QUERY_SOURCE_FOLDER = args.query_source_folder
OUTPUT_FOLDER_VALID_IDENTITIES = args.output_folder_valid_identities

if __name__ == '__main__':

    with open(os.path.join(DATABASE_SOURCE_FOLDER, 'img_path2embedding{}.pickle'.format('_prep.png'.split('.')[0])), 'rb') as f:
        database_dict_prep = dill.load(f)
    with open(os.path.join(DATABASE_SOURCE_FOLDER, 'img_path2embedding{}.pickle'.format('_prep_overlay.png'.split('.')[0])), 'rb') as f:
        database_dict_overlay = dill.load(f)

    with open(os.path.join(QUERY_SOURCE_FOLDER, 'img_path2embedding{}.pickle'.format('_prep.png'.split('.')[0])), 'rb') as f:
        query_dict_prep = dill.load(f)
    with open(os.path.join(QUERY_SOURCE_FOLDER, 'img_path2embedding{}.pickle'.format('_prep_overlay.png'.split('.')[0])), 'rb') as f:
        query_dict_overlay = dill.load(f)

    identities = [k for k in os.listdir(DATABASE_SOURCE_FOLDER) if os.path.isdir(os.path.join(DATABASE_SOURCE_FOLDER, k))]

    valid_identities = []

    for identity in tqdm(identities):

        valid_identity = True

        # check for database
        database_identity_dir = os.path.join(DATABASE_SOURCE_FOLDER, identity)
        movies_w_valid_results = [k for k in os.listdir(database_identity_dir) if
                                  os.path.isdir(os.path.join(database_identity_dir, k))]
        
        if len(movies_w_valid_results) < 2:
            valid_identity = False
        else:
            for movie in movies_w_valid_results:
                movie_dir = os.path.join(database_identity_dir, movie)
                try:
                    # get embedding
                    image_paths_prep = [k for k in os.listdir(movie_dir) if '_prep.' in k]
                    emb = database_dict_prep[os.path.join(movie_dir, image_paths_prep[0])]
                    image_paths_overlay = [k for k in os.listdir(movie_dir) if '_prep_overlay.' in k]
                    emb = database_dict_overlay[os.path.join(movie_dir, image_paths_overlay[0])]
                except Exception as e:
                    valid_identity = False

        # check for query
        query_identity_dir = os.path.join(QUERY_SOURCE_FOLDER, identity)
        movies_w_valid_results = [k for k in os.listdir(query_identity_dir) if
                                  os.path.isdir(os.path.join(query_identity_dir, k))]

        if len(movies_w_valid_results) < 2:
            valid_identity = False
        else:
            for movie in movies_w_valid_results:
                movie_dir = os.path.join(query_identity_dir, movie)
                try:
                    # get embedding
                    image_paths_prep = [k for k in os.listdir(movie_dir) if '_prep.' in k]
                    emb = query_dict_prep[os.path.join(movie_dir, image_paths_prep[0])]
                    image_paths_overlay = [k for k in os.listdir(movie_dir) if '_prep_overlay.' in k]
                    emb = query_dict_overlay[os.path.join(movie_dir, image_paths_overlay[0])]
                except Exception as e:
                    valid_identity = False

        if valid_identity:
            valid_identities.append(identity)

    print('Number of valid identities: ', len(valid_identities))

    with open(f'{OUTPUT_FOLDER_VALID_IDENTITIES}/valid_identities.pickle', 'wb') as f:
            dill.dump(valid_identities, f)
