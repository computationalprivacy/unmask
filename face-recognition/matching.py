"""Match images."""
import os
import numpy as np
import dill
from tqdm import tqdm
import configparser
from scipy.spatial import distance as scipy_dist

from matching_scenarios import SETUP_DICT

config = configparser.ConfigParser()
config.read('matching_config.ini')

SETUP = config['DEFAULT']['SETUP']
print('Running for setup: ', SETUP)
DATABASE_SOURCE_FOLDER = config['DEFAULT']['DATABASE_SOURCE_FOLDER']
QUERY_SOURCE_FOLDER = config['DEFAULT']['QUERY_SOURCE_FOLDER']

# define query and database dict
# query
QUERY_EMBEDDING_DICT = SETUP_DICT[SETUP]['QUERY_EMBEDDING_DICT']

with open(os.path.join(QUERY_SOURCE_FOLDER, QUERY_EMBEDDING_DICT), 'rb') as f:
    query_dict = dill.load(f)

# database
DATABASE_EMBEDDING_DICT = SETUP_DICT[SETUP]['DATABASE_EMBEDDING_DICT']

with open(os.path.join(DATABASE_SOURCE_FOLDER, DATABASE_EMBEDDING_DICT), 'rb') as f:
    database_dict = dill.load(f)

# get the valid identities
VALID_IDENTITIES = config['DEFAULT']['VALID_IDENTITIES']
with open(VALID_IDENTITIES, 'rb') as f:
    valid_identities = dill.load(f)

# get exactly one movie per identity
# make it random but deterministic
rng = np.random.default_rng(seed = 42)
valid_prefixes = []
for identity in valid_identities:
    database_identity_dir = os.path.join(DATABASE_SOURCE_FOLDER, identity)
    movies_w_valid_results = [k for k in os.listdir(database_identity_dir) if
                                  os.path.isdir(os.path.join(database_identity_dir, k))]
    movie_chosen = rng.choice(movies_w_valid_results, size=1, replace = False)[0]
    valid_prefixes.append(identity + '_' + movie_chosen)

# create subset of dicts needed for matching
query_dict_to_use, database_dict_to_use = query_dict.copy(), database_dict.copy()
for key in query_dict.keys():
    prefix = key.split('/')[5] + '_' + key.split('/')[6]
    if prefix not in valid_prefixes:
        query_dict_to_use.pop(key)

for key in database_dict.keys():
    prefix = key.split('/')[5] + '_' + key.split('/')[6]
    if prefix not in valid_prefixes:
        database_dict_to_use.pop(key)

# get all labels 
img_path2label = {}
for k in database_dict_to_use.keys():
    label = os.path.basename(k).split('.')[0] 
    img_path2label[k] = label

def get_distances(query_emb, db_embs):
    dist_from_prep = []
    
    for db_emb in db_embs:
        dist_from_prep.append(scipy_dist.cosine(
            query_emb, db_emb))

    return dist_from_prep

success_top1 = 0
label_in_rankings = []

for query_image in tqdm(list(query_dict_to_use.keys())):
    query_embedding = query_dict_to_use[query_image]
    ground_truth_label = os.path.basename(query_image).split('.')[0] 
    
    assert ground_truth_label in img_path2label.values()

    distances = get_distances(query_embedding, database_dict_to_use.values())
    
    sort_indices = np.argsort(distances)
    sorted_pred_labels = np.array(list(img_path2label.values()))[sort_indices]

    predicted_label = sorted_pred_labels[0]
    
    if predicted_label == ground_truth_label:
        success_top1 += 1

    label_in_rankings.append(list(sorted_pred_labels).index(ground_truth_label))

print('We computed the best match for n_individuals: ', len(query_dict_to_use))
print('Top 1 accuracy : ', success_top1 / len(query_dict_to_use))
print('Mean rank : ', np.mean(label_in_rankings))

