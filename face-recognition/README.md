# Facial Recognition

This subfolder contains all code necessary to reproduce our facial recognition setup. 
Recall that we wish to compare different setups: 
- Face2Face: original face as both query database image
- Mask2Face: mask as query image and original image as database image
- Mask2Mask: mask as both query and database image

Note that we can define 'mask' in different ways (overlay, raw mask with white background, raw mask with black background). 
We refer to `matching_scenarios.py` for more detail. 

The facial recognition is based on [Insightface](https://github.com/deepinsight/insightface).

We have two main scripts: 
- `compute_embeddings`: given a source folder and a suffix for target images, compute all embeddings for all images. 
Save the results in a dictionary. You'll have to specify the following in `compute_embeddings_config.ini`:
    - `SOURCE_FOLDER`: The folder containing all images, assuming again the following structure:
        - Database
            - Identity
                - Video number
                    - Image path(s) [original image, prepped image]
        - Query
            - Identity
                - Video number
                    - Image path(s) [original image, prepped image]
    Note that the script is designed to be run on the database and query folder seperately.
    - `IMAGE_SUFFIX`: the suffix of the image of which you'd like to compute the embedding. 
    If following the paradigm from image_prep and the masking pipeline: 
        - `_prep.png`: the cropped face in its original state
        - `_prep_overlay.png`: the face with the mask overlay
        - `_prep_mask_only.png`: the raw mask. By default, this is with white background. 
        If you wish black backrgound you change so in the script.
    - `EMBEDDING_DICT_NAME`: the name of the dictionary of all embeddings (with image paths as keys and embeddings as values)
- `matching.py`: given the scenario, run through all query images and find the predicted identity in the database. 
  This is done by computing the cosine similarity between the query image embedding and all embeddings in the database. 
  The identity corresponding to the closest image in the database is then the prediction. 
  We then compute and print the rank 1 accuracy of all query images/identities
  You'll have to specify the following in `matching_config.ini`:
  - `SETUP`: the scenario to be executed
  - `DATABASE_SOURCE_FOLDER`: path to the database folder
  - `QUERY_SOURCE_FOLDER`: path to the query folder
  - `VALID_IDENTITIES`: path to a pickle file of a list of valid identities. To be tuned by the user.
  
## Environment
Install Anaconda3, python 3.7, and then `pip install -r requirements.txt`
                    
## Instructions to run

You can now first run `python compute_embeddings.py` for both the query as database for all image types of interest. 
After updating the config files and the scenarios, you can then run `python matching.py`.