# Concerns about using a digital mask to safeguard patient privacy

In this repo, we provide the code used in the paper "Concerns about using a Digital Mask to safeguard patient privacy". 
The paper can be found here: XXX. 

Specifcally, we reproduce the setup used by [Yang et al.](https://www.nature.com/articles/s41591-022-01966-1) used to evaluate the privacy preservation of the proposed 
Digital Mask as closely as possible. For this we need three steps:
1. Image preprocessing;
2. Computing the masked images;
3. Executing the face recognition.

The code that is required to run all the three steps above is available in this repo, as laid out below. 
Each relevant subfolder also contains its own README with more instructions (including the requirements).

Note that the code leverages significant amount of code from other publicly available repositories. 
We provide a copy of their MIT licenses in the respective subfolders. 

## 1. Image preprocessing

The folder `image_prep_selection` contains all code necessary to select and preprocess facial images of a set of individuals. 
These preprocessed images will then be used as input for our masking pipeline. 
This code is entirely based on [FacePose_Pytorch](https://github.com/WIKI2020/FacePose_pytorch), which allows us to:
- Use a pretrained RetinaFace model to detect faces. An image is only considered when exactly one face is 
      detected with a minimum accuracy.
- Use a pre-trained PFLD model to detect facial landmarks, which are then used to estimate the pose (yaw, pitch and roll). 
      Only images with a pitch and yaw below 20° were considered (based on visual experimentation).

More details can be found in `image_prep_selection/FacePose_pytorch/README.md`.
 
## 2. Computing the masked images

The folder `Flame_RingNet_face_reconstruction` contains all code necessary to compute the mask from a given image. 
The main functionality is based on the [RingNet framework](https://github.com/soubhiksanyal/RingNet), which allows us to:
- Reconstruct the 3D face mesh as a Flame model from a given image
- Use Opendr to render this mesh back to 2D with the correct orientaton


In order to cut out the rendered mesh to the face, we use a custom landmark detection model (that also includes the forehead) 
as available [here](https://github.com/codeniko/shape_predictor_81_face_landmarks). Note that we only need the 
pretrained model which we use in `Flame_RingNet_face_reconstruction/RingNet/landmark_cropping.py`

Any changes or additions we have made to these functionalities are explained in the README in the subfolder, `Flame_RingNet_face_reconstruction/RingNet/README.md`.

## 3. Executing the face recognition

The folder `face_recognition` contains all code necessary to replicate the facial recognition setup.
The main functionality is based on [Insightface](https://github.com/deepinsight/insightface), which allows us 
to compute the embeddings for all query and database images considered. 

We then leverage the cosine similarity between the embeddings of query and database images.

Finally, given a query image, the predicted identity corresponds to the database image whose embedding is the closest 
to the embedding of the query image. 

As a metric, we use rank-1 accuracy for all individuals.

More details can be found in `face-recognition/README.md`.
    
## 4. Data

In our paper, we used data from the publicly available YouTubeFaces dataset, which is available for download [here](http://www.cs.tau.ac.il/~wolf/ytfaces/) with reference to the paper [Face Recognition in Unconstrained Videos with Matched Background Similarity](http://www.cs.tau.ac.il/~wolf/ytfaces/WolfHassnerMaoz_CVPR11.pdf).