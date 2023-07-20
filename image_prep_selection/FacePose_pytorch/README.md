# Image preparation and selection

This subfolder contains all code necessary to filter the correct images from a given dataset to be used for our masking pipeline.

This is entirely based on [FacePose_Pytorch](https://github.com/WIKI2020/FacePose_pytorch), which allows us to:
- Use a pretrained RetinaFace model to detect faces. An image is only considered when exactly one face is 
detected with a minimum accuracy.
- Use a pre-trained PFLD model to detect facial landmarks, which are then used to estimate the pose (yaw, pitch and roll). 
Only images with a pitch and yaw below 20° were considered (based on visual experimentation).

We have two main scripts: 
- `filter_images.py`: given a source folder and the filtering hyperparameters, this script will run through all images 
and save the cropped versions of all facial images that meet the requirements in a new folder called 'filtered_images' for each identity. 
- `select_for_masking.py`: after applying the script above, this script will randomly sample two frames from each video for each identity. 
The final result will be a folder called 'ready_for_masking' that is prepared as input for our facial recognition set up. 
One image will be added to the database folder, while the other will be added to the query folder. More specifically, the output will look like:
    - Database
        - Identity
            - Video number
                - Image path(s) [original image, prepped image]
    - Query
        - Identity
            - Video number
                - Image path(s) [original image, prepped image]

## Environment
- First you have to install the requirements for FacePose_Pytorch. 
    - Install Anaconda3, python 3.7,and then:
    - `pip install numpy opencv-python`
    - `pip install torch==1.4.0`
    - `pip install torchvision==0.5.0`
- The libraries we then added are: 
    - `Pillow==9.3.0`
    - `tqdm==4.64.1`
    - `configparser==5.3.0`

## Change the config file as you wish
- Refer to filter_images_config.ini, where you can specify: 
    - `MIN_CONFIDENCE_FACE_DETECTOR`: the minimum confidence score you want for one detected face
    - `MAX_CONFIDENCE_OTHER_FACE`: the maximum confidence you want for a potentially second detected face
    - `MAX_YAW`: the maximum value for estimated yaw
    - `MAX_ROLL`: the maximum value for estimated roll
    - `MAX_PITCH`: the maximum value for estimated pitch
    - `CROP_MARGIN_SCALE`: the scale of the face bounding box for the cropped image as result
    - `SOURCE_DIR`: the source directory where the images are stored. We assume the following structure: 
        - SOURCE_DIR/
            - Identity_1:
                - Video_0: 
                    - Frame_0...X.jpg
                - Video_N: 
                    - Frame_0...X.jpg
            - Identity_K:
                - Video_0: 
                    - Frame_0...X.jpg
                - Video_M: 
                    - Frame_0...X.jpg
                    
## Instructions to run

After you've set up the python environment and changed the config file, you can run the filtering file with: `python filter_images.py`

You can then also run: `python select_for_masking.py`

If both are successful, you can then go ahead to the masking stage, before going to the facial recognition. 