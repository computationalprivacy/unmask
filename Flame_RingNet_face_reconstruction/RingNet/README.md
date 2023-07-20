# Masking using RingNet

This subfolder contains all code necessary for the masking pipeline. Given a 2D image, we will 
compute the 3D face reconstruction as a Flame model using RingNet, after which we will render this back to 2D using 
openDr and cut out the desired dimensions using a fine-tuned landmark detector. 

We have one main script called `batch_process.py`, which: 
- Takes as input the `batch_process_config.ini` file, with: 
    - `INPUT_FOLDER`: the input folder containing all images to be processed assuming a structured as outlined below
    - `OUTPUT_FOLDER`: the output folder where all images and .obj files will be saved as well for backup
- The `INPUT_FOLDER` assumes an input folder (either 'query' or 'database') as achieved by running the image_prep_selection pipeline, or: 
    - Database
        - Identity
            - Video number
                - Image path(s) [original image, prepped image]
    - Query
        - Identity
            - Video number
                - Image path(s) [original image, prepped image]
    Note that the script is designed to be run on the database and query folder seperately. 
- For each prepared image, will compute the mask and save the overlay result (so original image with mask) and the mask only, 
in the same directory as the prepared image. 

## Environment

For replicating the environment necessary for RingNet, we refer mostly to [their README](https://github.com/soubhiksanyal/RingNet). 
As this is running 2.7, we ran into some dependency issues. Some references that helpes us to create a working env on MacOs: 
- Following their instructions, we installed a Python 2.7 environment with tensorflow 1.12.0
- Activate the environment, upgrade pip using `pip install --upgrade pip==19.1.1`
- Run `pip install -r requirements.txt` (we made some changes to their requirements.txt that worked for us)
- Run `pip install opendr==0.77`
- Clone the [mesh repo](https://github.com/TimoBolkart/mesh)
- Followed the commands in the MakeFile: 
    - `pip install nose2 pyopengl pillow pyzmq pyyaml`
    - `cd mesh`
    - `python setup.py sdist`
    - `python setup.py --verbose bdist_wheel` (on MacOs you might have to run `export CFLAGS="-mmacosx-version-min=10.7 -stdlib=libc++"`)
    - `pip install --no-deps --verbose --no-cache-dir .`
    - `cd dist`
    - `pip install --verbose --no-cache-dir *.whl`
- Now download the models as instructed in the RingNet readme under 'Download models'
- Now the env should be able to run RingNet

We also need to be able to run the fine-tuned landmark detection model:
- Clone [this repo](https://github.com/codeniko/shape_predictor_81_face_landmarks)
- We only need the fine-tuned dlib model `shape_predictor_81_face_landmarks.dat`
- Create a folder `shape_predictor_81_face_landmarks` and move the model here
- Run `pip install dlib==19.24.0`

Additionally, you should run `pip install tqdm==4.64.1 configparser==4.0.2`

Now, after specifying the right folders in `batch_process_confi.ini`, you should be able to run the script `batch_process.py`

## What did we change from the RingNet functionality? 

- Added `landmark_cropping.py` to get a 2D facial mask rendered instead of the entire mesh. 
First, the 81 landmarks are detected using the fine-tuned model, after which scipy's convexhull is used
to cut out the mask from the rendered mesh and create both the overlay and the raw image. 
Note that we're still using the rendering as implemented using openDr by the RingNet authors. 
- Used their [demo.py](https://github.com/soubhiksanyal/RingNet/blob/master/demo.py) to write batch_helper.py and bath_process.py, 
which allows us to compute the mask we want for a series of images in a custom folder structure.

We thank the RingNet authors for clear documentation and great functionality that enabled us to implement this process. 
