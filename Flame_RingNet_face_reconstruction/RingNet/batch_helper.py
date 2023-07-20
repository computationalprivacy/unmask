from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import skimage.io as io
import cv2
import tensorflow as tf
from psbody.mesh import Mesh
from PIL import Image

from util import renderer as vis_util
from util import image as img_util
from util.project_on_mesh import compute_texture_map
from run_RingNet import RingNet_inference

from landmark_cropping import get_mask

def visualize(img, proc_param, verts, cam, renderer, img_name='test_image'):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted = vis_util.get_original(
        proc_param, verts, cam, img_size=img.shape[:2])

    # Render results
    rend_img = renderer(
        vert_shifted*1.0, cam=cam_for_render, img_size=img.shape[:2])

    overlay, mask_only = get_mask(og_image = img, mesh_image = rend_img)

    # save overlay as png
    overlay_im = Image.fromarray(overlay)
    overlay_im.save(img_name + '_overlay.png')

    with open(img_name + 'overlay.npy', 'wb') as f:
        np.save(f, overlay)

    mask_only_im = Image.fromarray(mask_only)
    mask_only_im.save(img_name + '_mask_only.png')

    with open(img_name + 'mask_only.npy', 'wb') as f:
        np.save(f, mask_only)

    return None

def create_texture(img, proc_param, verts, faces, cam, texture_data):
    cam_for_render, vert_shifted = vis_util.get_original(proc_param, verts, cam, img_size=img.shape[:2])

    texture_map = compute_texture_map(img, vert_shifted, faces, cam_for_render, texture_data)
    return texture_map


def preprocess_image(config):
    img_path = config.img_path
    img = io.imread(img_path)
    if np.max(img.shape[:2]) != config.img_size:
        print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.0#scaling_factor
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def main(config, template_mesh, renderer):
    sess = tf.Session()
    model = RingNet_inference(config, sess=sess)
    input_img, proc_param, img = preprocess_image(config)
    vertices, flame_parameters = model.predict(np.expand_dims(input_img, axis=0), get_parameters=True)
    cams = flame_parameters[0][:3]
    visualize(img, proc_param, vertices[0], cams, renderer,
              img_name=config.out_folder + '/images/' + config.img_path.split('/')[-1][:-4])

    if config.save_obj_file:
        if not os.path.exists(config.out_folder + '/mesh'):
            os.mkdir(config.out_folder + '/mesh')
        mesh = Mesh(v=vertices[0], f=template_mesh.f)
        mesh.write_obj(config.out_folder + '/mesh/' + config.img_path.split('/')[-1][:-4] + '.obj')

    if config.save_flame_parameters:
        if not os.path.exists(config.out_folder + '/params'):
            os.mkdir(config.out_folder + '/params')
        flame_parameters_ = {'cam':  flame_parameters[0][:3], 'pose': flame_parameters[0][3:3+config.pose_params], 'shape': flame_parameters[0][3+config.pose_params:3+config.pose_params+config.shape_params],
         'expression': flame_parameters[0][3+config.pose_params+config.shape_params:]}
        np.save(config.out_folder + '/params/' + config.img_path.split('/')[-1][:-4] + '.npy', flame_parameters_)

    if config.neutralize_expression:
        from util.using_flame_parameters import make_prdicted_mesh_neutral
        if not os.path.exists(config.out_folder + '/neutral_mesh'):
            os.mkdir(config.out_folder + '/neutral_mesh')
        neutral_mesh = make_prdicted_mesh_neutral(config.out_folder + '/params/' + config.img_path.split('/')[-1][:-4] + '.npy', config.flame_model_path)
        neutral_mesh.write_obj(config.out_folder + '/neutral_mesh/' + config.img_path.split('/')[-1][:-4] + '.obj')

    if config.save_texture:
        if not os.path.exists(config.flame_texture_data_path):
            print('FLAME texture data not found')
            return
        texture_data = np.load(config.flame_texture_data_path, allow_pickle=True)[()]
        texture = create_texture(img, proc_param, vertices[0], template_mesh.f, cams, texture_data)

        if not os.path.exists(config.out_folder + '/texture'):
            os.mkdir(config.out_folder + '/texture')

        cv2.imwrite(config.out_folder + '/texture/' + config.img_path.split('/')[-1][:-4] + '.png', texture[:,:,::-1])
        mesh = Mesh(v=vertices[0], f=template_mesh.f)
        mesh.vt = texture_data['vt']
        mesh.ft = texture_data['ft']
        mesh.set_texture_image(config.out_folder + '/texture/' + config.img_path.split('/')[-1][:-4] + '.png')
        mesh.write_obj(config.out_folder + '/texture/' + config.img_path.split('/')[-1][:-4] + '.obj')

