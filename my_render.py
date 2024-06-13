# Author: han sun (sunhan1997@sjtu.edu.cn)
"""Renders RGB-D images and xyz map of an object model."""

import os
import cv2
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import view_sampler
from PIL import Image
import os.path as osp
import math
import argparse

### create xyz map
def calc_xyz_bp_fast(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
        np.einsum(
            "ijkl,ijlm->ijkm",
            R.T.reshape(1, 1, 3, 3),
            depth.reshape(height, width, 1, 1)
            * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
            - T.reshape(1, 1, 3, 1),
        ).squeeze()
        * mask.reshape(height, width, 1)
    )
    return ProjEmb

def angle2Rmat(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# parser = argparse.ArgumentParser()
# parser.add_argument("--obj_name_id",default='1')
# args = parser.parse_args()

# PARAMETERS.
################################################################################
CAD_MODE_PATH = './example/obj_000016.ply'
obj_id = 0
output_dir = './tmp/custom_dataset'

renderer_type = "vispy"
ambient_weight = 0.1  # Weight of ambient light [0, 1]
shading = "phong"  # 'flat', 'phong'

# Intrinsic parameters for RGB rendering.
intrinsic = np.array(
            [[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]]  ## mp6d
)
fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
# Create the RGB renderer.
width_rgb, height_rgb = 640, 480 # mp6d
ren_rgb = renderer.create_renderer(
    width_rgb, height_rgb, renderer_type, mode="rgb", shading=shading
)
ren_rgb.set_light_ambient_weight(ambient_weight)
# Add object models to the RGB renderer.
ren_rgb.add_object(obj_id, CAD_MODE_PATH)

# Create the depth renderer.
(width_depth,height_depth,) = ( width_rgb, height_rgb,)
ren_depth = renderer.create_renderer(width_depth, height_depth, renderer_type, mode="depth")
ren_depth.add_object(obj_id, CAD_MODE_PATH)
################################################################################

# Rendering.
Poses = np.load('./example/obj_poses_level1.npy')



start_num = 81 #81 161, 321
for idx_frame in range(start_num,len(Poses)):
    print(idx_frame)
    pose = Poses[idx_frame]
    R = pose[:3,:3]
    t = np.array([0,0,500])
    rgb = ren_rgb.render_object(
        obj_id, R, t, fx, fy, cx, cy)["rgb"]
    depth = ren_depth.render_object(
        obj_id, R, t, fx, fy, cx, cy)["depth"]
    xyz_show = calc_xyz_bp_fast(depth, R, t, intrinsic)
    xyz_np = calc_xyz_bp_fast(depth, R, t, intrinsic)

    cv2.imshow('rgb', rgb)
    cv2.imshow('xyz_show', xyz_show/82.0)
    cv2.waitKey(0)


    rgb = Image.fromarray(np.uint8(rgb))
    rgb.save(osp.join(output_dir, f"{idx_frame-start_num:06d}.png"))
    np.save(osp.join(output_dir, f"{idx_frame-start_num:06d}.npy"), xyz_np)
