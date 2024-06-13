# Author: han sun (sunhan1997@sjtu.edu.cn)
""" Estimate the 6D pose estimation"""


import time
import torch
import numpy as np
from PIL import Image
import cv2
import os
from plyfile import PlyData
import argparse

#######  gluestrick
from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline
import scipy.io as scio
#######  DepthAnything
from Depth_Anything.depth_anything.dpt import DepthAnything

from utils import transform,calc_2d_bbox,OTLoss,create_bounding_box

import matplotlib.pyplot as plt


patch_h = 28
patch_w = 28
feat_dim = 1024

###########  load foundation model -----> DINOv2 from DepthAnything (checkpoints from https://github.com/LiheYoung/Depth-Anything)
dinov2_vits14 = DepthAnything.from_pretrained('./Depth_Anything/checkpoints/depth_anything_vitl14', local_files_only=True).cuda()


##########  set GlueStick
conf = {
    'name': 'two_view_pipeline',
    'use_lines': True,
    'extractor': {
        'name': 'wireframe',
        'sp_params': {
            'force_num_keypoints': False,
            'max_num_keypoints': 1000,  ### sunhan
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
        },
        'max_n_lines': 1000,  ### sunhan
    },
    'matcher': {
        'name': 'gluestick',
        'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
        'trainable': False,
    },
    'ground_truth': {
        'from_pose_depth': False,
    }
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline_model = TwoViewPipeline(conf).to(device).eval()



########### get the features of the template
template_number = 81
ref_features = torch.zeros(template_number, 784,1024).cuda()
for i in range(template_number):
    ref_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).cuda()
    ref_img_path = '/home/sunh/6D_ws/other_network/cnos-main/tmp/custom_dataset/{:06d}.png'.format(i)
    # 打开图像并转换为RGB模式
    ref_img = Image.open(ref_img_path).convert('RGB')
    bbox = ref_img.getbbox()
    ref_img = np.array(ref_img)
    ref_img = ref_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    ref_img_save = cv2.resize(
        ref_img, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite('/media/sunh/Samsung_T5/临时/ZS6D/{}_rgb.png'.format(i),ref_img_save)

    ref_img = Image.fromarray(ref_img)
    # 对图像进行转换操作，并将其存储在imgs_tensor的第一个位置
    ref_tensor[0] = transform(ref_img)[:3]

    with torch.no_grad():
        features_dict,_ = dinov2_vits14(ref_tensor)
        ref_feature = features_dict[0][0]
        ref_features[i] = ref_feature



####### intrinsic_matrix and read CAD model
intrinsic_matrix = np.array( [[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]] ) ## mp6d



# parser = argparse.ArgumentParser()
# parser.add_argument("--sence_id",default='3')
# parser.add_argument("--obj_name_id",default='1')
# args = parser.parse_args()

## start
obj_name_id = 16
## mp gt
model_path = os.path.join('./example/obj_000016.ply')
ply = PlyData.read(model_path)
pt_cld_data = ply.elements[0].data

object_number_in_mask = obj_name_id-15


for query_i in range(3):
    print(query_i)
    obj_name = 'obj_{:02d}'.format(obj_name_id-15)
    with open(f"./example/data/{query_i:06d}-box.txt") as f:
        bbox = f.readlines()
        obj_name_list = []
        for idx in range(len(bbox)):
            obj_bb = bbox[idx].split(" ")
            obj_name_list.append(obj_bb[0])
        if obj_name in obj_name_list:
            obj_id = obj_name_list.index(obj_name)
            obj_bb = bbox[obj_id].split(" ")
        else:
            print('no this object !!!')


    mask_test = cv2.imread('./example/data/{:06d}-label.png'.format(query_i), cv2.IMREAD_ANYDEPTH)
    mask_test[mask_test != object_number_in_mask] = 0
    mask_number = len(mask_test[mask_test == object_number_in_mask])


    ### get intrinsic_matrix/pose/bbox
    dataFile = './example/data/{:06d}-meta.mat'.format( query_i)
    data = scio.loadmat(dataFile)
    intrinsic_matrix = data['intrinsic_matrix']
    poses = data['poses']
    gt_pose = poses[:, :, obj_id]




    tar_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).cuda()
    rgb_o = cv2.imread('./example/data/{:06d}-color.png'.format(query_i))
    mask = mask_test


    rgb = cv2.cvtColor(rgb_o, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = 0
    ys, xs = np.nonzero(mask > 0)
    tar_bbox = calc_2d_bbox(xs, ys, (640, 480))  # mp6d
    x1, y1, w1, h1 = tar_bbox
    img = rgb.copy()
    img_cv2 = img[int(y1):int(y1 + h1), int(x1): int(x1 + w1)]

    ############### global matching
    img = Image.fromarray(img_cv2)
    tar_tensor[0] = transform(img)[:3]

    with torch.no_grad():
        features_dict = dinov2_vits14(tar_tensor)
        target_feature = features_dict[0]  #
        target_feature = target_feature[0][0][0]

    ########## OT distance
    score_list = []
    for i in range(0, template_number):
        ref_feature = ref_features[i]

        score = OTLoss(target_feature, ref_feature)
        score = score.detach().cpu().numpy()
        score_list.append(score)

    score_idx_5 = np.argsort(np.array(score_list))[:3]
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    ############### local matching
    match_number = []
    mkpts0_all = []
    mkpts1_all = []
    ref_img_all = []
    for i in range(3):
        frame = cv2.resize(
            img_cv2_gray, (512, 512), interpolation=cv2.INTER_AREA)

        ref_img_path = './tmp/custom_dataset/{:06d}.png'.format(score_idx_5[i])
        ref_img = Image.open(ref_img_path).convert('RGB')
        bbox = ref_img.getbbox()
        ref_img = np.array(ref_img)
        ref_img = ref_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        frame_ref = cv2.resize(
            ref_img_gray, (512, 512), interpolation=cv2.INTER_AREA)

        torch_gray0, torch_gray1 = numpy_image_to_torch(frame), numpy_image_to_torch(frame_ref)
        torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
        x = {'image0': torch_gray0, 'image1': torch_gray1}
        with torch.no_grad():
            pred = pipeline_model(x)

        pred = batch_to_np(pred)
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
        line_matches = pred["line_matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]

        valid_matches = line_matches != -1
        match_indices = line_matches[valid_matches]
        matched_lines0 = line_seg0[valid_matches]
        matched_lines1 = line_seg1[match_indices]

        img0, img1 = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), cv2.cvtColor(frame_ref, cv2.COLOR_GRAY2BGR)

        ############### show local keypoint matching
        plot_images([img0, img1], ['Image 1 - line matches', 'Image 2 - line matches'], dpi=200, pad=2.0)
        plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
        plt.gcf().canvas.manager.set_window_title('Line Matches')
        plt.savefig('/home/sunh/github/IndusPose/example/result/line_{}_{}.png'.format(query_i,i))

        plot_images([img0, img1], ['Image 1 - point matches', 'Image 2 - point matches'], dpi=200, pad=2.0)
        plot_matches(matched_kps0, matched_kps1, 'green', lw=1, ps=0)
        plt.gcf().canvas.manager.set_window_title('Point Matches')
        plt.savefig('/home/sunh/github/IndusPose/example/result/point_{}_{}.png'.format(query_i,i))

        match_number.append(len(matched_kps0))
        mkpts0_all.append(matched_kps0)
        mkpts1_all.append(matched_kps1)
        ref_img_all.append(ref_img)

    match_idx = match_number.index(max(match_number))
    mkpts0 = mkpts0_all[match_idx]
    mkpts1 = mkpts1_all[match_idx]
    ref_img = ref_img_all[match_idx]

    img_h, img_w, _ = img_cv2.shape
    ref_h, ref_w, _ = ref_img.shape

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)


    color_ref = cv2.imread('./tmp/custom_dataset/{:06d}.png'.format(score_idx_5[match_idx]))
    coor_ref = np.load('./tmp/custom_dataset/{:06d}.npy'.format(score_idx_5[match_idx]))
    ys, xs = np.nonzero(color_ref[:, :, 0] > 0)
    ref_bbox = calc_2d_bbox(xs, ys, (640, 480)) ## linemod
    x, y, w, h = ref_bbox
    coor_ref = coor_ref[int(y):int(y + h), int(x): int(x + w)]
    color_ref_patch = color_ref[int(y):int(y + h), int(x): int(x + w)]


    ### pnp
    mapping_2d = []
    mapping_3d = []

    if mkpts0.shape[0] ==0  or mkpts0.shape[0] < 40:
        coor_ref = cv2.resize(coor_ref, (w1, h1), interpolation=cv2.INTER_NEAREST)  # w h
        y_all, x_all = np.where(coor_ref[:, :, 0] > 0)
        for i_coor in range(len(y_all)):
            x, y = x_all[i_coor], y_all[i_coor]
            coord_3d = coor_ref[y, x]
            coord_3d = coord_3d
            mapping_2d.append((x + x1, y + y1))
            mapping_3d.append(coord_3d)
    else:
        coor_ref = cv2.resize(coor_ref, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)  # w h
        for (x0, y0), (x_r, y_r) in zip(mkpts0, mkpts1):
            x, y = int(x0*(img_w/512.)), int(y0*(img_h/512.))
            coord_3d = coor_ref[int(y_r*(ref_h/512.)), int(x_r*(ref_w/512.))]
            coord_3d = coord_3d
            mapping_2d.append((x + x1, y + y1))
            mapping_3d.append(coord_3d)




    ############################ PnP Ransac #####################
    try:
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                      np.array(mapping_2d, dtype=np.float32), intrinsic_matrix,
                                                      distCoeffs=None,
                                                      iterationsCount=50, reprojectionError=1.0,
                                                      flags=cv2.SOLVEPNP_P3P)
        R, _ = cv2.Rodrigues(rvecs)
        pred_pose = np.concatenate([R, tvecs.reshape((3, 1))], axis=-1)

        pred_pose[:3, 3] = pred_pose[:3, 3]
        img = create_bounding_box(rgb_o, pred_pose, pt_cld_data, intrinsic_matrix, color=(0, 0, 255))  # red
        img = create_bounding_box(img, gt_pose[:3, :4], pt_cld_data, intrinsic_matrix, color=(0, 255, 0))  # green
        cv2.imwrite('./example/result/{}.png'.format(query_i),img)


    except Exception as e:
        print('PNP error')


