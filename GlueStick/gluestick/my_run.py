import argparse
import os
from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline


def main():
    # Parse input parameters


    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': 1000, ### sunhan
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': 300,  ### sunhan
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

    gray0 = cv2.imread('/home/sunh/6D_ws/other_network/cnos-main/tmp/custom_dataset/000079.png', 0)
    gray1 = cv2.imread('/home/sunh/6D_ws/other_network/cnos-main/tmp/custom_dataset/000080.png', 0)

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}
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

    # Plot the matches
    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    # plot_images([img0, img1], ['Image 1 - detected lines', 'Image 2 - detected lines'], dpi=200, pad=2.0)
    # plot_lines([line_seg0, line_seg1], ps=4, lw=2)
    # plt.gcf().canvas.manager.set_window_title('Detected Lines')
    # plt.savefig('detected_lines.png')

    plot_images([img0, img1], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
    plot_keypoints([kp0, kp1], colors='c')
    plt.gcf().canvas.manager.set_window_title('Detected Points')
    plt.savefig('detected_points.png')

    plot_images([img0, img1], ['Image 1 - line matches', 'Image 2 - line matches'], dpi=200, pad=2.0)
    plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
    plt.gcf().canvas.manager.set_window_title('Line Matches')
    plt.savefig('line_matches.png')

    plot_images([img0, img1], ['Image 1 - point matches', 'Image 2 - point matches'], dpi=200, pad=2.0)
    plot_matches(matched_kps0, matched_kps1, 'green', lw=1, ps=0)
    plt.gcf().canvas.manager.set_window_title('Point Matches')
    plt.savefig('point_matches.png')
    plt.show()


if __name__ == '__main__':
    main()