"""
Author: Min Seok Lee and Wooseok Shin
TRACER: Extreme Attention Guided Salient Object Tracing Network
git repo: https://github.com/Karel911/TRACER
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from util.utils import get_files_recursive


def edge_generator(dataset_masks):
    parent_dir = os.path.dirname(os.path.normpath(dataset_masks))
    save_path = os.path.join(parent_dir, 'edges')
    os.makedirs(save_path, exist_ok=True)
    mask_list = get_files_recursive(dataset_masks, ext=["jpg", "jpeg", "png", "cr2", "webp", "tiff", 'tif'])

    for i, img_path in tqdm(enumerate(mask_list), total=len(mask_list)):
        _, img_rel_path = img_path.split(dataset_masks)
        if img_rel_path.startswith('/'):
            img_rel_path = img_rel_path[1:]

        mask = cv2.imread(img_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.int64(mask > 128)

        [gy, gx] = np.gradient(mask)
        tmp_edge = gy * gy + gx * gx
        tmp_edge[tmp_edge != 0] = 1
        bound = np.uint8(tmp_edge * 255)
        bound_file_path = os.path.join(save_path, img_rel_path)
        os.makedirs(os.path.dirname(bound_file_path), exist_ok=True)
        cv2.imwrite(bound_file_path, bound)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-masks-path', type=str, action='append', help='dataset masks path')
    args = parser.parse_args()
    for dataset_masks in args.dataset_masks_path:
        print('Generating edges for dataset', dataset_masks)
        edge_generator(dataset_masks)
