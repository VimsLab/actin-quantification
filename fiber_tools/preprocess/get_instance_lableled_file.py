import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
import shutil

def copy_file(root, coco_anno_file, output_dir):
    coco = COCO(coco_anno_file)
    ids = list(coco.imgs.keys())
    for img_id in ids:
        file_path = coco.loadImgs(img_id)[0]['file_name']
        input_path = os.path.join(root, file_path)
        output_path = os.path.join(output_dir, file_path)
        shutil.copy(input_path, output_path)
        

if __name__ == '__main__':
    # cnt_overlap_number()
    # first data
    root = '/opt/FTE/users/chrgao/datasets/Textile/data_for_zfb/new_data/mian_zhuma_hard_data/selected_img/'
    coco_anno_file = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/common_instance_json_format/fb_coco_style_fifth.json'
    output_dir = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/instance_labeled_file'
    copy_file(root, coco_anno_file, output_dir)
    

