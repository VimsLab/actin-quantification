import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from fiber_tools.common.constant import TRAIN_LIST, TEST_LIST, FIBER_DICT

def pologons_to_mask(polygons, size):
    height, width = size
    # formatting for COCO PythonAPI
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def cnt_overlap_number(img_dir, ann_file, lap_thres=0.0002):
    coco = COCO(ann_file)
    ids = list(coco.imgs.keys())
    line = ''
    for img_id in ids:
        file_path = coco.loadImgs(img_id)[0]['file_name']
        file_path = os.path.join(img_dir, file_path)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        coco.showAnns(anns)
        im = cv2.imread(file_path)
        masks = []
        for idx, ann in enumerate(anns):
            bbox = ann['bbox']
            category_id = ann['category_id']
            mask = pologons_to_mask(ann['segmentation'],im.shape[:-1])
            masks.append(mask)

        overlap_num = 0
        for idx, anchor_mask in enumerate(masks):
            for cmp_idx in range(idx+1, len(masks)):
                cmp_mask = masks[cmp_idx]
                lap_num = np.sum(anchor_mask&cmp_mask)
                lap_rate = lap_num/(im.shape[1]*im.shape[0])
                if lap_rate>lap_thres:
                    overlap_num+=1
        tmp_line = file_path+','+str(overlap_num)+'\n'
        print(tmp_line)
        line+= tmp_line
    
    return line

def get_overlap():
    line = ''
    train_list, test_list, fiber_dict = TRAIN_LIST, TEST_LIST, FIBER_DICT
    for train_info in test_list:
        img_dir = fiber_dict[train_info]['img_dir']
        ann_file = fiber_dict[train_info]['ann_file']
        img_dir = os.path.join(prefix, img_dir)
        ann_file = os.path.join(prefix, ann_file)
        line+=cnt_overlap_number(img_dir, ann_file)

    return line

if __name__ == '__main__':
    prefix = '/opt/FTE/users/chrgao/datasets/'
    output = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/txt_format/overlap_reg/overlap.txt'
    w = open(output, 'w')
    line = get_overlap()
    w.write(line)
    w.close()
    