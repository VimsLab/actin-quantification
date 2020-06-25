import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from fiber_tools.common.util.color_map import GenColorMap 
from fiber_tools.common.util.cv2_util import pologons_to_mask, mask_to_pologons
from get_skeleton_intersection import get_skeleton_intersection 

from skimage.morphology import skeletonize

from tqdm import tqdm
import json
from fiber_tools.common.util.osutils import isfile

def draw_mask(im, mask, color):

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32) 

    mask = mask>0
    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    #combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    combined = cv2.merge([r, g, b])
    
    return combined.astype(np.uint8)


def show_anno(root, coco_anno_file):

    CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)
    #coco-like, but not coco
    with open(coco_anno_file, "r") as read_file:
        coco = json.load(read_file)
    # import pdb 
    # pdb.set_trace()
    for i in range(len(coco)):
        item = coco[i]
        image_name = item['imgInfo']['img_name']
        file_path = os.path.join(root, image_name)
 
        im = cv2.imread(file_path)
        im_h, im_w, _ = im.shape
        skel_di5 = item['unit']['skel_di5']
        canvas = np.zeros(im.shape,dtype = np.float32) 
        canvas = im
        print (image_name)
        for idx, ann in enumerate(skel_di5):
            color = CLASS_2_COLOR[idx + 1]            

            mask = pologons_to_mask(ann,im.shape[:-1])
            
            #canvas = draw_mask(canvas, mask, color)
            canvas = draw_mask(canvas, mask, color)

        
            #cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            #cv2.putText(im, str(category_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        im = cv2.resize(im,(int(im_w), int(im_h)),interpolation=cv2.INTER_CUBIC)
        im_nobox = cv2.resize(canvas,(int(im_w), int(im_h)),interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        ret, bw = cv2.threshold(gray,0.000001, 255, cv2.THRESH_BINARY)
        # print (bw)
        bw = np.asarray(bw)
        bw = bw.astype(np.uint8)

        
        cv2.imshow("skel",canvas)
        cv2.waitKey(0)
        # cv2.destroyWindow('skel')
        
        # cv2.imwrite("/home/yiliu/work/fiberPJ/data/fiber_labeled_data/skel_mix_di3/" + prefix + "_label.png", im_nobox)
        # cv2.imwrite("/home/yiliu/work/fiberPJ/data/fiber_labeled_data/skel_mix_di3/" + prefix + "_rgb.png", im)
        # cv2.imwrite("/home/yiliu/work/fiberPJ/data/fiber_labeled_data/skel_mix_di3/" + prefix + "_bw.png", bw)

        # cv2.imshow("poly", im_nobox)
        # cv2.waitKey(0)
        # cv2.imshow("poly_2", im)
        # cv2.waitKey(0)

if __name__ == '__main__':
    # cnt_overlap_number()
    # root = '/opt/FTE/users/chrgao/datasets/Textile/data_for_zfb/new_data/mian_zhuma_hard_data/selected_img/'
    # root = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/instance_labeled_file'
    root = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file'
    # coco_anno_file = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/common_instance_json_format/fb_coco_style_fifth.json'
    # coco_anno_file = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/fb_coco_style_fifth.json'
    coco_anno_file = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/skel_5_non_coco_fifth.json'
    show_anno(root, coco_anno_file)
    anno_root = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/'
    output_root = anno_root
    img_root = root
    ori_file = 'fb_coco_style_fifth.json'
    target_file = 'skel_5_coco_style_fifth.json'

    
    

