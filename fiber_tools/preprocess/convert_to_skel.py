import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import scipy.ndimage as ndimage
import math
from pycocotools.coco import COCO
from fiber_tools.common.util.color_map import GenColorMap 
from fiber_tools.common.util.cv2_util import pologons_to_mask, mask_to_pologons
from get_skeleton_intersection import * 

from skimage.morphology import skeletonize
from least_square import least_squares

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

def get_skel(img, kernelSize):

#################
    skel = skeletonize(img)
    skel = skel.astype(np.uint8)

    skel = skel[25:skel.shape[0] - 25, 25 : skel.shape[1] - 25]

    canvas = np.zeros(skel.shape)
    intersections = get_skeleton_intersection(skel)
    # print (len(intersections)) 
    if len(intersections):
        # cv2.imshow('t', skel * 255.)
        # cv2.waitKey(0)

        for p in intersections:

            cv2.circle(skel, p , 2, (0), -1)
             
        im2, contours, hierarchy = cv2.findContours(skel,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        longest = 0

        for idx, contour in enumerate(contours):
            if cv2.arcLength(contour, False) > cv2.arcLength(contours[longest], False):
                longest = idx

        new_contour_arr = [tuple(row) for row in contours[longest]] #remove duplicates 
        uniques = np.unique(new_contour_arr, axis = 0)
        longest_contour = uniques
        
        # endpoints = [longest_contour[0,0,:], longest_contour[-1,0,:]]

        # cv2.circle(canvas, (longest_contour[0,0,0], longest_contour[0,0,1]) , 10, 1, -1)
        # cv2.circle(canvas, (longest_contour[-1,0,0], longest_contour[-1,0,1]), 10, 1, -1)
        cv2.drawContours(canvas, contours, longest, 1, kernelSize)
        skel = canvas

    else:
        im2, contours, hierarchy = cv2.findContours(skel,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            cv2.drawContours(canvas, contours, 0, 1, kernelSize)
        
            # new_contour_arr = [tuple(row) for row in contours[0]] #remove duplicates 
            # uniques = np.unique(new_contour_arr, axis = 0)
            # longest_contour = uniques
            # cv2.circle(canvas, (longest_contour[0,0,0], longest_contour[0,0,1]) , 10, 1, -1)   
            # cv2.circle(canvas, (longest_contour[-1,0,0], longest_contour[-1,0,1]), 10, 1, -1)
    
            # endpoints = [longest_contour[0,0,:], longest_contour[-1,0,:]]
            skel = canvas
        else:
            # endpoints = []
            skel = canvas
    
    indexes = np.where(skel > 0)
    # print(indexes)
    if len(indexes) > 1:
        k, _ = least_squares(indexes[1], indexes[0])
    else:
        k = 0
    
    # cv2.imshow('t', skel * 255.)
    # cv2.waitKey(0)
    # kernel = np.ones((kernelSize, kernelSize), np.uint8) 
    # skel = cv2.dilate(skel, kernel, iterations = 1)

    # cv2.imshow('t',skel * 255.)
    # cv2.waitKey(0)
###############
    # skel = skel.astype(np.uint8)
    # indexes = np.where(skel > 0)


    return skel.astype(np.uint8), k

def get_skel_distance_trans(img, radius_laplace):

#################

    img = img.astype(np.float32)  
    threshold = img > 0
    skeleton = np.zeros(img.shape)

    distance_image = ndimage.distance_transform_edt(threshold)
    morph_laplace_image = ndimage.morphological_laplace(distance_image, (radius_laplace,radius_laplace))
    skel = morph_laplace_image < morph_laplace_image.min()/2
    skeleton[skel] = 1
    # import pdb; pdb.set_trace()
    cv2.imshow('t',skeleton)
    cv2.waitKey(0)
###############
    # skel = skel.astype(np.uint8)
    # indexes = np.where(skel > 0)


    return skel.astype(np.uint8)

def trans_anno(img_root, anno_root, output_root, ori_file, target_file): #anno_root is the dir of input
    file_exist=False
    no_ori=False
    train_anno = os.path.join(output_root, target_file)
    if isfile(train_anno):
        file_exist = True
    ori_anno = os.path.join(anno_root,ori_file)
    if isfile(ori_anno)==False:
        no_ori = True
    if file_exist==False and no_ori==False:
        coco_fiber = COCO(ori_anno)
        coco_ids = coco_fiber.getImgIds()
        catIds = coco_fiber.getCatIds()
        train_data = []
        print('transforming annotations...')
        num_bad_images = 0
        num_good_images = 0

        for img_id in tqdm(coco_ids):
            img_ok = True
            img = coco_fiber.loadImgs(img_id)[0]
            file_path = os.path.join(img_root, img['file_name'])

            this_image = cv2.imread(file_path)
            img_shape = this_image.shape[:-1]

            annIds = coco_fiber.getAnnIds(imgIds=img['id'], catIds=catIds)
            anns = coco_fiber.loadAnns(annIds)

            
            seg = []
            skel_di0_polo = []
            skel_di3_polo = []
            skel_di5_polo = []

            skel_0_15 = []
            skel_15_30 = []
            skel_30_45 = []
            skel_45_60 = []
            skel_60_75 = []
            skel_75_90 = []
            skel_90_105 = []
            skel_105_120 = []
            skel_120_135 = []
            skel_135_150 = []
            skel_150_165 = []
            skel_165_180 = []
            final_12 = []

            for idx, ann in enumerate(anns):



                mask = pologons_to_mask(ann['segmentation'], img_shape)

                skel_di0, k = get_skel(mask, 2)

                angle = math.atan(k)* 180 / math.pi
                if angle > 180:
                    angle -= 180
                elif angle < 0:
                    angle += 180

                

                skel_di0_polo.append(skel_di0_polo_tmp)
                # skel_di5_polo.append(skel_di5_polo_tmp)
                # skel_di3_polo.append(skel_di3_polo_tmp)
                
                if (angle <= 15):
                    skel_0_15.append(skel_di0_polo_tmp)
                elif(angle > 15 and angle <= 30):
                    skel_15_30.append(skel_di0_polo_tmp)
                elif(angle > 30 and angle <= 45):
                    skel_30_45.append(skel_di0_polo_tmp)
                elif(angle > 45 and angle <= 60):
                    skel_45_60.append(skel_di0_polo_tmp)
                elif(angle > 60 and angle <= 75):
                    skel_60_75.append(skel_di0_polo_tmp)
                elif(angle > 75 and angle <= 90):
                    skel_75_90.append(skel_di0_polo_tmp)
                elif(angle > 105 and angle <= 120):
                    skel_105_120.append(skel_di0_polo_tmp)
                elif(angle > 120 and angle <= 135):
                    skel_120_135.append(skel_di0_polo_tmp)
                elif(angle > 135 and angle <= 150):
                    skel_135_150.append(skel_di0_polo_tmp)
                elif(angle > 150 and angle <= 165):
                    skel_150_165.append(skel_di0_polo_tmp)
                elif(angle > 165 and angle <= 180):
                    skel_165_180.append(skel_di0_polo_tmp)
    # import pdb;pdb.set_trace()

                try:
                    # testing1 = pologons_to_mask(skel_di5_polo_tmp,img_shape)
                    # testing2 = pologons_to_mask(skel_di3_polo_tmp,img_shape)
                    testing3 = pologons_to_mask(skel_di0_polo_tmp,skel_di0.shape)
                except:
                    num_bad_images += 1
                    img_ok = False
                
            final_12.append(skel_0_15)
            final_12.append(skel_15_30)
            final_12.append(skel_30_45)
            final_12.append(skel_45_60)
            final_12.append(skel_60_75)
            final_12.append(skel_75_90)
            final_12.append(skel_90_105)
            final_12.append(skel_105_120)
            final_12.append(skel_120_135)
            final_12.append(skel_135_150)
            final_12.append(skel_150_165)
            final_12.append(skel_165_180)
            
            bbox = ann['bbox']
            file_name = img['file_name']

            single_data = {}
            unit = {}
            # unit['bbox'] = bbox
            ###########################
            unit['segmentation'] = seg
            # unit['skel_di0'] = skel_di0_polo
            # unit['sk15'] = skel_0_15
            # unit['sk30'] = skel_15_30
            # unit['sk45'] = skel_30_45
            # unit['sk60'] = skel_45_60
            # unit['sk75'] = skel_60_75
            # unit['sk90'] = skel_75_90
            # unit['sk105'] = skel_90_105
            # unit['sk120'] = skel_105_120
            # unit['sk135'] = skel_120_135
            # unit['sk150'] = skel_135_150
            # unit['sk165'] = skel_150_165
            # unit['sk180'] = skel_165_180
            ######################################
            
            unit['final_12'] = final_12


            # unit['skel_di3'] = skel_di3_polo
            # unit['skel_di5'] = skel_di5_polo
            single_data ['unit'] = unit


            imgInfo ={}
            imgInfo ['imgID'] = img_id
            imgInfo ['img_name'] = file_name
            single_data ['imgInfo'] = imgInfo
            
            if img_ok:
                train_data.append(single_data)
                num_good_images = num_good_images + 1 
                # print(num_good_images)
            else:
                print(num_bad_images)
            # else:

                # print(num_bad_images)

            # new_ann['segmentation'] = skel_di5_polo
            # import pdb
            # pdb.set_trace()
            # coco_fiber.anns[annIds[idx]]['segmentation'] = skel_di5_polo

        print('saving transformed annotation...')
        with open(train_anno,'w') as wf:
            json.dump(train_data, wf)
            # json.dumps(coco_fiber.anns, wf)
            # json.dumps(coco_fiber.cats, wf)
            # json.dumps(coco_fiber.imgs, wf)
        print('done')
    if no_ori:
        print('''WARNING! There is no annotation file find   at {}. 
            Make sure you have put annotation files into the right folder.'''
            .format(ori_anno))


if __name__ == '__main__':
    # cnt_overlap_number()
    # root = '/opt/FTE/users/chrgao/datasets/Textile/data_for_zfb/new_data/mian_zhuma_hard_data/selected_img/'
    # root = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/instance_labeled_file'
    root = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file'
    # coco_anno_file = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/common_instance_json_format/fb_coco_style_fifth.json'
    coco_anno_file = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/fb_coco_style_fifth.json'
    # coco_anno_file = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/skel_5_coco_style_fifth.json'
    # show_anno(root, coco_anno_file)
    anno_root = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/'
    output_root = anno_root
    img_root = root
    ori_file = 'fb_coco_style_fifth.json'
    # target_file = 'skel_1_non_coco_fifth.json'
    target_file = 'testing_multi_v3.json'

    
    trans_anno(img_root, anno_root, output_root, ori_file, target_file)
    

