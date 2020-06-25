import os
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform

import torch
import torch.utils.data as data

# import sys
# sys.path.insert(0, '../')

from utils.cv2_util import pologons_to_mask
from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

# import sys
# sys.path.insert(0, '../256.192.model/')
# from config import cfg
# import torchvision.datasets as datasets
# import pdb

class MscocoMulti(data.Dataset):
    def __init__(self, cfg, train=True):
        self.img_folder = cfg.img_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        self.crop_width = cfg.crop_width
        self.debug = False
        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry

        with open(cfg.gt_path) as anno_file:   
            self.anno = json.load(anno_file)
    def cropAndResizeImage(self, img, crop_width, output_shape):
        height, width = self.output_shape[0], self.output_shape[1]
        curr_height, curr_width = img.shape[0], img.shape[1]
        image = img[crop_width : img.shape[0] - crop_width, crop_width : img.shape[1] - crop_width]





    def augmentationCropImage(self, img, bbox, joints=None):  
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        mean_value = self.pixel_means
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])      
        bbox += add
        objcenter += add
        if self.is_train:
            joints[:, :2] += add
            inds = np.where(joints[:, -1] == 0)
            joints[inds, :2] = -1000000 # avoid influencing by data processing
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if self.is_train:
            crop_width = crop_width * (1 + 0.25)
            crop_height = crop_height * (1 + 0.25)  
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width  

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)                               

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        if self.is_train:
            joints[:, 0] = joints[:, 0] - min_x
            joints[:, 1] = joints[:, 1] - min_y

            joints[:, 0] *= x_ratio
            joints[:, 1] *= y_ratio
            label = joints[:, :2].copy()
            valid = joints[:, 2].copy()

        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))  
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

        if self.is_train:
            return img, joints, details
        else:
            return img, details


    def data_augmentation(self, img, label, operation):
        height, width = img.shape[0], img.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])
        
        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img = skimage.transform.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                             int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        for i in range(n):
            label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            label[i][2] *= (
            (label[i][0] >= 0) & (label[i][0] < width) & (label[i][1] >= 0) & (label[i][1] < height))

        # flip augmentation
        if operation == 1:
            img = cv2.flip(img, 1)
            cod = []
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                if x >= 0:
                    x = width - 1 - x
                cod.append((x, y, label[i][2]))
            # **** the joint index depends on the dataset ****    
            for (q, w) in self.symmetry:
                cod[q], cod[w] = cod[w], cod[q]
            for i in range(n):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
                allc.append(cod[i][2])
            label = np.array(allc).reshape(n, 3)

        # rotated augmentation
        if operation > 1:      
            angle = random.uniform(0, self.rot_factor)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotMat, (width, height))
            
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                v = label[i][2]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(int(coor[0]))
                allc.append(int(coor[1]))
                v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
                allc.append(int(v))
            label = np.array(allc).reshape(n, 3).astype(np.int)
        return img, label

    def get_keypoint_discs_offset(self, all_keypoints, offset_map_point, img_shape, radius):
        
        #WHY NOT JUST USE IMDILATE
        #TO DO: USE discs, then use the offsets map(single point), find the value. then Map back to discs. 

        map_shape = (img_shape[0], img_shape[1])
        offset_map_circle = np.zeros(map_shape)
        offset_map_circle_debug = np.zeros(map_shape)

        idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))

        discs = [[] for _ in range(len(all_keypoints))]
        # centers is the same with all keypoints. 
        # Will change later.
        centers = all_keypoints
        dists = np.zeros(map_shape+(len(centers),))
        for k, center in enumerate(centers):
            dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1)) #Return the distance map to the point.

        # import pdb; pdb.set_trace()

        if len(centers) > 0:
            inst_id = dists.argmin(axis=-1)   #To which points its the closest 

        count = 0
        for i in range(len(all_keypoints)):
            
            discs[i].append(np.logical_and(inst_id == count, dists[:,:,count]<= radius))
            # import pdb; pdb.set_trace()

            offset_map_circle_debug[discs[i][0]] = 1.0
            offset_map_circle[discs[i][0]] = offset_map_point[dists[:,:,count] == 0]

            # print(offset_map_point[dists[:,:,count] == 0])
            # print(count)

            count +=1
        # tmp = np.asarray(offset_map_circle_debug * 255.0)
        # cv2.imshow('t', tmp)
        # cv2.waitKey(0)


        return discs, offset_map_circle

  
    def __getitem__(self, index):

        a = self.anno[index]

        image_name = a['imgInfo']['img_name']
        img_path = os.path.join(self.img_folder, image_name)
        image = scipy.misc.imread(img_path, mode='RGB')
        # print(image.shape)
        crop_width = self.crop_width
        image = image[crop_width : image.shape[0] - crop_width, crop_width : image.shape[1] - crop_width]


        # if self.is_train:
        if True:
            start_points = a['unit']['start_points']
            start_points_offsets = a['unit']['start_points_offsets']
            control_points = a['unit']['control_points']
            off_sets = a['unit']['off_sets']

            # list object to arrary object with shape (N,2)
            start_points_label = np.reshape(np.asarray(start_points), (-1, 2))

            # h for horizontal, v for vertical
            start_points_offsets_h = np.reshape(np.asarray(start_points_offsets), (-1, 2))[:, 0]
            start_points_offsets_v = np.reshape(np.asarray(start_points_offsets), (-1, 2))[:, 1]


            control_points_label = np.reshape(np.asarray(control_points), (-1, 2))

            control_points_offsets_h = np.reshape(np.asarray(off_sets), (-1, 2))[:, 0]
            control_points_offsets_v = np.reshape(np.asarray(off_sets), (-1, 2))[:, 1]

            all_label = np.concatenate((start_points_label,control_points_label))
 
            

            cropped_image_shape = (image.shape[0], image.shape[1])

            

            mask = np.zeros(cropped_image_shape, dtype = np.float32)
            
            v_scale =  self.out_res[0] / cropped_image_shape[0]
            h_scale =  self.out_res[1] / cropped_image_shape[1] 
            
            start_points_map = np.zeros(cropped_image_shape, dtype = np.float32)
            control_points_map = np.zeros(cropped_image_shape, dtype = np.float32)
            off_sets_map_h = np.zeros(cropped_image_shape, dtype = np.float32) 
            off_sets_map_v = np.zeros(cropped_image_shape, dtype = np.float32) 

            # import pdb; pdb.set_trace()

            for idx, i in enumerate(start_points_label):
                # print (idx)
                start_points_map[i[1], i[0]] = 1
                off_sets_map_h[i[1], i[0]] = start_points_offsets_h[idx] * v_scale 
                off_sets_map_v[i[1], i[0]] = start_points_offsets_v[idx] * h_scale

            for idx, i in enumerate(control_points_label):
                # print(idx)
                control_points_map[i[1], i[0]] = 1
                off_sets_map_h[i[1], i[0]] = control_points_offsets_h[idx] * v_scale 
                off_sets_map_v[i[1], i[0]] = control_points_offsets_v[idx] * h_scale

            # import pdb; pdb.set_trace()
            # cv2.imshow('t', start_points_map * 255.)
            # cv2.waitKey()
            # import pdb; pdb.set_trace()

            # cv2.imshow('t', control_points_map * 255.)
            # cv2.waitKey()
            
            _, off_sets_map_h = self.get_keypoint_discs_offset(all_label, off_sets_map_h, cropped_image_shape, 5)
            _, off_sets_map_v = self.get_keypoint_discs_offset(all_label, off_sets_map_v, cropped_image_shape, 5)
 
            start_points_map = cv2.resize(start_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)
            control_points_map = cv2.resize(control_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)
            off_sets_map_h = cv2.resize(off_sets_map_h, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)
            off_sets_map_v = cv2.resize(off_sets_map_v, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)


        image = cv2.resize(image, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)
        # print (image.shape)
            # cv2.imshow('t', mask * 255.0)
            # cv2.waitKey(0)
            # points = np.array(a['unit']['skel_di5']).reshape(self.num_class, 3).astype(np.float32)

        # gt_bbox = a['unit']['GT_bbox']


        # if self.is_train:
        #     image, points, details = self.augmentationCropImage(image, gt_bbox, points)
        # else:
        #     image, details = self.augmentationCropImage(image, gt_bbox)

        if self.is_train:
            # image, points = self.data_augmentation(image, points, a['operation'])  
            img = im_to_torch(image)  # CxHxW
            # import pdb; pdb.set_trace()
        
            # Color dithering
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

            # points[:, :2] //= 4 # output size is 1/4 input size
            # pts = torch.Tensor(points)
        else:
            img = im_to_torch(image)

        img = color_normalize(img, self.pixel_means)



        # if self.is_train:
        if True:
            target15 = np.zeros((4,  self.out_res[0], self.out_res[1]))
            target11 = np.zeros((4, self.out_res[0], self.out_res[1]))
            target9 = np.zeros((4, self.out_res[0], self.out_res[1]))
            target7 = np.zeros((4, self.out_res[0], self.out_res[1]))

            # import pdb; pdb.set_trace()
            
            target15[0, :, :] = start_points_map 
            target11[0, :, :] = start_points_map
            target9[0, :, :] = start_points_map
            target7[0, :, :] = start_points_map
            
            target15[1, :, :] = control_points_map 
            target11[1, :, :] = control_points_map
            target9[1, :, :] = control_points_map
            target7[1, :, :] = control_points_map

            target15[2, :, :] = off_sets_map_h 
            target11[2, :, :] = off_sets_map_h
            target9[2, :, :] = off_sets_map_h
            target7[2, :, :] = off_sets_map_h

            target15[3, :, :] = off_sets_map_v 
            target11[3, :, :] = off_sets_map_v
            target9[3, :, :] = off_sets_map_v
            target7[3, :, :] = off_sets_map_v


            # import pdb; pdb.set_trace()


            for i in range(2):
                # if pts[i, 2] > 0: # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                target15[i] = generate_heatmap(target15[i], self.cfg.gk15)
                target11[i] = generate_heatmap(target11[i], self.cfg.gk11)
                target9[i] = generate_heatmap(target9[i], self.cfg.gk9)
                target7[i] = generate_heatmap(target7[i], self.cfg.gk7) 

            if self.debug:             
                cv2.imshow('t', target7[1] * 255.)
                cv2.waitKey(0)


                cv2.imshow('t', target7[0] * 255.)
                cv2.waitKey(0)

                cv2.imshow('t', target7[1] * 255.)
                cv2.waitKey(0)

                cv2.imshow('t', target15[2]/np.max(target15[2]) * 255.) # minus sign here
                cv2.waitKey(0)
                cv2.imshow('t', target15[3]/np.max(target15[3]) * 255.)
                cv2.waitKey(0)


            targets = [torch.Tensor(target15[:2]), torch.Tensor(target11[:2]), torch.Tensor(target9[:2]), torch.Tensor(target7[:2])]
            targets_offset = [torch.Tensor(target15[2:]), torch.Tensor(target11[2:]), torch.Tensor(target9[2:]), torch.Tensor(target7[2:])]
            # import pdb; pdb.set_trace()


  
        # import pdb; pdb.set_trace()
        meta = {'index' : index, 'imgID' : a['imgInfo']['imgID'], 
        # 'GT_bbox' : np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]), 
        'img_path' : img_path}##, 'augmentation_details' : details}

        if self.is_train:
            return img, targets, targets_offset, meta
        else:
            # meta['det_scores'] = a['score']
            return img, targets, targets_offset, meta

    def __len__(self):
        return len(self.anno)