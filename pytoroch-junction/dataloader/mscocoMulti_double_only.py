import os
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform
#import imageio
import copy

import torch
import torch.utils.data as data

from utils.cv2_util import pologons_to_mask
from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
from utils.preprocess import get_keypoint_discs_offset, compute_mid_long_offsets,compute_short_offsets, get_keypoint_discs, compute_mid_offsets
from utils.preprocess import compute_closest_control_point_offset
from utils.preprocess import visualize_offset, visualize_points, visualize_label_map
from utils.preprocess import draw_mask



import imageio
from PIL import Image
from matplotlib import pyplot as plt

# import sys
# sys.path.insert(0, '../256.192.model/')
# from config import cfg
# import torchvision.datasets as datasets
# import pdb


class MscocoMulti_double_only(data.Dataset):
    def __init__(self, cfg, train=True):
        self.img_folder = cfg.img_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.disc_radius = cfg.disc_radius
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

    def offset_to_color_map(self, offset):

        # print(np.max(offset/np.max(np.absolute(offset))))
        offset = offset / np.max(np.absolute(offset))
        # import pdb;pdb.set_trace()

        # color_map = np.zeros((3,) + offset.shape )
        positive = copy.copy(offset)
        negative = copy.copy(offset)

        positive[np.where(positive<0)] = 0
        negative[np.where(negative>0)] = 0
        # import pdb;pdb.set_trace()

        negative = np.absolute(negative)

        # import pdb;pdb.set_trace()
        r = positive.astype(np.float32)
        g = negative.astype(np.float32)
        b = np.zeros(offset.shape, dtype = np.float32)
        color_map = cv2.merge([r,g,b])
        # import pdb;pdb.set_trace()

        return color_map.astype(np.float32)



    def __getitem__(self, index):

        a = self.anno[index]
        # print(index)
        image_name = a['img_info']['file_name']
        # print (image_name)
        image_org_name = image_name.split('_predict')[0] + '_original.png'
        image_org_path = os.path.join(self.img_folder,'..','origin',image_org_name)
        # print (image_org_path)
        img_path = os.path.join(self.img_folder, image_name)

        # image = scipy.misc.imread(img_path, mode='RGB')
        image = imageio.imread(img_path)
        image = np.tile(image,(3,1,1)).transpose(1,2,0)


        crop_width = self.crop_width

        # if self.is_train:
        if self.is_train:

            # import pdb; pdb.set_trace()
            instances_annos = a['instances']
            image_intersections = a['intersections']


            cropped_image_shape = (image.shape[0], image.shape[1])
            mask_label = np.zeros(cropped_image_shape, dtype = np.float32)




            h_scale =  self.out_res[0] / cropped_image_shape[0]
            v_scale =  self.out_res[1] / cropped_image_shape[1]

            end_points_map_final = np.zeros(cropped_image_shape, dtype = np.float32)
            intersections_points_map_final = np.zeros(cropped_image_shape, dtype = np.float32)

            # Short offset
            end_points_off_sets_shorts_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            end_points_off_sets_shorts_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            intersections_points_shorts_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            intersections_points_shorts_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            skel_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            skel_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            #segmentation mask
            skel_mask = np.zeros(cropped_image_shape, dtype = np.float32)

            idx = 0
            for instance in instances_annos:

                end_points = instance['endpoints']
                skel = instance['skel']
                end_points_label = np.reshape(np.asarray(end_points), (-1, 2))
                skel_label = np.reshape(np.asarray(skel), (-1, 2))
                if idx == 0 :
                    end_points_label_final = end_points_label
                    skel_label_final = skel_label
                    idx = idx + 1
                else:
                    end_points_label_final = np.concatenate((end_points_label_final, end_points_label))
                    skel_label_final = np.concatenate((skel_label_final, skel_label))

            end_points_label_final = end_points_label_final.astype(int)


            for idx, i in enumerate(end_points_label_final):
                try:
                    end_points_map_final[int(i[1]), int(i[0])] = 1
                except:
                    print(image_name)
                    print(end_points_label_final)

            if (len(image_intersections) != 0):
                image_intersections_label = np.reshape(np.asarray(image_intersections), (-1, 2))

                for idx, i in enumerate(image_intersections_label):
                    intersections_points_map_final[int(i[1]), int(i[0])] = 1

                image_intersections_label = image_intersections_label.astype(int)

                image_intersections_discs = get_keypoint_discs(image_intersections_label, cropped_image_shape, radius = self.disc_radius)
                intersections_point_short_offset, canvas = compute_short_offsets(image_intersections_label, image_intersections_discs, map_shape = cropped_image_shape, radius =  self.disc_radius)
                _, intersections_points_map_final = get_keypoint_discs_offset(image_intersections_label, intersections_points_map_final, cropped_image_shape, radius = self.disc_radius)
                            # intersections points
                intersections_points_shorts_map_h_final = intersections_point_short_offset[:,:,0]
                intersections_points_shorts_map_v_final = intersections_point_short_offset[:,:,1]




            end_points_discs = get_keypoint_discs(end_points_label_final, cropped_image_shape, radius = self.disc_radius)
            end_points_short_offset, canvas = compute_short_offsets(end_points_label_final, end_points_discs, map_shape = cropped_image_shape, radius =  self.disc_radius)
            _, end_points_map_final = get_keypoint_discs_offset(end_points_label_final, end_points_map_final, cropped_image_shape, radius = self.disc_radius)


            segmentation_mask = (image[:,:,0] > 0) * 1.0

            # skel_offset, canvas = compute_closest_control_point_offset(end_points_label_final, segmentation_mask, map_shape = cropped_image_shape)



            end_points_map_final = cv2.resize(end_points_map_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            intersections_points_map_final = cv2.resize(intersections_points_map_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #end points short offsets
            end_points_off_sets_shorts_map_h_final = end_points_short_offset[:,:,0]
            end_points_off_sets_shorts_map_v_final = end_points_short_offset[:,:,1]
            end_points_off_sets_shorts_map_h_final = cv2.resize(end_points_off_sets_shorts_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            end_points_off_sets_shorts_map_v_final = cv2.resize(end_points_off_sets_shorts_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )


            intersections_points_shorts_map_h_final = cv2.resize(intersections_points_shorts_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            intersections_points_shorts_map_v_final = cv2.resize(intersections_points_shorts_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #Area offsets map
            # skel_map_h_final = skel_offset [:,:,0]
            # skel_map_v_final = skel_offset [:,:,1]

            skel_map_h_final = cv2.resize(skel_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            skel_map_v_final = cv2.resize(skel_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #Image
            image = cv2.resize(image, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)

            # Change scale
            end_points_off_sets_shorts_map_h_final *= h_scale
            end_points_off_sets_shorts_map_v_final *= v_scale

            intersections_points_shorts_map_h_final *= h_scale
            intersections_points_shorts_map_v_final *= v_scale

            skel_map_h_final *= h_scale
            skel_map_v_final *= v_scale



            if self.debug:

                # show control points
               #  import pdb; pdb.set_trace()
                print(image.shape)
                image_for_draw = np.tile(image,(3,1,1)).transpose(1,2,0)
                canvas = np.zeros_like(end_points_map_final)

                canvas = visualize_points(canvas, end_points_map_final)
                print(canvas.shape)
                print(image_for_draw.shape)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])

                cv2.imshow('end_points_map_final',combined_show)
                cv2.waitKey(0)

                # show end points
                canvas = np.zeros_like(intersections_points_map_final)
                canvas = visualize_points(canvas, intersections_points_map_final)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                cv2.imshow('intersections_points_map_final',combined_show)
                cv2.waitKey(0)

                # show short offsets
                canvas = np.zeros_like(intersections_points_map_final)
                canvas = visualize_offset(canvas, end_points_off_sets_shorts_map_h_final, end_points_off_sets_shorts_map_v_final)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                cv2.imshow('short offsets_end_points',combined_show)
                cv2.waitKey(0)



                # show control points
                canvas = np.zeros_like(intersections_points_map_final)
                canvas = visualize_offset(canvas, intersections_points_shorts_map_h_final, intersections_points_shorts_map_v_final)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                cv2.imshow('short offset intersections',combined_show)
                cv2.waitKey(0)

                # show area offset
                canvas = np.zeros_like(intersections_points_map_final)
                canvas = visualize_offset(canvas, skel_map_h_final, skel_map_v_final)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                cv2.imshow('short offsets_skel',combined_show)
                cv2.waitKey(0)



        if self.is_train:
            # image, points = self.data_augmentation(image, points, a['operation'])
            img = im_to_torch(image)  # CxHxW


            # Color dithering
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        else:
            # print (image.shape)
            image = np.asarray(image)
            h, w = image.shape[:2]
            crop_height = h % 32
            crop_width = w % 32
            image = image[int(crop_height / 2) : int(h - crop_height /2),int(crop_width / 2) : int(w - crop_width /2),:]

            new_h, new_w = image.shape[:2]

            image_org = imageio.imread(image_org_path, as_gray=False, pilmode="RGB")
            h_org = image_org.shape[0]
            w_org = image_org.shape[1]

            crop_org_height = max(0, h_org - h)
            crop_org_width = max(0, w_org - w)
            print(image.shape)
            print(image_org.shape)
            image_org = image_org[int(crop_org_height / 2) : int(h_org - crop_org_height / 2),\
                                    int(crop_org_width / 2) : int(w_org - crop_org_width /2),:]
            image_org = cv2.resize(image_org, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)

            # image = cv2.resize(image, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)
            print (image.shape)
            img = im_to_torch(image)


        img = color_normalize(img, self.pixel_means)


        if self.is_train:
        # if True:
            # print ('hello3')
            # ground_truth = [end_points_map_final, end_points_map_label, control_points_map_label,
            #                 control_points_off_sets_shorts_map_h_final, control_points_off_sets_shorts_map_v_final,
            #                 end_points_off_sets_shorts_map_h_final, end_points_off_sets_shorts_map_v_final,
            #                 off_sets_nexts_map_h_final,off_sets_nexts_map_v_final,
            #                 off_sets_prevs_map_h_final,off_sets_prevs_map_v_final,
            #                 area_offset_map_h_final, area_offset_map_v_final,
            #                 segmentation_mask_final]
            ground_truth = [end_points_map_final,
                            intersections_points_map_final,

                            end_points_off_sets_shorts_map_h_final, end_points_off_sets_shorts_map_v_final,
                            intersections_points_shorts_map_h_final, intersections_points_shorts_map_v_final,
                            ]

            targets = torch.Tensor(ground_truth)
        # import pdb; pdb.set_trace()
        meta = {'index' : index, 'imgID' : a['img_info']['file_name'],
    # 'GT_bbox' : np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]),
        'img_path' : img_path}##, 'augmentation_details' : details}

        if self.is_train:
            return img, targets, meta
        else:
            # meta['det_scores'] = a['score']
            return img, image_org, meta

    def __len__(self):
        return len(self.anno)
        # return 8
