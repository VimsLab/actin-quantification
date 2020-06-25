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
from utils.preprocess import visualize_offset, visualize_points


import imageio
from PIL import Image  
from matplotlib import pyplot as plt

# import sys
# sys.path.insert(0, '../256.192.model/')
# from config import cfg
# import torchvision.datasets as datasets
# import pdb
def draw_mask(im, mask, color):

    
    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32) 
    
    # import pdb; pdb.set_trace()
    mask = mask>0
    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    #combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    combined = cv2.merge([r, g, b])
    
    return combined.astype(np.uint8)

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
        self.debug = True
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
        image_name = a['imgInfo']['img_name']
        img_path = os.path.join(self.img_folder, image_name)
        # image = scipy.misc.imread(img_path, mode='RGB')
        image = imageio.imread(img_path)
        # print(image.shape)
        crop_width = self.crop_width
        image = image[crop_width : image.shape[0] - crop_width, crop_width : image.shape[1] - crop_width]

        # if self.is_train:
        if True:
            end_points = a['unit']['end_points']
            control_points = a['unit']['control_points']
            off_sets_prevs = a['unit']['off_sets_prevs']
            off_sets_nexts = a['unit']['off_sets_nexts']

            # list object to arrary object with shape (N,2)
            end_points_label = np.reshape(np.asarray(end_points), (-1, 2))

            # h for horizontal, v for vertical
            off_sets_prevs_h = np.reshape(np.asarray(off_sets_prevs), (-1, 2))[:, 0]
            off_sets_prevs_v = np.reshape(np.asarray(off_sets_prevs), (-1, 2))[:, 1]


            control_points_label = np.reshape(np.asarray(control_points), (-1, 2))

            off_sets_nexts_h = np.reshape(np.asarray(off_sets_nexts), (-1, 2))[:, 0]
            off_sets_nexts_v = np.reshape(np.asarray(off_sets_nexts), (-1, 2))[:, 1]

            # all_label = np.concatenate((end_points_label,control_points_label))
            all_label = control_points_label
            

            cropped_image_shape = (image.shape[0], image.shape[1])

            

            mask = np.zeros(cropped_image_shape, dtype = np.float32)
            
            h_scale =  self.out_res[0] / cropped_image_shape[0]
            v_scale =  self.out_res[1] / cropped_image_shape[1] 
            
            end_points_map = np.zeros(cropped_image_shape, dtype = np.float32)

            control_points_map = np.zeros(cropped_image_shape, dtype = np.float32)
            off_sets_nexts_map_h = np.zeros(cropped_image_shape, dtype = np.float32) 
            off_sets_nexts_map_v = np.zeros(cropped_image_shape, dtype = np.float32)
            off_sets_prevs_map_h = np.zeros(cropped_image_shape, dtype = np.float32) 
            off_sets_prevs_map_v = np.zeros(cropped_image_shape, dtype = np.float32) 

            # import pdb; pdb.set_trace()

            for idx, i in enumerate(end_points_label):
                end_points_map[i[1], i[0]] = 1

            #     off_sets_nexts_map_h[i[1], i[0]] = start_points_offsets_h[idx] * v_scale 
            #     off_sets_nexts_map_v[i[1], i[0]] = start_points_offsets_v[idx] * h_scale

            for idx, i in enumerate(control_points_label):
                # print(idx)
                control_points_map[i[1], i[0]] = 1
                off_sets_nexts_map_h[i[1], i[0]] = off_sets_nexts_h[idx] #* h_scale 
                off_sets_nexts_map_v[i[1], i[0]] = off_sets_nexts_v[idx] #* v_scale
                off_sets_prevs_map_h[i[1], i[0]] = off_sets_prevs_h[idx] #* h_scale
                off_sets_prevs_map_v[i[1], i[0]] = off_sets_prevs_v[idx] #* v_scale

            end_points_discs = get_keypoint_discs(end_points_label, cropped_image_shape, radius = self.disc_radius)
            control_points_discs = get_keypoint_discs(control_points_label, cropped_image_shape, radius = self.disc_radius)
            
            control_points_short_offset, canvas = compute_short_offsets(control_points_label, control_points_discs, map_shape = cropped_image_shape, radius =  self.disc_radius)
            end_points_short_offset, canvas = compute_short_offsets(end_points_label, end_points_discs, map_shape = cropped_image_shape, radius =  self.disc_radius)

          
            control_points_prevs_offset, canvas = compute_mid_offsets(control_points_label, off_sets_prevs_map_h, off_sets_prevs_map_v, cropped_image_shape, control_points_discs)
            control_points_nexts_offset, canvas = compute_mid_offsets(control_points_label, off_sets_nexts_map_h, off_sets_nexts_map_v, cropped_image_shape, control_points_discs)
            
            control_points_prevs_prevs_offset, canvas = compute_mid_long_offsets(control_points_label, off_sets_prevs_map_h, off_sets_prevs_map_v, cropped_image_shape, control_points_discs)
            control_points_nexts_nexts_offset, canvas = compute_mid_long_offsets(control_points_label, off_sets_nexts_map_h, off_sets_nexts_map_v, cropped_image_shape, control_points_discs)


            _, end_points_map = get_keypoint_discs_offset(end_points_label, end_points_map, cropped_image_shape, radius =  self.disc_radius)
            _, control_points_map = get_keypoint_discs_offset(control_points_label, control_points_map, cropped_image_shape, radius =  self.disc_radius)
 
            # end_points_map = cv2.resize(end_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # control_points_map = cv2.resize(control_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # off_sets_nexts_map_h = cv2.resize(off_sets_nexts_map_h, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # off_sets_nexts_map_v = cv2.resize(off_sets_nexts_map_v, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            
            # off_sets_prevs_map_h = cv2.resize(off_sets_prevs_map_h, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # off_sets_prevs_map_v = cv2.resize(off_sets_prevs_map_v, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
  
            end_points_map = cv2.resize(end_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            end_points_off_sets_shorts_map_h = cv2.resize(end_points_short_offset[:,:,0], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            end_points_off_sets_shorts_map_v = cv2.resize(end_points_short_offset[:,:,1], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )



            control_points_map = cv2.resize(control_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_shorts_map_h = cv2.resize(control_points_short_offset[:,:,0], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_shorts_map_v = cv2.resize(control_points_short_offset[:,:,1], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            off_sets_nexts_map_h = cv2.resize(control_points_nexts_offset[:,:,0], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_nexts_map_v = cv2.resize(control_points_nexts_offset[:,:,1], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            
            off_sets_prevs_map_h = cv2.resize(control_points_prevs_offset[:,:,0], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_prevs_map_v = cv2.resize(control_points_prevs_offset[:,:,1], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            
            off_sets_nexts_nexts_map_h = cv2.resize(control_points_nexts_nexts_offset[:,:,0], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_nexts_nexts_map_v = cv2.resize(control_points_nexts_nexts_offset[:,:,1], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            
            off_sets_prevs_prevs_map_h = cv2.resize(control_points_prevs_prevs_offset[:,:,0], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_prevs_prevs_map_v = cv2.resize(control_points_prevs_prevs_offset[:,:,1], (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            

            image = cv2.resize(image, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)


            end_points_off_sets_shorts_map_h *= h_scale 
            end_points_off_sets_shorts_map_v *= v_scale 

            off_sets_shorts_map_h *= h_scale 
            off_sets_shorts_map_v *= v_scale 
            off_sets_nexts_map_h *= h_scale 
            off_sets_nexts_map_v *= v_scale 
            off_sets_prevs_map_h *= h_scale 
            off_sets_prevs_map_v *= v_scale  

            off_sets_nexts_nexts_map_h *= h_scale 
            off_sets_nexts_nexts_map_v *= v_scale 
            off_sets_prevs_prevs_map_h *= h_scale 
            off_sets_prevs_prevs_map_v *= v_scale 
            
            if self.debug:
            # import pdb; pdb.set_trace()
                canvas = np.zeros_like(off_sets_prevs_map_h)
                canvas = visualize_points(canvas, end_points_map)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                combined_show = cv2.resize(combined_show, ( 768, 576), interpolation = cv2.INTER_NEAREST )
                
                cv2.imwrite('test/combined.png', combined_show) 
                print ('hello')
                #plt.imshow(combined_show)
                # cv2.waitKey()

                canvas = np.zeros_like(off_sets_prevs_map_h)
                canvas = visualize_offset(canvas, off_sets_shorts_map_h, off_sets_shorts_map_v)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                combined_show = cv2.resize(combined_show, ( 768, 576), interpolation = cv2.INTER_NEAREST )
                #plt.imshow(combined_show)
                cv2.waitKey()

                canvas = np.zeros_like(off_sets_prevs_map_h)
                canvas = visualize_offset(canvas, off_sets_prevs_map_h, off_sets_prevs_map_v)
                combined_show = draw_mask(image, canvas, [0., 255., 0.])
                combined_show = cv2.resize(combined_show, ( 768, 576), interpolation = cv2.INTER_NEAREST )

                #plt.imshow(combined_show)
                cv2.waitKey()

                canvas_2 = np.zeros_like(off_sets_prevs_map_h)
                canvas_2 = visualize_offset(canvas_2, off_sets_nexts_map_h, off_sets_nexts_map_v)
                combined_show_2 = draw_mask(image, canvas_2, [0., 255., 0.])
                combined_show_2 = cv2.resize(combined_show_2, ( 768, 576), interpolation = cv2.INTER_NEAREST )
                #plt.imshow( combined_show_2)
                cv2.waitKey()

                canvas_2 = np.zeros_like(off_sets_prevs_map_h)
                canvas_2 = visualize_offset(canvas_2, off_sets_nexts_nexts_map_h, off_sets_nexts_nexts_map_v)
                combined_show_2 = draw_mask(image, canvas_2, [0., 255., 0.])
                combined_show_2 = cv2.resize(combined_show_2, ( 768, 576), interpolation = cv2.INTER_NEAREST )
                #plt.imshow(combined_show_2)
                cv2.waitKey()

                canvas_2 = np.zeros_like(off_sets_prevs_map_h)
                canvas_2 = visualize_offset(canvas_2, off_sets_prevs_prevs_map_h, off_sets_prevs_prevs_map_v)
                combined_show_2 = draw_mask(image, canvas_2, [0., 255., 0.])
                combined_show_2 = cv2.resize(combined_show_2, ( 768, 576), interpolation = cv2.INTER_NEAREST )
                #plt.imshow(combined_show_2)
                cv2.waitKey()

            


        
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
            target15 = np.zeros((8,  int(self.out_res[0] / 8), int(self.out_res[1] / 8)))
            target11 = np.zeros((8, int(self.out_res[0] / 4), int(self.out_res[1] / 4)))
            target9 = np.zeros((8, int(self.out_res[0] / 2), int(self.out_res[1] / 2)))
            target7 = np.zeros((8, self.out_res[0], self.out_res[1]))

            # import pdb; pdb.set_trace()
            
            target15[0, :, :] = cv2.resize(end_points_map, ( int(self.out_res[1] / 8), int(self.out_res[0] /8)),interpolation = cv2.INTER_NEAREST )
            target11[0, :, :] = cv2.resize(end_points_map, ( int(self.out_res[1] / 4), int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )
            target9[0, :, :] = cv2.resize(end_points_map, ( int(self.out_res[1] / 2), int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )
            target7[0, :, :] = end_points_map
            
            target15[1, :, :] = cv2.resize(control_points_map, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST ) 
            target11[1, :, :] = cv2.resize(control_points_map, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )
            target9[1, :, :] = cv2.resize(control_points_map, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )
            target7[1, :, :] = control_points_map

            target15[2, :, :] = cv2.resize(off_sets_nexts_map_h, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST )  
            target11[2, :, :] = cv2.resize(off_sets_nexts_map_h, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )
            target9[2, :, :] = cv2.resize(off_sets_nexts_map_h, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )
            target7[2, :, :] = off_sets_nexts_map_h

            target15[3, :, :] = cv2.resize(off_sets_nexts_map_v, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST )   
            target11[3, :, :] = cv2.resize(off_sets_nexts_map_v, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )  
            target9[3, :, :] = cv2.resize(off_sets_nexts_map_v, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )  
            target7[3, :, :] = off_sets_nexts_map_v

            target15[4, :, :] = cv2.resize(off_sets_prevs_map_h, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST )   
            target11[4, :, :] = cv2.resize(off_sets_prevs_map_h, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )   
            target9[4, :, :] = cv2.resize(off_sets_prevs_map_h, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )   
            target7[4, :, :] = off_sets_prevs_map_h

            target15[5, :, :] = cv2.resize(off_sets_prevs_map_v, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST )    
            target11[5, :, :] = cv2.resize(off_sets_prevs_map_v, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )   
            target9[5, :, :] = cv2.resize(off_sets_prevs_map_v, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )   
            target7[5, :, :] = off_sets_prevs_map_v

            target15[6, :, :] = cv2.resize(off_sets_shorts_map_h, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST )    
            target11[6, :, :] = cv2.resize(off_sets_shorts_map_h, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )   
            target9[6, :, :] = cv2.resize(off_sets_shorts_map_h, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )   
            target7[6, :, :] = off_sets_shorts_map_h

            target15[7, :, :] = cv2.resize(off_sets_shorts_map_v, (int(self.out_res[1] / 8),int(self.out_res[0] /8)), interpolation = cv2.INTER_NEAREST )    
            target11[7, :, :] = cv2.resize(off_sets_shorts_map_v, (int(self.out_res[1] / 4),int(self.out_res[0] /4)), interpolation = cv2.INTER_NEAREST )   
            target9[7, :, :] = cv2.resize(off_sets_shorts_map_v, (int(self.out_res[1] / 2),int(self.out_res[0] /2)), interpolation = cv2.INTER_NEAREST )   
            target7[7, :, :] = off_sets_shorts_map_v


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
        # return 8
