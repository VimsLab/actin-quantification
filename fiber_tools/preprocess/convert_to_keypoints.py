import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import scipy.ndimage as ndimage
import math
import skfmm

from pycocotools.coco import COCO
from fiber_tools.common.util.color_map import GenColorMap 
from fiber_tools.common.util.cv2_util import pologons_to_mask, mask_to_pologons
from get_skeleton_intersection import get_skeleton_intersection_and_endpoint, get_skeleton_endpoint


from skimage.morphology import skeletonize
from least_square import least_squares

from tqdm import tqdm
import json
from fiber_tools.common.util.osutils import isfile

def get_angle(x, y):
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle_radius=np.arccos(cos_angle)
    angle_degree=angle_radius*360/2/np.pi
    return angle_degree

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

def get_keypoints(img, true_image, step, crop_edge, debug):

    start_points = []
    control_points = []
    off_sets = []
    off_sets_short = []
#################
    skel = skeletonize(img)
    skel = skel.astype(np.uint8)
    skel = skel[crop_edge:skel.shape[0] - crop_edge, crop_edge : skel.shape[1] - crop_edge]
    true_image = true_image[crop_edge:true_image.shape[0] - crop_edge, crop_edge : true_image.shape[1] - crop_edge, :] 
    canvas = np.zeros(skel.shape)
    intersections, _ = get_skeleton_intersection_and_endpoint(skel)

    # center = (endpoints[0][1], endpoints[0][0])
    # distance = geodesic_distance_transform(skel,center)
    # cv2.imshow('t', distance)
    # cv2.waitKey(0)
    import pdb; pdb.set_trace()

    # Prune branches caused by skeletonization
    if len(intersections):
        # print('more than one intersections')
        # seperate the branches as individual contours.
        for p in intersections:
            cv2.circle(skel, p , 2, (0), -1)
        import pdb; pdb.set_trace()     
        # find contours
        im2, contours, hierarchy = cv2.findContours(skel,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        # find the longest contour.
        longest = 0
        ########################################################################
        #canvas_debug = np.zeros(skel.shape)
        #######################################################################
        for idx, contour in enumerate(contours):
            #####################################################################
            # canvas_debug = cv2.drawContours(canvas_debug, contours, idx, 1, 1)
            # cv2.imshow('t', canvas_debug)
            # cv2.waitKey(0)
            #######################################################################
            if cv2.arcLength(contour, False) > cv2.arcLength(contours[longest], False):
                longest = idx

        longest_contour = contours[longest]

        reset_contour_canvas = np.zeros(skel.shape)
        reset_contour_canvas = cv2.drawContours(reset_contour_canvas, contours, longest, 1, 1)
        
        # cv2.imshow('t', reset_contour_canvas)
        # cv2.waitKey(0)
        
        endpoints = get_skeleton_endpoint(reset_contour_canvas)
        try:
            stpt = endpoints[0] #start point default
        except:
            return start_points, off_sets_short, control_points, off_sets
            #return index in cv2 format. (x, y) / (col, row)
        # import pdb; pdb.set_trace()
        stpt = endpoints[0] #start point default
        #left most and down most point as start point
        for ept in endpoints:
            if (ept[0] < stpt[0]):
                stpt = ept
            elif(ept[0] == stpt[0]):
                if(ept[1] < stpt[1]):
                    stpt = ept


        #swap: start from start point
        cut_point_x = np.where(longest_contour[:,0,0] == stpt[0])
        cut_point_y = np.where(longest_contour[:,0,1] == stpt[1])
        cut_point = cut_point_x and cut_point_y
        # import pdb;pdb.set_trace()
        cut_point = cut_point[0][0]
        part1 = longest_contour[cut_point:]
        part2 = longest_contour[:cut_point]
        longest_contour = np.concatenate((part1, part2))
    
        longest_contour = longest_contour[0 : int(len(longest_contour)/2): step]
        # longest_contour = new_contours[0]
        # start_points.append(longest_contour[0][0][0]) 
        # start_points.append(longest_contour[0][0][1])
        if debug:
            cv2.circle(canvas, (longest_contour[0][0][0],longest_contour[0][0][1]) , 3, 1, -1)
        
        interval_2 = 2 
        interval_1 = 1
        if (len(longest_contour)> (2 * interval_2 + 1)):
            for pt in range(len(longest_contour)):
                if pt == 0:
                    start_points.append(longest_contour[pt][0][0]) 
                    start_points.append(longest_contour[pt][0][1])

                    if debug:                    
                        cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                        combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                        cv2.imshow('t',combined_show)
                        cv2.waitKey(0)
                if pt >= interval_1 and pt <= (len(longest_contour) - interval_1 - 1):
                    curr_dir = longest_contour[pt][0]-longest_contour[pt - interval_1][0]
                    further_dir = longest_contour[pt + interval_1][0]-longest_contour[pt][0]
                    angle = get_angle(curr_dir, further_dir)
                    if (angle > 25):
                        if debug:
                            print ('interval_1')
                            print (angle)
                        # import pdb; pdb.set_trace()
                        control_points.append(longest_contour[pt][0][0])
                        control_points.append(longest_contour[pt][0][1])
                        off_sets.append(longest_contour[pt][0][0] - longest_contour[0][0][0])
                        off_sets.append(longest_contour[pt][0][1] - longest_contour[0][0][1])
                        if debug:
                            cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                            combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                            cv2.imshow('t',combined_show)
                            cv2.waitKey(0)
                        continue
                if pt >= interval_2 and pt <= (len(longest_contour) - interval_2 - 1):
                    curr_dir = longest_contour[pt][0]-longest_contour[pt - interval_2][0]
                    further_dir = longest_contour[pt + interval_2][0]-longest_contour[pt][0]
                    angle = get_angle(curr_dir, further_dir)
                    if (angle > 15):
                        if debug:
                            print ('interval_2')
                            print (angle)
                        # import pdb; pdb.set_trace()
                        control_points.append(longest_contour[pt][0][0])
                        control_points.append(longest_contour[pt][0][1])
                        off_sets.append(longest_contour[pt][0][0] - longest_contour[0][0][0])
                        off_sets.append(longest_contour[pt][0][1] - longest_contour[0][0][1])
                        if debug:
                            cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                            combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                            cv2.imshow('t',combined_show)
                            cv2.waitKey(0)
                        continue
                if pt == len(longest_contour) - 1:
                    control_points.append(longest_contour[pt][0][0])
                    control_points.append(longest_contour[pt][0][1])
                    off_sets.append(longest_contour[pt][0][0] - longest_contour[0][0][0])
                    off_sets.append(longest_contour[pt][0][1] - longest_contour[0][0][1])
                    if debug:
                        cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                        combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                        cv2.imshow('t',combined_show)
                        cv2.waitKey(0)

        for pt in range(0, len(control_points), 2):
            if pt == 0:
                off_sets_short.append(control_points[pt] -  longest_contour[0][0][0])
                off_sets_short.append(control_points[pt + 1] -  longest_contour[0][0][1])
            else:
                off_sets_short.append(control_points[pt] - control_points[pt - 2])
                off_sets_short.append(control_points[pt + 1] - control_points[pt + 1 - 2])

            curr = (control_points[pt], control_points[pt + 1])
            next_pt = (control_points[pt] - off_sets[pt], control_points[pt + 1] - off_sets[pt + 1])

        # cv2.drawContours(canvas, contours, longest, 1, kernelSize)
        if debug:
            for pt in range(0, len(control_points), 2):

                curr = (control_points[pt], control_points[pt + 1])
                next_pt = (control_points[pt] - off_sets_short[pt], control_points[pt + 1] - off_sets_short[pt + 1])

                cv2.arrowedLine(canvas, curr, next_pt, 1, 3)
            combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
            cv2.imshow('t',combined_show)
            cv2.waitKey(0)     

        skel = canvas

    else:
        # find contours
        im2, contours, hierarchy = cv2.findContours(skel,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # canvas_debug = np.zeros(skel.shape)
        # canvas_debug = cv2.drawContours(canvas_debug, contours, 0, 1, 1)
        # cv2.imshow('t', canvas_debug)
        # cv2.waitKey(0)

        if len(contours) > 0:
            #################################################################
            # new_contour_arr = [tuple(row) for row in contours[0]] #remove duplicates 
            # uniques = np.unique(new_contour_arr, axis = 0)
            ################################################################
       
            reset_contour_canvas = np.zeros(skel.shape)
            reset_contour_canvas = cv2.drawContours(reset_contour_canvas, contours, 0, 1, 1)
            endpoints = get_skeleton_endpoint(reset_contour_canvas)
                #return index in cv2 format. (x, y) / (col, row)
            # import pdb; pdb.set_trace()
            try:
                stpt = endpoints[0] #start point default
            except:
                return start_points, off_sets_short, control_points, off_sets
            #left most and down most point as start point
            for ept in endpoints:
                if (ept[0] < stpt[0]):
                    stpt = ept
                elif(ept[0] == stpt[0]):
                    if(ept[1] < stpt[1]):
                        stpt = ept
    


            longest_contour = contours[0]

            #swap: start from start point

            cut_point_x = np.where(longest_contour[:,0,0] == stpt[0])
            cut_point_y = np.where(longest_contour[:,0,1] == stpt[1])

            cut_point = cut_point_x and cut_point_y
            cut_point = cut_point[0][0]
            part1 = longest_contour[cut_point:]
            part2 = longest_contour[:cut_point]

            longest_contour = np.concatenate((part1, part2))
        

            longest_contour = longest_contour[0 : int(len(longest_contour)/2): step]


            # longest_contour = new_contours[0]
            # start_points.append(longest_contour[0][0][0]) 
            # start_points.append(longest_contour[0][0][1])
            if debug:
                cv2.circle(canvas, (longest_contour[0][0][0],longest_contour[0][0][1]) , 3, 1, -1)
            
            interval_2 = 2 
            interval_1 = 1

            if (len(longest_contour)> (2 * interval_2 + 1)):
                for pt in range(len(longest_contour)):
                    if pt == 0:
                        start_points.append(longest_contour[pt][0][0]) 
                        start_points.append(longest_contour[pt][0][1])
                        

                        if debug:
                            cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                            combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                            cv2.imshow('t',combined_show)
                            cv2.waitKey(0)

                    if pt >= interval_1 and pt <= (len(longest_contour) - interval_1 - 1):
                        curr_dir = longest_contour[pt][0]-longest_contour[pt - interval_1][0]
                        further_dir = longest_contour[pt + interval_1][0]-longest_contour[pt][0]
                        angle = get_angle(curr_dir, further_dir)
                        # print (angle)
                        if (angle > 25):
                            if debug:
                                print ('interval_1')
                                print (angle)
                            # import pdb; pdb.set_trace()
                            control_points.append(longest_contour[pt][0][0])
                            control_points.append(longest_contour[pt][0][1])

                            off_sets.append(longest_contour[pt][0][0] - longest_contour[0][0][0])
                            off_sets.append(longest_contour[pt][0][1] - longest_contour[0][0][1])
                            if debug:
                                cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                                combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                                cv2.imshow('t',combined_show)
                                cv2.waitKey(0)
                            continue
                    if pt >= interval_2 and pt <= (len(longest_contour) - interval_2 - 1):
                        curr_dir = longest_contour[pt][0]-longest_contour[pt - interval_2][0]
                        further_dir = longest_contour[pt + interval_2][0]-longest_contour[pt][0]
                        angle = get_angle(curr_dir, further_dir)
                        # print (angle)
                        if (angle > 15):
                            if debug:
                                print ('interval_2')
                                print (angle)
                            # import pdb; pdb.set_trace()
                            control_points.append(longest_contour[pt][0][0])
                            control_points.append(longest_contour[pt][0][1])
                            off_sets.append(longest_contour[pt][0][0] - longest_contour[0][0][0])
                            off_sets.append(longest_contour[pt][0][1] - longest_contour[0][0][1])

                            if debug:
                                cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                                combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                                cv2.imshow('t',combined_show)
                                cv2.waitKey(0)
                            continue
                    if pt == len(longest_contour) - 1:
                        control_points.append(longest_contour[pt][0][0])
                        control_points.append(longest_contour[pt][0][1])
                        off_sets.append(longest_contour[pt][0][0] - longest_contour[0][0][0])
                        off_sets.append(longest_contour[pt][0][1] - longest_contour[0][0][1])
                        if debug:
                            cv2.circle(canvas, (longest_contour[pt][0][0],longest_contour[pt][0][1]) , 2, 1, -1)
                            combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                            cv2.imshow('t',combined_show)
                            cv2.waitKey(0)
        
            for pt in range(0, len(control_points), 2):
                if pt == 0:
                    off_sets_short.append(control_points[pt] -  longest_contour[0][0][0])
                    off_sets_short.append(control_points[pt + 1] -  longest_contour[0][0][1])
                else:
                    off_sets_short.append(control_points[pt] - control_points[pt - 2])
                    off_sets_short.append(control_points[pt + 1] - control_points[pt + 1 - 2])

                curr = (control_points[pt], control_points[pt + 1])
                next_pt = (control_points[pt] - off_sets[pt], control_points[pt + 1] - off_sets[pt + 1])
                
                # if debug:
                    # cv2.arrowedLine(canvas, curr, next_pt, 1, 3)
            # combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
            # cv2.imshow('t',combined_show)
            # cv2.waitKey(0)   

            if debug:
                for pt in range(0, len(control_points), 2):

                    curr = (control_points[pt], control_points[pt + 1])
                    next_pt = (control_points[pt] - off_sets_short[pt], control_points[pt + 1] - off_sets_short[pt + 1])

                    cv2.arrowedLine(canvas, curr, next_pt, 1, 3)
                combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
                cv2.imshow('t',combined_show)
                cv2.waitKey(0)            
          
            skel = canvas
        else:
            skel = canvas
    
    return start_points, off_sets_short, control_points, off_sets

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

        # for img_id in tqdm(coco_ids):
        for img_id in tqdm(coco_ids):
            img_ok = True
            img = coco_fiber.loadImgs(img_id)[0]
            file_path = os.path.join(img_root, img['file_name'])

            this_image = cv2.imread(file_path)
            img_shape = this_image.shape[:-1]

            annIds = coco_fiber.getAnnIds(imgIds=img['id'], catIds=catIds)
            anns = coco_fiber.loadAnns(annIds)

            
            start_points = []
            control_points = []
            off_sets = []
            off_sets_shorts = []

            for idx, ann in enumerate(anns):
                mask = pologons_to_mask(ann['segmentation'], img_shape)

                start_point, off_sets_short, control_point, off_set= get_keypoints(mask, this_image, step = 20, crop_edge = 25, debug = True)
                start_points += start_point
                off_sets_shorts += off_sets_short
                control_points += control_point
                off_sets+= off_set

            for i in range(len(start_points)):
                start_points[i] = int(start_points[i])
                # start_points_offsets[i] = int(start_points_offsets[i])
            
            for i in range(len(control_points)):
                control_points[i] = int(control_points[i])
                off_sets[i] = int(off_sets[i])
                off_sets_shorts[i] = int(off_sets_shorts[i])
            # import pdb; pdb.set_trace()
            file_name = img['file_name']

            single_data = {}
            unit = {}
            ###########################
            unit['start_points'] = start_points
            unit['off_sets_shorts'] = off_sets_shorts
            unit['control_points'] = control_points
            unit['off_sets'] = off_sets

            single_data ['unit'] = unit


            imgInfo ={}
            imgInfo ['imgID'] = img_id
            imgInfo ['img_name'] = file_name
            imgInfo ['file_path'] = file_path

            single_data ['imgInfo'] = imgInfo
            
            if img_ok:
                train_data.append(single_data)
                num_good_images = num_good_images + 1 
                # print(num_good_images)
            else:
                print(num_bad_images)

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
#################################################################################################
    folders = [x[0] for x in os.walk('//opt/intern/users/yiliu/instance_labeled_file/')] 
    for i in range(2, len(folders),3):
        img_root = folders[i]
        anno_root = folders[i + 1] + '/'
        output_root = anno_root 

        ori_file = 'ann_common.json'
        target_file = 'ann_keypoint_short_long_offset_test.json'
        # import pdb; pdb.set_trace()

        trans_anno(img_root, anno_root, output_root, ori_file, target_file)

# #######################################################
    # root = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file'
    # # coco_anno_file = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/common_instance_json_format/fb_coco_style_fifth.json'
    # coco_anno_file = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/fb_coco_style_fifth.json'
    # # coco_anno_file = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/skel_5_coco_style_fifth.json'
    # # show_anno(root, coco_anno_file)
    # anno_root = '/home/yiliu/work/fiberPJ/data/fiber_labeled_data/'
    # output_root = anno_root
    # img_root = root
    # ori_file = 'fb_coco_style_fifth.json'
    # # target_file = 'skel_1_non_coco_fifth.json'
    # target_file = 'ann_keypoint_short_long_offset.json'

    
    # trans_anno(img_root, anno_root, output_root, ori_file, target_file)
    

