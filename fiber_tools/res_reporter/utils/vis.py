from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import math

import numpy as np
import cv2
import random
import pdb
import os

def mkdir(tmp_dir):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

def get_file_saved_name(prefix_dir, image_path, 
    pred_name, target_name, vis_type):
    elements = image_path.split('/')
    image_name = elements[-1]
    tmp_dir = os.path.join(prefix_dir, elements[-3]+'_'+elements[-2], vis_type, 
        target_name, pred_name)
    mkdir(tmp_dir)
    output_name = os.path.join(tmp_dir, image_name)

    return output_name

def trans_img(imageData, image_len=448):
    cord = imageData[:,:,0].nonzero()

    x_cord = cord[0]
    x_min = min(x_cord)
    x_max = max(x_cord)
    x_max = min(x_min+image_len,x_max)

    y_cord = cord[1]
    y_min = min(y_cord)
    y_max = max(y_cord)
    y_max = min(y_min+image_len,y_max)

    res_img = []
    for i in range(3):
        crop_img = imageData[x_min:x_max, y_min:y_max, i]
        x_pad = (image_len-(x_max-x_min))//2+1
        y_pad = (image_len-(y_max-y_min))//2+1
        tmp_img = np.pad(crop_img,((x_pad,x_pad),(y_pad,y_pad)),'constant', constant_values=(0,0))
        res_img.append(tmp_img[:,:,None])
    res_img = np.concatenate(res_img, axis = 2)
    res_img = res_img[:image_len,:image_len,:]
    
    return res_img

def get_cam(img, cam, cam_weight=0.3, img_weight=0.5):
    cam = cam - cam.min()
    cam = ((cam / cam.max()) * 255).astype(np.uint8)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam[np.where(cam > 255)] = 255
    cam = cam.astype(np.uint8)
    img_cam = cam * cam_weight + img * img_weight
    return img_cam

def vis_imgs(file=None, prefix=None, cams=None, 
    predict=None, gt=None, labelmap=None):
    for idx,(f,p,g) in enumerate(zip(file, predict, gt)):
        pred_name = labelmap[p]
        target_name = labelmap[g]
        out_path = get_file_saved_name(prefix, f, 
            pred_name, target_name, vis_type='origin')
        img = cv2.imread(f)
        img = trans_img(img)
        cv2.imwrite(out_path, img)
        if cams is not None:
            img_cam = get_cam(img, cams[idx])
            out_path = get_file_saved_name(prefix, f, 
                pred_name, target_name, vis_type='cam')
            cv2.imwrite(out_path, img_cam)
        

    
