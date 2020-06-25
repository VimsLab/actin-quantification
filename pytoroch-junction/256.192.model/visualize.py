import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import numpy as np

from test_config import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from tqdm import tqdm
from colorMap import GenColorMap


def main():
    annoType = ['segm', 'bbox', 'keypoints']
    annoType = annoType[1]
    prefix = 'person_keypoints' if annoType=='keypoints' else 'instances'
    print('Running demo for *%s* results.'%(annoType))
    
    dataDir = '//opt/opendata_all/mscoco/coco/coco_val2017'
    dataType = 'val2017'
    annoFile = 'result/result.json'
    print (annoFile)
    eval_gt = COCO(cfg.ori_gt_path)
    result = eval_gt.loadRes(annoFile)
    
    imgIds=sorted(eval_gt.getImgIds())
    imgIds=imgIds[0:100]
    ids = list(eval_gt.imgs.keys())

    for img_id in ids:
        file_path = eval_gt.loadImgs(img_id) [0]['file_name']
        file_path = os.path.join(dataDir, file_path)
        ann_ids = result.getAnnIds(imgIds=img_id)
        im = cv2.imread(file_path)
        anns = result.loadAnns(ann_ids)

        # cv2.imshow('test',im)
        # print('------------------------')
        # cv2.waitKey(0)
        im_h, im_w, _ = im.shape

        canvas = np.zeros(im.shape, dtype = np.float32) 
        for idx, ann in enumerate(anns):
            CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)
            color = CLASS_2_COLOR[idx + 1]
            bbox = ann['bbox']
            keypoints = ann['keypoints']

            for p in range(0, len(keypoints), 3):
                # print (keypoints[p])
                # print (keypoints[p+1])
                # print (keypoints[p+2])
                cv2.circle(im, (int(keypoints[p]), int(keypoints[p + 1])), 5, color, -1)
            category_id = ann['category_id']


            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            

            # mask = pologons_to_mask(ann['segmentation'],im.shape[:-1])
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(im, str(category_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            
        im = cv2.resize(im,(int(im_w), int(im_h)),interpolation=cv2.INTER_CUBIC)
        im_nobox = cv2.resize(canvas,(int(im_w), int(im_h)),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('test',im)
        print('------------------------')
        cv2.waitKey(0)

      

if __name__ == '__main__':

    main()