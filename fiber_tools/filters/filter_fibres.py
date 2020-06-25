# -*-coding:utf-8-*-
import sys
sys.path.insert(0, '../../')

import os
import shutil
import cv2
import numpy as np
import multiprocessing
from filter import SizeFilter,BlurFilter,StripeFilter
from fiber_tools.common.util.init_util import parse 

def filter(line, filter_list, output_file):
    img_path, label = line.strip().split(',')
    to_be_filter = False
    for filter_func in filter_list:
        if filter_func(img_path):
            print('filter', img_path)
            return (None, label)
    
    return (line, int(label))
    
def write_file(line):
    if line[0] is not None:
        w.write(line[0])
        print(line[0].strip())
        w.flush()
        img_number[line[1]]+=1

if __name__ == "__main__":
    # python3 filter_fibres.py --cfg ../config/filter_fibers.yml
    cfg = parse()
    input_file = cfg.FILTER.INPUT
    output_file = cfg.FILTER.OUTPUT
    size_thres = cfg.FILTER.SIZE_THRES
    blur_thres = cfg.FILTER.BLUR_THRES
    stripe_blur_thres = cfg.FILTER.STRIPE_BLUR_THRES
    side_blur_thres = cfg.FILTER.STRIPE_SIDE_THRES

    filter_list = []
    if size_thres>0:
        filter_list.append(SizeFilter(size_thres))

    if blur_thres>0:
        filter_list.append(BlurFilter(blur_thres))
    
    if (stripe_blur_thres>0 and side_blur_thres>0):
        filter_list.append(StripeFilter(stripe_blur_thres, 
            side_blur_thres)) 
    
    w = open(output_file, 'w')
    pool = multiprocessing.Pool(10)
    img_number = [0]*5
    with open(input_file) as lines:
        for line in lines:
            pool.apply_async(filter, (line, filter_list, output_file),
                callback = write_file)
            # res = filter(line, filter_list, output_file)
            # write_file(res)
    pool.close()
    pool.join()
    w.close()
