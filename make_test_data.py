from matplotlib.path import Path

import scipy.stats as st
import matplotlib.patches as patches
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import json

from skimage.morphology import skeletonize
from PIL import Image, ImageDraw
from tqdm import tqdm
from get_intersection_and_endpoint import get_skeleton_endpoint, get_skeleton_intersection_and_endpoint, get_skeleton_intersection


def main():
    gt_file = os.path.join('./dataset/', 'mt_test.json')

    directory_curve_pool = './dataset/mts'

    if not os.path.exists(directory_curve_pool):
        os.makedirs(directory_curve_pool)

    files = os.listdir(directory_curve_pool)
    train_data = []
    for i in tqdm(range(len(files))):
        single_data = {}
        img_info = {}

        instances = []

        img_info ['file_name'] = files[i]
        img_info ['file_path'] = directory_curve_pool

        single_data ['img_info'] = img_info

        train_data.append(single_data)


    print('saving transformed annotation...')
    with open(gt_file,'w') as wf:
        json.dump(train_data, wf)
        print('done')


if __name__ == '__main__':
    main()
