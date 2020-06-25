import os
import os.path
import sys
import numpy as np
import cv2
import torch

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    # img_dir = os.path.join(cur_dir,'/usa/yliu1/codeToBiomix/intersection_detection/dataset/curves_pool')
    # img_dir = os.path.join(cur_dir,'/usa/yliu1/codeToBiomix/intersection_detection/dataset/curves_pool_di_5')
    img_dir = os.path.join(cur_dir,'/usa/yliu1/codeToBiomix/intersection_detection/dataset/curves_pool_t_junction_zigzag_di_5')

    model = 'CPN50'

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6,40,6))

    weight_decay = 1e-5

    num_class = 2
    # img_path = os.path.join(root_dir, 'data', 'COCO2017', 'train2017')
    img_path = img_dir
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    # data augmentation setting
    scale_factor=(0.7, 1.35)
    rot_factor=45

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
#########################################################################################
    # tensorboard_path = './runs/intersection_detection'
    # # tensorboard_path = './runs/intersection_detection_weights_4_2_1'
    # batch_size = 1
    # data_shape = (256, 256)
    # output_shape = (256, 256)
    # disc_radius = 6
    # info = 'Training with data_shape =(256, 256) output_shape = (256, 256) No batch Norm last layer'
###################################################################################

    # tensorboard_path = './runs/intersection_detection_weights_4_2_1_disc_3'
    # batch_size = 1
    # data_shape = (256, 256)
    # output_shape = (256, 256)
    # disc_radius = 3
    # info = 'Training with data_shape =(256, 256) output_shape = (256, 256) No batch Norm last layer'
    # #############################################################################

    tensorboard_path = './runs/intersection_detection_weights_4_2_1_disc_5'
    batch_size = 64
    # data_shape = (128, 128)
    data_shape = (256, 256)
    output_shape = (256, 256)
    disc_radius = 10
    info = 'Training with data_shape =(128, 128) output_shape = (128, 128) No batch Norm last layer'
    #############################################################################

    crop_width = 0

    # disc_radius = 12

    gaussain_kernel = (7, 7)

    # gt_path = os.path.join('/usa/yliu1/codeToBiomix/intersection_detection/dataset/', 'gt_curves.json')
    # gt_path = os.path.join('/usa/yliu1/codeToBiomix/intersection_detection/dataset/', 'gt_curves_di_5_test.json')
    # gt_path = os.path.join('/usa/yliu1/codeToBiomix/intersection_detection/dataset/', 'gt_curves_t_junction_di_5_test.json')
    gt_path = os.path.join('/usa/yliu1/codeToBiomix/intersection_detection/dataset/', 'curves_pool_t_junction_zigzag_di_5.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_short_long_offset.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_3_non_coco_fifth.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_1_non_coco_fifth.json')

cfg = Config()
add_pypath(cfg.root_dir)

