import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    # img_dir = os.path.join(cur_dir,'/usa/yliu1/codeToBiomix/intersection_detection/dataset/test_pool')
    img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/mts')
    # img_dir = os.path.join(cur_dir,'/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/img')

    model = 'CPN50' # option 'CPN50', 'CPN101'

    num_class = 2
    # img_path = os.path.join(root_dir, 'data', 'COCO2017', 'val2017')
    img_path = img_dir
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    # data_shape = (256, 192)
    # output_shape = (64, 48)
    # data_shape = (608, 800)
    # output_shape = (608, 800)
    # data_shape = (448, 608)
    # output_shape = (448, 608)


    # data_shape = (576, 768)
    # output_shape = (576, 768)
##################################################################
    # data_shape = (384, 496)
    # output_shape = (192, 256)

# ##################################################################
    data_shape = (256, 256)
    # output_shape = (256, 256)
    # output_shape = (512, 512)
    # data_shape = (512, 512)
    # data_shape = (512, 1024)
    output_shape = (608, 1024)
    # output_shape = (608, 1248)
    disc_radius = 3
    PEAK_THRESH = 0.001
##################################################################
    disc_kernel_points_16 = np.ones((16,16))
    disc_kernel_points_12 = np.ones((12,12))
    disc_kernel_points_8 = np.ones((8,8))
    disc_kernel_points_3 = np.ones((3,3))


    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)

    crop_width = 25


    use_GT_bbox = False
    if use_GT_bbox:
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_val.json')
    else:
        # if False, make sure you have downloaded the val_dets.json and place it into annotation folder
          # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
          # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'testing_multi_v3.json')
        # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_short_long_offset.json')
        # gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/anno'
        #     , 'ann_keypoint_double_offset_test.json')
        # gt_path = os.path.join('/usa/yliu1/codeToBiomix/intersection_detection/dataset/', 'gt_curves_test.json')
        gt_path = os.path.join('/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'mt_test.json')


    # ori_gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
    # # ori_gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_short_long_offset.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/anno'
    #         , 'ann_keypoint_double_offset_test.json')
    # ori_gt_path = os.path.join('/usa/yliu1/codeToBiomix/intersection_detection/dataset/', 'gt_curves_test.json')
    ori_gt_path = os.path.join('/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'mt_test.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))
