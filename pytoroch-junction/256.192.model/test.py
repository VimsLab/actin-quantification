import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt

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
from dataloader.mscocoMulti_double_only import MscocoMulti_double_only
from dataloader.mscoco_backup_7_17 import mscoco_backup_7_17
from tqdm import tqdm

from utils.postprocess import resize_back_output_shape
from utils.postprocess import compute_heatmaps, get_keypoints
from utils.postprocess import group_skeletons, refine_next,group_one_skel
from utils.postprocess import fast_march
from utils.postprocess import normalize_include_neg_val

from utils.color_map import GenColorMap

from utils.preprocess import visualize_offset


import xlwt
#TO DO:
#Fix Line 75; Only get one image here, but there are two images in total
#            input_var = torch.autograd.Variable(inputs.cuda()[:1])
#
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def draw_mask_color(im, mask, color):
    mask = mask>0.02

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    # import pdb; pdb.set_trace()

    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32)*0.5
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.float32)

def draw_mask(im, mask):
    # import pdb; pdb.set_trace()
    mask = mask > 0.02

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    r[mask] = 1.0
    g[mask] = 0.0
    b[mask] = 0.0

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 1.0
    # combined = im.astype(np.float32)
    return combined.astype(np.float32)


def main(args):
    debug_show = True
    debug_show = False
    write_image = True
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained = True)
    model = torch.nn.DataParallel(model).cuda().to(device)

    # model =model.cuda()

    # img_dir = os.path.join(cur_dir,'/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/')

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti_double_only(cfg, train=False),
        batch_size=args.batch*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # load trainning weights
    checkpoint_file = os.path.join(args.checkpoint, args.test+'.pth.tar')

    checkpoint = torch.load(checkpoint_file)
    # print("info : '{}'").format(checkpoint['info'])

    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    # change to evaluation mode
    model.eval()

    print('testing...')
    full_result = []
    torch.no_grad()

    file_name = []

    final_len_skel = []
    final_num_skel = []
    final_ave_len_skel = []

    final_len_march = []
    final_num_march = []
    final_ave_len_march = []

    # for i, (inputs, targets, meta) in enumerate(test_loader):
    for i, (inputs, image_org, meta) in enumerate(test_loader):
        if i == 0:
            continue

        this_file_name = meta['imgID']
        file_name.append(this_file_name)
        image_org_ = image_org.data.cpu().numpy()[0]
            #-------------------------------------------------------------
        input_var = torch.autograd.Variable(inputs.to(device))
        input_ = input_var.data.cpu().numpy()[0]

            #-----------------------------------------------------------------


        outputs = model(input_var)
        end_point_pred, intersection_point_pred, end_points_short_offsets_pred, intersection_points_short_offsets_pred = outputs


        image_show = np.transpose(input_,(1,2,0))
        # image_show = image_org_ # * 0.6 + image_show * 0.4
        # cv2.imshow('org',image_show)
        # cv2.waitKey(0)
        # image_show = cv2.resize(image_show, )
        intersection_points_short_h = intersection_points_short_offsets_pred[:,0,:,:].cpu().detach().numpy()[0]
        intersection_points_short_v = intersection_points_short_offsets_pred[:,1,:,:].cpu().detach().numpy()[0]
        canvas = np.zeros_like(intersection_points_short_h)
        canvas = visualize_offset(canvas, intersection_points_short_h, intersection_points_short_v)

        if debug_show:
            combined_show = draw_mask(image_show, canvas)
        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )
            cv2.imshow('intersection_points_short.png',combined_show*1.0)
            cv2.waitKey(0)



        image_show = np.transpose(input_,(1,2,0))
        control_points_map_pred_h = end_points_short_offsets_pred[:,0,:,:].cpu().detach().numpy()[0]
        control_points_map_pred_v = end_points_short_offsets_pred[:,1,:,:].cpu().detach().numpy()[0]
        canvas = np.zeros_like(control_points_map_pred_h)
        canvas = visualize_offset(canvas, control_points_map_pred_h, control_points_map_pred_v)


        combined_show = draw_mask(image_show, canvas)
        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )
        if debug_show:
            cv2.imshow('end_points_short.png',combined_show)
            cv2.waitKey(0)

        mask_pred_numpy = end_point_pred[0,0,:,:].cpu().detach().numpy()
        canvas = np.zeros_like(mask_pred_numpy)

        mask_pred_numpy = (mask_pred_numpy>0.5) * 1.0

        combined_show = draw_mask(image_show, mask_pred_numpy)
        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )

        if debug_show:
            cv2.imshow('end_point.png',combined_show)
            cv2.waitKey(0)



        mask_pred_numpy = intersection_point_pred[0,0,:,:].cpu().detach().numpy()
        canvas = np.zeros_like(mask_pred_numpy)

        mask_pred_numpy = (mask_pred_numpy>0.4) * 1.0
        combined_show = draw_mask(image_show, mask_pred_numpy)

        # combined_show = image_show * 0.7 + np.tile(np.expand_dims(intersection_point_pred[0,0,:,:].cpu().detach().numpy(), -1),3) * 0.3

        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )
        if debug_show:
            cv2.imshow('intersection.png ',combined_show)
            cv2.waitKey(0)

        # import pdb; pdb.set_trace()
        canvas = np.zeros_like(mask_pred_numpy)
        heatmap_control = compute_heatmaps(mask_pred_numpy, intersection_points_short_offsets_pred[0,:,:,:].cpu().detach().numpy())
        if debug_show:
            # combined_show = image_show * 0.7 + np.tile(np.expand_dims(normalized_heatmap_control, -1),3) * 0.3
            combined_show = draw_mask(image_show, heatmap_control)
        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )
            # cv2.imshow('bla.png', combined_show * 1.0)
            cv2.waitKey(0)



        heatmap_control = gaussian_filter(heatmap_control, sigma=4)

        normalized_heatmap_control = normalize_include_neg_val(heatmap_control)



        kp_control = get_keypoints(heatmap_control)
        kp_control_intersections= kp_control
        for i in range(len(kp_control)):

            curr = (kp_control[i]['xy'][0],kp_control[i]['xy'][1])
            cv2.circle(canvas, curr, 3, 1, 1)

        # import pdb; pdb.set_trace()
        combined_show = draw_mask(image_show, canvas)
        if debug_show:

        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )
            cv2.imshow('keypoints.png', combined_show * 1.0)
            cv2.waitKey(0)

        if write_image:
            cv2.imwrite('keypoints_write.png',combined_show*255.0)

        canvas = np.zeros_like(mask_pred_numpy)
        heatmap_control = compute_heatmaps(end_point_pred[0,0,:,:].cpu().detach().numpy(), end_points_short_offsets_pred[0,:,:,:].cpu().detach().numpy())
        heatmap_control = gaussian_filter(heatmap_control, sigma=5)
        kp_control = get_keypoints(heatmap_control)

        for i in range(len(kp_control)):

            curr = (kp_control[i]['xy'][0],kp_control[i]['xy'][1])
            cv2.circle(canvas, curr, 3, 1, 1)

        # import pdb; pdb.set_trace()
        if debug_show:
            combined_show = draw_mask(image_show, canvas)
        # combined_show =  cv2.resize(combined_show, (512,512), interpolation = cv2.INTER_NEAREST )
            cv2.imshow('end_point.png', combined_show)
            cv2.waitKey(0)
        mask = image_show
        mask = 1.0 * (rgb2gray(image_show)> 0)
        # cv2.imshow('end_point', mask)
        # cv2.waitKey(0)
        # import pdb;pdb.set_trace()import time

        start_time = time.time()
        file_name_to_save = this_file_name[0][:-4]
        skel_len, num_skel, march_len, num_march = fast_march(kp_control_intersections,mask, filename = file_name_to_save, write = True)
        print("--- %s seconds ---" % (time.time() - start_time))

        final_len_skel.append(skel_len * 2)
        final_num_skel.append(num_skel)
        final_ave_len_skel.append(skel_len * 2 / num_skel)

        final_len_march.append(march_len * 2)
        final_num_march.append(num_march)
        final_ave_len_march.append(march_len * 2 / num_march)


    book = xlwt.Workbook()
    sh = book.add_sheet('Sheet 1')
    for row, this_file_name in enumerate(file_name):
        sh.write(row, 0, this_file_name)

    for row, length in enumerate(final_len_skel):
        sh.write(row, 1, float(length))

    for row, num in enumerate(final_num_skel):
        sh.write(row, 2, float(num))

    for row, ave_len in enumerate(final_ave_len_skel):
        sh.write(row, 3, float(ave_len))

    for row, length in enumerate(final_len_march):
        sh.write(row, 4, float(length))

    for row, num in enumerate(final_num_march):
        sh.write(row, 5, float(num))

    for row, ave_len_march in enumerate(final_ave_len_march):
        sh.write(row, 6, float(ave_len_march))

    book.save('test.xlsx')

    # evaluate on COCO
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=1, type=int,
                        help='test batch size (default: 32)')
    parser.add_argument('-t', '--test', default='CPN256x192', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')
    main(parser.parse_args())
