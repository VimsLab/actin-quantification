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
from dataloader.mscoco_backup_7_17 import mscoco_backup_7_17
from tqdm import tqdm

from utils.postprocess import get_keypoints, resize_back_output_shape

#TO DO:
#Fix Line 75; Only get one image here, but there are two images in total
#            input_var = torch.autograd.Variable(inputs.cuda()[:1]) 
#


def draw_mask(im, mask):
    # import pdb; pdb.set_trace()
    mask = mask > 0.5

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32) 
 
    r[mask] = 1.0
    g[mask] = 0.0 
    b[mask] = 0.0

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    # combined = im.astype(np.float32)
    return combined.astype(np.float32)


def main(args):
    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained = True)
    model = torch.nn.DataParallel(model).cuda()
    # model =model.cuda()

    # img_dir = os.path.join(cur_dir,'/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/')

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg, train=False),
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

    for i, (inputs, points_targets, targets_offset, meta) in enumerate(test_loader):
            import pdb; pdb.set_trace()
            print(i)
            #-------------------------------------------------------------
            _,_,_, points_targets_last = points_targets
            _,_,_, targets_offset_last = targets_offset

            # import pdb; pdb.set_trace()
            points_targets_last = points_targets_last.data.cpu().numpy()

            start_points_target = points_targets_last[0, 0]
            control_points_target = points_targets_last[0, 1]

            targets_offset_last = targets_offset_last.data.cpu().numpy()
            # import pdb; pdb.set_trace()
            targets_offset_h = targets_offset_last[0, 0]
            targets_offset_v = targets_offset_last[0, 1]

            #-----------------------------------------------------------------
            # input_var =inputs.cuda()[:1] #!!!!!!!!!!!!!!
            input_var = torch.autograd.Variable(inputs.cuda()[:1])    
            global_outputs, refine_output, global_offset = model(input_var)

            refine_output = refine_output.data.cpu().numpy()
            start_points_result = refine_output[0,0]
            control_points_result = refine_output[0,1]


            # cv2.imwrite("/home/yiliu/work/fiberPJ/pytorch-cpn/256.192.model/checkpoint_test_7_12/epoch17checkpoint_result/" + prefix + "_result.png", combined * 255.)
            # import pdb; pdb.set_trace()
            global_offset = global_offset[3][0].data.cpu().numpy()

            off_sets_nexts_map_h = resize_back_output_shape(global_offset[0], cfg.output_shape) 
            off_sets_nexts_map_v = resize_back_output_shape(global_offset[1], cfg.output_shape)     
            off_sets_prevs_map_h = resize_back_output_shape(global_offset[2], cfg.output_shape)     
            off_sets_prevs_map_v = resize_back_output_shape(global_offset[3], cfg.output_shape)     
            offset_short_h = resize_back_output_shape(global_offset[4], cfg.output_shape)
            offset_short_v = resize_back_output_shape(global_offset[5], cfg.output_shape) 

            # offset_h = global_offset[0] 
            # offset_v = global_offset[1]     
            # offset_short_h = global_offset[2]
            # offset_short_v = global_offset[3]  

            print('offset_hv')
            print(np.max(off_sets_nexts_map_h))     
            print(np.max(off_sets_nexts_map_v))     


            input_ = input_var.data.cpu().numpy()[0]
            input_ = input_.transpose(1,2,0)

            
            visual_heatmap = control_points_result
            start_point_heatmap = start_points_result
            # import pdb;pdb.set_trace()
            heatmaps = gaussian_filter(visual_heatmap, sigma=2)
            st_heatmaps = gaussian_filter(start_point_heatmap, sigma=2)

            canvas = np.zeros(visual_heatmap.shape)

            pred_kp = get_keypoints(heatmaps)
            pred_st_kp = get_keypoints(st_heatmaps)
            # import pdb;pdb.set_trace()
            # import pdb; pdb.set_trace()
            print ('off_sets_nexts_map')
            for x in range(0, canvas.shape[1], 10):
                for y in range(0, canvas.shape[0], 10):
                    curr = (x, y)
                    offset_x = off_sets_nexts_map_h[y, x]
                    offset_y = off_sets_nexts_map_v[y, x]
                    next_pt = (int(x + offset_x), int(y + offset_y))
                    cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

            for i in range(len(pred_st_kp)):
                curr = (pred_st_kp[i]['xy'][0],pred_st_kp[i]['xy'][1])

                cv2.circle(canvas, curr, 4, 1, -1)

                # canvas[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]] = 1
            canvas = cv2.resize(canvas, (768,576), interpolation = cv2.INTER_NEAREST )
            new_input_ =  cv2.resize(input_, (768,576), interpolation = cv2.INTER_NEAREST )
            combined = draw_mask(new_input_, canvas) 
            cv2.imshow('keypoints', combined)
            cv2.waitKey(0)
            import pdb; pdb.set_trace()
############################################################################################
            canvas = np.zeros(visual_heatmap.shape)
            # for i in range(len(pred_kp)):
                
            #     curr = (pred_kp[i]['xy'][0],pred_kp[i]['xy'][1])
            #     offset_x = offset_h[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]]
            #     offset_y = offset_v[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]]
            #     # offset_x = targets_offset_h[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]]
            #     # offset_y = targets_offset_v[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]]
            #     next_pt = (int(pred_kp[i]['xy'][0] + offset_x), int (pred_kp[i]['xy'][1] + offset_y))
            #     cv2.arrowedLine(canvas, curr, next_pt, 1, 1)
            print ('off_sets_nexts_map')
            for x in range(0, canvas.shape[1], 10):
                for y in range(0, canvas.shape[0], 10):
                    curr = (x, y)
                    offset_x = off_sets_prevs_map_h[y, x]
                    offset_y = off_sets_prevs_map_v[y, x]
                    next_pt = (int(x + offset_x), int(y + offset_y))
                    cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

            for i in range(len(pred_st_kp)):
                curr = (pred_st_kp[i]['xy'][0],pred_st_kp[i]['xy'][1])

                cv2.circle(canvas, curr, 3, 1, -1)

                # canvas[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]] = 1
            canvas = cv2.resize(canvas, (768,576), interpolation = cv2.INTER_NEAREST )
            new_input_ =  cv2.resize(input_, (768,576), interpolation = cv2.INTER_NEAREST )
            combined = draw_mask(new_input_, canvas) 
            cv2.imshow('keypoints', combined)
            cv2.waitKey(0)
############################################################################################
            canvas = np.zeros(visual_heatmap.shape)
            print ('short_offset_map')
            for x in range(0, canvas.shape[1], 5):
                for y in range(0, canvas.shape[0], 5):
                    curr = (x, y)
                    offset_x = offset_short_h[y, x]
                    offset_y = offset_short_v[y, x]
                    next_pt = (int(x + offset_x), int(y + offset_y))
                    cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

            for i in range(len(pred_st_kp)):
                curr = (pred_st_kp[i]['xy'][0],pred_st_kp[i]['xy'][1])

                cv2.circle(canvas, curr, 3, 1, -1)

                # canvas[pred_kp[i]['xy'][1],pred_kp[i]['xy'][0]] = 1
            canvas = cv2.resize(canvas, (768,576), interpolation = cv2.INTER_NEAREST )
            new_input_ =  cv2.resize(input_, (768,576), interpolation = cv2.INTER_NEAREST )
            combined = draw_mask(new_input_, canvas) 
            cv2.imshow('keypoints', combined)
            cv2.waitKey(0)
#########################################################################################
            import pdb; pdb.set_trace()
            visual_heatmap = start_points_result
            combined = draw_mask(input_, visual_heatmap) 
            # start_point_compare = np.hstack((visual_heatmap, start_points_target))
            cv2.imshow('test', combined )
                # cv2.waitKey(0)

            cv2.waitKey(0)
            import pdb; pdb.set_trace()
            print ('control points')
            visual_heatmap = control_points_result
                # import pdb; pdb.set_trace()     
            combined = draw_mask(input_, visual_heatmap) 
            # compare_horizontal = np.hstack((visual_heatmap, control_points_target))
            # cv2.imshow('test', compare_horizontal )
            #     # cv2.waitKey(0)
            cv2.imshow('ex', combined)
            cv2.waitKey(0)

            # # import pdb; pdb.set_trace()

            # print ('offset_h points')
            # visual_heatmap = np.absolute(offset_h)
            # target_offset = np.absolute(targets_offset_h)
            # # import pdb; pdb.set_trace()     
            # # combined = draw_mask(input_, visual_heatmap) 
            # compare_horizontal = np.hstack((visual_heatmap, target_offset))
            # cv2.imshow('test', compare_horizontal )

            # cv2.waitKey(0)

            # print ('offset_v points')
            # visual_heatmap = np.absolute(offset_v)
            # target_offset = np.absolute(targets_offset_v)
            # # import pdb; pdb.set_trace()     
            # # combined = draw_mask(input_, visual_heatmap) 
            # compare_horizontal = np.hstack((visual_heatmap, target_offset))
            # cv2.imshow('test', compare_horizontal )

            # cv2.waitKey(0)
 ####################################################################################
            # import pdb; pdb.set_trace()


      
            # visual_heatmap = score_map[0,0,:,:]
            # # import pdb; pdb.set_trace()
            # combined = draw_mask(input_, visual_heatmap)

            # compare_horizontal = np.hstack((visual_heatmap, target_last[0,:,:]))
            # cv2.imshow('test', compare_horizontal )
            # cv2.imshow('ex', combined)
            # image_path = meta['img_path'][:1][0]
            # prefix = os.path.splitext(image_path)
            # prefix = prefix[0].split('/')[-1]
            # # import pdb; pdb.set_trace()            

            # #cv2.imwrite("/home/yiliu/work/fiberPJ/pytorch-cpn/256.192.model/checkpoint_test_7_12/epoch17checkpoint_result/" + prefix + "_result.png", combined * 255.)

            # cv2.waitKey(0)
            # del global_outputs, refine_output
            # import pdb; pdb.set_trace()

            # if args.flip == True:
            #     flip_global_outputs, flip_output = model(flip_input_var)
            #     flip_score_map = flip_output.data.cpu()
            #     flip_score_map = flip_score_map.numpy()

            #     for i, fscore in enumerate(flip_score_map):
            #         fscore = fscore.transpose((1,2,0))
            #         fscore = cv2.flip(fscore, 1)
            #         fscore = list(fscore.transpose((2,0,1)))
            #         for (q, w) in cfg.symmetry:
            #            fscore[q], fscore[w] = fscore[w], fscore[q] 
            #         fscore = np.array(fscore)
            #         score_map[i] += fscore
            #         score_map[i] /= 2

    #         ids = meta['imgID'].numpy()
    #         det_scores = meta['det_scores']
    #         for b in range(inputs.size(0)):
    #             details = meta['augmentation_details']
    #             single_result_dict = {}
    #             single_result = []
                
    #             single_map = score_map[b]
    #             r0 = single_map.copy()
    #             r0 /= 255
    #             r0 += 0.5
    #             v_score = np.zeros(17)
    #             for p in range(17): 
    #                 single_map[p] /= np.amax(single_map[p])
    #                 border = 10
    #                 dr = np.zeros((cfg.output_shape[0] + 2*border, cfg.output_shape[1]+2*border))
    #                 dr[border:-border, border:-border] = single_map[p].copy()
    #                 dr = cv2.GaussianBlur(dr, (21, 21), 0)
    #                 lb = dr.argmax()
    #                 y, x = np.unravel_index(lb, dr.shape)
    #                 dr[y, x] = 0
    #                 lb = dr.argmax()
    #                 py, px = np.unravel_index(lb, dr.shape)
    #                 y -= border
    #                 x -= border
    #                 py -= border + y
    #                 px -= border + x
    #                 ln = (px ** 2 + py ** 2) ** 0.5
    #                 delta = 0.25
    #                 if ln > 1e-3:
    #                     x += delta * px / ln
    #                     y += delta * py / ln
    #                 x = max(0, min(x, cfg.output_shape[1] - 1))
    #                 y = max(0, min(y, cfg.output_shape[0] - 1))
    #                 resy = float((4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
    #                 resx = float((4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
    #                 v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
    #                 single_result.append(resx)
    #                 single_result.append(resy)
    #                 single_result.append(1)   
    #             if len(single_result) != 0:
    #                 single_result_dict['image_id'] = int(ids[b])
    #                 single_result_dict['category_id'] = 1
    #                 single_result_dict['keypoints'] = single_result
    #                 single_result_dict['score'] = float(det_scores[b])*v_score.mean()
    #                 full_result.append(single_result_dict)

    # result_path = args.result
    # if not isdir(result_path):
    #     mkdir_p(result_path)
    # result_file = os.path.join(result_path, 'result.json')
    # with open(result_file,'w') as wf:
    #     json.dump(full_result, wf)

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