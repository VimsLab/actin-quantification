import numpy as np

from container.score_container import ScoreContainer
import utils.metric as metric
from utils.process import softmax, label_map
from utils.vis import vis_imgs, mkdir

import sys
sys.path.insert(0, '../../')
from fiber_tools.common.util.init_util import parse 
from fiber_tools.common.util.vis_util import save_to_html 
from fiber_tools.common.util.time_util import get_cur_time 

import random

import pdb

class ClsReporter:
    def __init__(self):

        with open(cfg.CLS.GT) as f:
            index = []
            gt = []
            for x in f:
                index.append(x.strip().split(',', 1)[0])
                gt.append([int(x.strip().split(',')[-1])])
            gt = np.array(gt, dtype=np.int32)
            self.container = ScoreContainer(index)
            self.container.set('gt', gt)

        if len(cfg.CLS.FC) > 0:
            out = np.load(cfg.CLS.FC)
            self.container.set('out', out) 

        if len(cfg.CLS.CAM.PREDICT) > 0:
            predict_cam = np.load(cfg.CLS.CAM.PREDICT)
            self.container.set('predict_cam', predict_cam)

        if len(cfg.CLS.CAM.GT) > 0:
            gt_cam = np.load(cfg.CLS.CAM.GT)
            self.container.set('gt_cam', gt_cam)

    def set(self, name, value):
        self.container.set(name, value)

    def topk(self):
        self.container.map('out', 'softmax', softmax)
        self.container.map('out', 'argsort', lambda x : np.argsort(x, axis=0)[::-1])

        gt = self.container.get('gt').flatten()
        sort = self.container.get('argsort')
        for i in cfg.CLS.TOPK.K:
            acc = metric.topk(i, gt, sort)
            print('top {}: '.format(i))
            print(acc)

    def confusion_matrix(self):
        gt = self.container.get('gt').flatten()
        self.container.map('out', 'argmax', lambda x : np.argmax(x, axis=0))
        predict = self.container.get('argmax').flatten()
        matrix = metric.confusion(gt, predict)

        print('Confusion Matrix:')
        print('    out')
        print('gt')
        output_dir = cfg.CLS.OUTPUT_DIR+'confuse/'
        mkdir(output_dir)
        output_path = output_dir+'cls.html'
        save_to_html(matrix, cfg.CLS.LABELS, cfg.CLS.LABELS, output_path)
        matrix = matrix.tolist()
        matrix = [[str(y) if y != 0 else '-' for y in x] for x in matrix]
        for i in matrix:
            print(i)
        
    def vis_img_cam(self, vis_cam=True):
        self.container.map('out', 'argmax', lambda x : np.argmax(x, axis=0))
        
        gt = self.container.get('gt').flatten()
        predict = self.container.get('argmax').flatten()
        file = self.container.get_raw('file')
        prefix = cfg.CLS.OUTPUT_DIR

        predict_cam = self.container.get('predict_cam') \
            if vis_cam else None
        vis_imgs(file=file, prefix=prefix, cams=predict_cam,
            predict=predict, gt=gt, labelmap=cfg.CLS.LABELS)

    def vis_badcase(self):
        self.container.map('out', 'argmax', lambda x : np.argmax(x, axis=0))

        gt = self.container.get('gt').flatten()
        predict = self.container.get('argmax').flatten()
        file = self.container.get_raw('file')

        badcase_idx =  np.where(predict != gt)[0]
        badcase_index = [file[x] for x in badcase_idx]
        print(badcase_index)
        predict = predict[badcase_idx]
        gt = gt[badcase_idx]

        # single_label(file=badcase_index, prefix=cfg.CLS.VIS.IMG_PREFIX, predict=predict, gt=gt, labelmap=cfg.CLS.VIS.LABEL_MAP)
        single_label(file=badcase_index, prefix='/', predict=predict, gt=gt, labelmap=None)

    def vis_cam_badcase_predict(self):

        gt = self.container.get('gt').flatten()
        predict = self.container.get('argmax').flatten()
        file = self.container.get_raw('file')
        predict_cam = self.container.get('predict_cam')

        badcase_idx =  np.where(predict != gt)[0]

        cam(file=[file[x] for x in badcase_idx], prefix=cfg.CLS.VIS.IMG_PREFIX, cam=predict_cam[badcase_idx])

    def vis_cam_badcase_gt(self):

        gt = self.container.get('gt').flatten()
        predict = self.container.get('argmax').flatten()
        file = self.container.get_raw('file')
        predict_cam = self.container.get('gt_cam')

        badcase_idx =  np.where(predict != gt)[0]

        cam(file=[file[x] for x in badcase_idx], prefix=cfg.CLS.VIS.IMG_PREFIX, cam=predict_cam[badcase_idx])

if __name__ == '__main__':
    cfg = parse()
    cls_rep = ClsReporter()
    cls_rep.confusion_matrix()
    cls_rep.vis_img_cam(cfg.CLS.CAM.ENABLE)