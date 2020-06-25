import numpy as np

from container.score_container import ScoreContainer
import utils.metric as metric
from utils.process import softmax, label_map
from utils.vis import vis_imgs, mkdir

import sys
sys.path.insert(0, '../../')
from collections import namedtuple, defaultdict
from fiber_tools.common.util.init_util import parse 
from fiber_tools.common.util.time_util import get_cur_time 
from fiber_tools.common.util.get_attr_from_path import get_sample_name 
from fiber_tools.common.constant \
import SL,ML,DENSITY, ML_IDX_TO_LABEL, SL_IDX_TO_LABEL, ml_label_trans
from fiber_tools.common.util.vis_util \
import save_to_html, get_common_table, get_confuse_table


import random

import pdb

# from collections import defaultdict
Fiber = namedtuple('Fiber', 'label, radius, path, score')

class Fibers:
    def __init__(self):
        self.fiber_dict = defaultdict(list) #key:fiber_cls, list(fibers)
        self.fiber_list = [0]*len(SL)

    def add_fiber(self, fiber_path, label_info, radius):
        label, score = label_info[0], label_info[1]
        # fiber_list = self.fiber_dict.setdefault(label, [])
        fiber_list = self.fiber_dict[label]
        fiber_list.append(Fiber(
            label = label,
            radius = radius,
            path = fiber_path,
            score = score
        ))

    def get_fiber_metric(self, fiber_list, topk = 300):
        fiber_list = sorted(fiber_list, key=lambda fiber: fiber.score, 
            reverse=True)
        radiuses = np.array([fiber.radius for fiber in fiber_list[:topk]])
        mean_rad = np.mean(radiuses)**2
        std_rad = np.std(radiuses)**2

        return mean_rad, std_rad

    def get_fiber_ingredient(self, sample_label):
        all_ingred = 0
        for label, ingredient in sample_label.items():
            fiber_list = self.fiber_dict[label]
            mean_rad, std_rad = self.get_fiber_metric(fiber_list)
            tmp_ingred = len(fiber_list)*(mean_rad+std_rad)*DENSITY[label]
            sample_label.update({label:tmp_ingred})
            all_ingred+=tmp_ingred

        for label, ingredient in sample_label.items():
            sample_label.update({label:ingredient/all_ingred})

        return sample_label

    def get_quanty(self):
        for i in range(len(SL)):
            # fiber_num = 
            self.fiber_list[i] = len(self.fiber_dict[i])
        if SL['ZHUMA'] in self.fiber_dict and \
            SL['YAMA'] in self.fiber_dict:
            merge_list = self.fiber_dict[SL['ZHUMA']]+\
                self.fiber_dict[SL['YAMA']]
            if len(self.fiber_dict[SL['ZHUMA']])>\
                len(self.fiber_dict[SL['YAMA']]):
                self.fiber_dict.update(
                    {SL['ZHUMA']: merge_list, SL['YAMA']: []}
                )
            else:
                self.fiber_dict.update(
                    {SL['YAMA']: merge_list, SL['ZHUMA']: []}
                )

        max_fiber_num = 0
        max_fiber_label = 0
        for label, fibers in self.fiber_dict.items():
            if len(fibers)>max_fiber_num:
                max_fiber_num = len(fibers)
                max_fiber_label = label

        sample_label = defaultdict(float)
        sample_label.update({max_fiber_label: 1.0})
        if max_fiber_label==SL['MIAN']:
            if len(self.fiber_dict[SL['YAMA']])/max_fiber_num>thres:
                sample_label.update({SL['YAMA']: 1.0})
            elif len(self.fiber_dict[SL['ZHUMA']])/max_fiber_num>thres:
                sample_label.update({SL['ZHUMA']: 1.0})

        elif max_fiber_label==SL['YAMA'] or max_fiber_label==SL['ZHUMA']:
            if len(self.fiber_dict[SL['MIAN']])/max_fiber_num>thres:
                sample_label.update({SL['MIAN']: 1.0})  
        
        elif max_fiber_label==SL['MAO']:
            if len(self.fiber_dict[SL['RONG']])/max_fiber_num>thres:
                sample_label.update({SL['RONG']: 1.0}) 
                
        elif max_fiber_label==SL['RONG']:
            if len(self.fiber_dict[SL['MAO']])/max_fiber_num>thres:
                sample_label.update({SL['MAO']: 1.0}) 

        if len(sample_label)>1:
            sample_label = self.get_fiber_ingredient(sample_label)

        return sample_label

class Sample:
    def __init__(self, sample_name, sample_label):
        self.sample_name = sample_name
        self.sample_label = sample_label
        self.fibers = Fibers()       #key:image_name, key:list(fibers)
        self.fiber_labels = np.array([0]*5)

    def add_fiber(self, sample_name, fiber_path, 
        predicts, radius):
        if np.sum(predicts>cls_thres)>0:
            cls_label = np.argmax(predicts)
            score = predicts[cls_label]
            self.fibers.add_fiber(fiber_path, 
                (cls_label, score), radius)

class SampleDict:
    def __init__(self):
        self.sample_dict = {}
        self.init()

    def init(self):
        radius_dict = {}
        if len(cfg.SAM.RADIUS) > 0:
            with open(cfg.SAM.RADIUS) as lines:
                for line in lines:
                    fiber_name, radius = line.strip().split(',')
                    radius_dict.update({fiber_name: float(radius)})

        out = np.load(cfg.SAM.FC)
        if len(radius_dict) > 0:
            assert len(out)==len(radius_dict),\
             'file number is not equal with predict results'

        with open(cfg.SAM.GT) as lines:
            for idx, line in enumerate(lines):
                fiber_path = line.strip().split(',')[0]
                cls_res = out[idx]
                radius = radius_dict.setdefault(fiber_name, 1.0)
                self.add_fiber(fiber_path, cls_res, radius)

    def add_fiber(self, fiber_path, predicts, radius):
        sample_name = get_sample_name(fiber_path)
        sample_label =  self.get_sample_label(sample_name)
        sample = self.sample_dict.setdefault(
            sample_name, Sample(sample_name, sample_label)
        )
        sample.add_fiber(sample_name, fiber_path, predicts, radius)

    def get_sample_label(self, sample_name):
        label_dict = defaultdict(float)
        labels = sample_name.split('/')[-1].split('_')
        for label, ingred in zip(labels[::2], labels[1::2]):
            ingred = float(ingred)
            label_dict.update({SL[label]:ingred})
        return label_dict

    def judge(self, sample_quanty, sample_gt, label_diff = 0.02):
        tmp_cls_acc, tmp_cls_quanty_acc = 1, 1
        if len(sample_quanty)!=len(sample_gt):
            return 0, 0
        for label in sample_gt:
            if label not in sample_quanty:
                return 0, 0
            tmp_diff = abs(sample_gt[label]-sample_quanty[label])
            if tmp_diff>label_diff:
                tmp_cls_quanty_acc = 0

        return tmp_cls_acc, tmp_cls_quanty_acc

    def get_sample_acc(self):
        cls_acc = 0.
        cls_quanty_acc = 0.
        gts = []
        predicts= []
        
        file_names_matrix = [[] for i in range(len(ML))]
        judge_res_matrix = [[] for i in range(len(ML))]
        ingred_res_matrix = [[] for i in range(len(ML))]
        for sample_name, sample in self.sample_dict.items():
            sample_quanty = sample.fibers.get_quanty()
            sample_gt = sample.sample_label
            predict_ml_label = ml_label_trans(sample_quanty)
            gt_ml_label = ml_label_trans(sample_gt)
            predicts.append(predict_ml_label)
            gts.append(gt_ml_label)
            tmp_cls_acc, tmp_cls_quanty_acc = self.judge(sample_quanty, sample_gt)
            cls_acc+=tmp_cls_acc
            cls_quanty_acc+=tmp_cls_quanty_acc
            file_names_matrix[gt_ml_label].append(sample_name)
            judge_res_matrix[gt_ml_label].append(sample.fibers.fiber_list)
            ingred_res = [0.0]*len(sample.fibers.fiber_list)
            for label in sample_quanty:
                ingred_res[label] = sample_quanty[label] 
            ingred_res_matrix[gt_ml_label].append(ingred_res)

        print('cls_acc', cls_acc/(len(self.sample_dict)))
        print('cls_quanty_acc', cls_quanty_acc/(len(self.sample_dict)))

        label_names = cfg.SAM.LABELS
        output_dir = cfg.SAM.OUTPUT_DIR+'sample/'
        mkdir(output_dir)
        output_path = output_dir+'sample_cls.html'
        single_label = ['MIAN', 'ZHUMA', 'YAMA', 'MAO', 'RONG']
        save_to_html(judge_res_matrix, single_label, 
            file_names_matrix, output_path, get_common_table)

        label_names = cfg.SAM.LABELS
        output_dir = cfg.SAM.OUTPUT_DIR+'sample/'
        mkdir(output_dir)
        output_path = output_dir+'ingredient.html'
        single_label = ['MIAN', 'ZHUMA', 'YAMA', 'MAO', 'RONG']
        save_to_html(ingred_res_matrix, single_label, 
            file_names_matrix, output_path, get_common_table)

        predicts = np.array(predicts)
        gts = np.array(gts)
        matrix = metric.confusion(gts, predicts)
        output_dir = cfg.SAM.OUTPUT_DIR+'sample/'
        mkdir(output_dir)
        output_path = output_dir+'confuse.html'
        ml_label_marks = [False]*len(ML)
        for index in gts:
            ml_label_marks[index] = True
        for index in predicts:
            ml_label_marks[index] = True
        ml_label = []
        for idx,mark in enumerate(ml_label_marks):
            if mark:
                ml_label.append(ML_IDX_TO_LABEL[idx]) 
                
        save_to_html([matrix], ml_label, 
            [ml_label], output_path, get_confuse_table)

def get_sample_acc():
    radius_dict = {}
    if len(cfg.SAM.RADIUS) > 0:
        with open(cfg.SAM.RADIUS) as lines:
            for line in lines:
                fiber_name, radius = line.strip().split(',')
                radius_dict.update({fiber_name: float(radius)})

    sample = SampleDict()
    out = np.load(cfg.SAM.FC)

    if len(radius_dict) > 0:
        assert len(out)==len(radius_dict),\
         'file number is not equal with predict results'

    with open(cfg.SAM.GT) as lines:
        for idx, line in enumerate(lines):
            fiber_path = line.strip().split(',')[0]
            cls_res = out[idx]
            radius = radius_dict.setdefault(fiber_name, 1.0)
            sample.add_fiber(fiber_path, cls_res, radius)

    sample.get_sampler_acc()


if __name__ == '__main__':
    cfg = parse()
    cls_thres = np.array(cfg.SAM.CLS_FILTER_THRES)
    thres = cfg.SAM.CLS_ERROR_RATE
    sample = SampleDict()
    sample.get_sample_acc()