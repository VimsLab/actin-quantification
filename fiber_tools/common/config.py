###########
#
#    
#   do not change this file
#
#
###########
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.PREPROCCESS = edict()
__C.PREPROCCESS.INPUT_DIR = ''
__C.PREPROCCESS.OUTPUT_File = ''

__C.CLEAREST = edict()
__C.CLEAREST.INPUT = ''
__C.CLEAREST.OUTPUT = ''
__C.CLEAREST.EXPOSURES = [9,10,11,12,13]

__C.FILTER = edict()
__C.FILTER.INPUT = ''
__C.FILTER.OUTPUT = ''
__C.FILTER.SIZE_THRES = 0.5
__C.FILTER.BLUR_THRES = 0.5
__C.FILTER.STRIPE_BLUR_THRES = 0.5
__C.FILTER.STRIPE_SIDE_THRES = 0.5

__C.CLEAREST = edict()
__C.CLEAREST.INPUT = ''
__C.CLEAREST.OUTPUT = ''
__C.CLEAREST.EXPOSURES = [9,10,11,12,13]

__C.RADIUS = edict()
__C.RADIUS.INPUT = ''
__C.RADIUS.OUTPUT = ''
__C.RADIUS.STEP = 1
__C.RADIUS.MAX_ITER = 255
__C.RADIUS.SHOW_POINTS = False
__C.RADIUS.SAMPLER_NUM = 5000

__C.CLS = edict()
__C.CLS.GT = '' 
__C.CLS.LABELS = ['MIAN', 'ZHUMA', 'YAMA', 'MAO', 'RONG']
__C.CLS.FC = ''
__C.CLS.CAM = edict()
__C.CLS.CAM.ENABLE = False
__C.CLS.CAM.PREDICT = ''
__C.CLS.CAM.GT = ''
__C.CLS.TOPK = edict()
__C.CLS.TOPK.K = [1]
__C.CLS.OUTPUT_DIR = ''

__C.SAM = edict()
__C.SAM.RADIUS = '' 
__C.SAM.GT = ''
__C.SAM.FC = ''
__C.SAM.OUTPUT_DIR = ''
__C.SAM.LABELS = ['MIAN', 'ZHUMA', 'YAMA', 'MAO', 'RONG']
__C.SAM.CLS_FILTER_THRES = [0.0, 0.0, 0.0, 0.0, 0.0]
__C.SAM.CLS_ERROR_RATE = 0.05



def get_output_dir(dataset_name, weights_filename):
    """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, dataset_name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(dataset_name, weights_filename):
    """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, dataset_name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(
                                      type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
          'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

