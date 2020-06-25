import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
import multiprocessing
from collections import namedtuple
from filter import BlurFilter
from fiber_tools.common.util.get_attr_from_path import \
get_exposure, get_field, get_image_name, get_sample_name
from fiber_tools.common.util.init_util import parse 

Fiber = namedtuple('Fiber', 'label, path')


class Field(object):
    '''
    just one field, #key:image_name, key:list(fibers)
    '''
    def __init__(self):
        self.image_dict = {}       
        

    def add_images(self, image_name, fibers):
        self.image_dict[image_name] = fibers

    def find_clearest(self):
        clearest_fibers = []
        anchor_img_name = None
        max_fiber_num = -10
        for image_name, fibers in self.image_dict.items():
            if len(fibers)>max_fiber_num:
                max_fiber_num = len(fibers)
                anchor_img_name = image_name
        anchor_fibers = self.image_dict[anchor_img_name]
        # for anchor_img_name, anchor_fibers in self.image_dict.items():
        for anchor_fiber in anchor_fibers:
            anchor_fiber_img = cv2.imread(anchor_fiber.path)
            anchor_fiber_img = cv2.cvtColor(anchor_fiber_img, cv2 .COLOR_BGR2GRAY)
            max_blur = blur_filter.detect(anchor_fiber_img)
            clearest_fiber = anchor_fiber
            clearest_fiber = self.compare_wtih_other_imgs(
                anchor_img_name,  
                clearest_fiber,
                anchor_fiber_img,
                max_blur
            )
            clearest_fibers.append(clearest_fiber)

        return clearest_fibers

    def compare_wtih_other_imgs(self, anchor_img_name, max_fiber, 
        anchor_fiber_img, max_blur):
        for cmp_img_name, cmp_fibers in self.image_dict.items():
            if anchor_img_name==cmp_img_name:
                continue

            max_overlap = -10
            max_overlap_fiber = None
            max_overlap_fiber_img = None
            for cmp_fiber in cmp_fibers:
                cmp_fiber_img = cv2.imread(cmp_fiber.path)
                cmp_fiber_img = cv2.cvtColor(cmp_fiber_img, cv2 .COLOR_BGR2GRAY)
                cur_lap = self.get_overlap(anchor_fiber_img, cmp_fiber_img)
                if cur_lap > max_overlap and cur_lap>0:
                    max_overlap = cur_lap
                    max_overlap_fiber_img = cmp_fiber_img
                    max_overlap_fiber = cmp_fiber

            if max_overlap_fiber_img is not None:
                cmp_blur = blur_filter.detect(max_overlap_fiber_img)
                if cmp_blur > max_blur:
                    max_blur = cmp_blur
                    max_fiber = max_overlap_fiber

        return max_fiber

    def get_overlap(self, image1, image2):
        w = min(image1.shape[1], image2.shape[1])
        h = min(image1.shape[0], image2.shape[0])
        image1 = image1[0:h, 0:w]
        image2 = image2[0:h, 0:w]
        overlap = np.sum((image1!=0)&(image2!=0))
        return overlap

#with the same sample
class Sample(object):
    def __init__(self):
        self.image_dict_field = {} #key:field_index, key: dict[image_dict]
        self.image_dict = {}       #key:image_name, key:list(fibers)
        self.field_number = 0

    def add_fiber(self, fiber_path, label):
        image_name = get_image_name(fiber_path)
        fiber_dicts = self.image_dict.setdefault(image_name, [])
        fiber_dicts.append(Fiber(
            label=label, path=fiber_path))

    def aggregate_fibers(self, fibers):
        res_image = None
        for fiber in fibers:
            fiber_path = fiber.path
            cur_fiber = cv2.imread(fiber_path)
            cur_fiber = cv2.cvtColor(cur_fiber, cv2 .COLOR_BGR2GRAY)
            cur_fiber.astype(np.float64)
            if res_image is None:
                res_image = cur_fiber 
                continue
            res_image+=cur_fiber
            res_image = res_image.clip(0.0, 255.0)
        
        return res_image

    def get_overlap_rate(self, image1, image2):
        h = min(image1.shape[0], image2.shape[0])
        w = min(image1.shape[1], image2.shape[1])
        image1 = image1[0:h, 0:w]
        image2 = image2[0:h, 0:w]
        overlap = np.sum((image1!=0)&(image2!=0))
        overlap = overlap/(min(np.sum(image1!=0), np.sum(image2!=0)))
        return overlap

    def find_begin_end(self, sorted_image_keys, rate_thres):
        anchor_img = None
        start_index, end_index = 0, len(sorted_image_keys)-1
        for i in range(0, end_index+1):
            cur_image_name = sorted_image_keys[i]
            fibers_cur_image = self.image_dict[cur_image_name]
            cur_image = self.aggregate_fibers(fibers_cur_image)
            if anchor_img is None:
                anchor_img = cur_image
            elif self.get_overlap_rate(cur_image, anchor_img)<rate_thres:
                start_index = i
                break

            images_with_same_field = self.image_dict_field.setdefault(
                self.field_number, Field()
            )
            images_with_same_field.add_images(
                cur_image_name, fibers_cur_image
            )
            if i==end_index:
                start_index = end_index+1

        for i in range(end_index, start_index, -1):
            cur_image_name = sorted_image_keys[i]
            fibers_cur_image = self.image_dict[cur_image_name]
            cur_image = self.aggregate_fibers(fibers_cur_image)
            if self.get_overlap_rate(cur_image, anchor_img)<rate_thres:
                end_index = i
                break

            images_with_same_field = self.image_dict_field.setdefault(
                self.field_number, Field()
            )
            images_with_same_field.add_images(
                cur_image_name, fibers_cur_image
            )

        self.field_number+=1
        return start_index, end_index

    # merge images by focus if we don't know which images 
    # belong to the same field before
    def aggregate_images_without_field_labels(self, rate_thres = 0.9):
        sorted_image_keys = sorted(self.image_dict.keys())
        image_index = 0

        start_index, end_index = self.find_begin_end(sorted_image_keys, rate_thres)
        anchor_img = None
        for i in range(start_index, end_index+1):
            cur_image_name = sorted_image_keys[i]
            fibers_cur_image = self.image_dict[cur_image_name]
            cur_image = self.aggregate_fibers(fibers_cur_image)
            if anchor_img is None:
                anchor_img = cur_image
            elif self.get_overlap_rate(cur_image, anchor_img)<rate_thres:
                anchor_img = cur_image
                self.field_number+=1

            images_with_same_field = self.image_dict_field.setdefault(
                self.field_number, Field()
            )
            images_with_same_field.add_images(
                cur_image_name, fibers_cur_image
            )

    def aggregate_images_with_field_labels(self):
        field_names = set([])
        for cur_image_name, fibers_cur_image in self.image_dict.items():
            cur_field_name = get_field(cur_image_name)
            images_with_same_field = self.image_dict_field.setdefault(
                cur_field_name, Field()
            )
            images_with_same_field.add_images(
                cur_image_name, fibers_cur_image
            )

    # def fix_fiber_label(self):
        

class SampleDict(object):
    def __init__(self):
        self.sample_dict = {}

    def add_fiber(self, fiber_path, label):
        sample_name = get_sample_name(fiber_path)
        fiber_dicts = self.sample_dict.setdefault(sample_name, Sample())
        fiber_dicts.add_fiber(fiber_path, label)

def get_sample_dict(input_path):
    sample_dict = SampleDict()
    with open(input_path) as lines:
        for line in lines:
            path, label = line.split(',')
            cur_exposure = get_exposure(path)
            if cur_exposure in exposures and \
                'field0' in path:
                sample_dict.add_fiber(path, label)

    return sample_dict


if __name__ == '__main__':
    def find_clearest_for_each_field(field_images):
        return field_images.find_clearest()
        
    def write_file(clearest_fibers):
        for fiber in clearest_fibers:
            print(fiber.path)
            w.write(fiber.path+','+fiber.label)

    blur_filter = BlurFilter(10)
    cfg = parse()
    exposures = set(cfg.CLEAREST.EXPOSURES)
    input_path = cfg.CLEAREST.INPUT
    output_path = cfg.CLEAREST.OUTPUT

    print('get_sample_dict')
    samples = get_sample_dict(input_path)
    
    print('begin_merge_by_field')
    for sample_name, sample in samples.sample_dict.items():
        # sample.aggregate_images_without_field_labels()
        sample.aggregate_images_with_field_labels()

    print('begin_find_clearest_fiber')
    w = open(output_path, 'w')
    pool = multiprocessing.Pool(10)
    for sample_name, sample in samples.sample_dict.items():
        for field_name, field_images in sample.image_dict_field.items():
            clearest_fibers = find_clearest_for_each_field(field_images)
            for fiber in clearest_fibers:
                print(fiber.path)
                w.write(fiber.path+','+fiber.label)
            # pool.apply_async(
            #     find_clearest_for_each_field, 
            #     (field_images,),
            #     callback = write_file
            # )

    pool.close()
    pool.join()
    w.close()