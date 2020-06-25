import sys
sys.path.insert(0, '../../')

import cv2
import math
import random
import numpy as np
import multiprocessing
from line import Line
from fiber_tools.common.util.init_util import parse 

random.seed(2)

class RadiusCnt:
    def __init__(self, step, max_iter, sampler_num, show_point=False):
        self.step = step
        self.max_iter = max_iter
        self.sampler_num = sampler_num
        self.show_point = show_point

    def get_distance(self, start, end):
        diff = end-start
        dist = diff[0]*diff[0]+diff[1]*diff[1]
        return math.sqrt(dist)

    def get_diameter(self, point1, point2 ,mask, image):
        h,w = mask.shape
        line = Line() 
        line.init_by_point(point1, point2)
        line.judge_axis(point1, point2)

        start_point = (point1+point2)/2.0
        end_point_plus = start_point.copy()
        end_point_minus = start_point.copy()

        for i in range(1, self.max_iter):
            tmp_point = start_point.copy()
            cur_steps = i*self.step
            next_point_puls = line.get_next_point(cur_steps, tmp_point)
            if self.show_point:
                cv2.circle(image, (int(next_point_puls[0]), int(next_point_puls[1])), 
                    5, (0, 0, 255) , 4)
                cv2.imshow("COCO detections", image)
                cv2.waitKey(0)
            if next_point_puls[0]<w and next_point_puls[1]<h \
                and next_point_puls[0]>=0 and next_point_puls[1]>=0:
                value = mask[int(next_point_puls[1])][int(next_point_puls[0])]
                if value == 0:
                    end_point_plus = line.get_next_point((i-1)*self.step, tmp_point)
                    break
            else:
                end_point_plus[0] = min(next_point_puls[0], w) if next_point_puls[0]>=0 else end_point_plus[0]
                end_point_plus[1] = min(next_point_puls[1], h) if next_point_puls[1]>=0 else end_point_plus[1]
                break

        for i in range(1, self.max_iter):
            tmp_point = start_point.copy()
            cur_steps =  -i*self.step   
            next_point_minus = line.get_next_point(cur_steps, tmp_point)
            if self.show_point:
                cv2.circle(image, (int(next_point_minus[0]), int(next_point_minus[1])),
                    5, (0, 255, 0) , 4)
                cv2.imshow("COCO detections", image)
                cv2.waitKey(0)
            if next_point_minus[0]<w and next_point_minus[1]<h \
                and next_point_minus[0]>=0 and next_point_minus[1]>=0:
                value = mask[int(next_point_minus[1])][int(next_point_minus[0])]
                if value == 0:
                    end_point_minus = line.get_next_point(-(i-1)*self.step, tmp_point)
                    break
            else:
                end_point_minus[0] = max(next_point_minus[0], 0) if next_point_minus[0]<w else end_point_minus[0]
                end_point_minus[1] = max(next_point_minus[1], 0) if next_point_minus[1]<h else end_point_minus[1]
                break 

        dist_plus = self.get_distance(start_point, end_point_plus)
        dist_minus = self.get_distance(start_point, end_point_minus)
        mid = (int(start_point[0]),int(start_point[1]))
        if dist_plus<=dist_minus:
            res = (int(end_point_minus[0]),int(end_point_minus[1]))
            dist = dist_minus
        else:
            res = (int(end_point_plus[0]),int(end_point_plus[1]))
            dist = dist_plus

        return dist, mid, res

    def sampler_pair(self, contours):
        '''
        randomly get the start and end points for get the perpendicular 
        '''
        point_number = len(contours[0])-1
        start_index = random.randint(0, point_number)
        start = contours[0][start_index][0]

        end_index = start_index
        end = contours[0][end_index][0]

        mid = (start+end)/2
        sampler_cnt = 0
        while (mid[0]==start[0] and mid[1]==start[1]) \
            or (mid[0]==end[0] and mid[1]==end[1]):
            if end_index == point_number:
                end_index = 0
            else:
                end_index+=1
            end = contours[0][end_index][0]
            mid = ((start+end)/2).astype(np.int32)
            sampler_cnt+=1
            if sampler_cnt>100:
                break
        return start, end

    def get_radius_per_fiber(self, input_fiber_path):
        input_fiber = cv2.imread(input_fiber_path)
        gray_input_fiber = cv2.cvtColor(input_fiber, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(
            gray_input_fiber, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        radius = 0.0
        for i in range(self.sampler_num):
            start, end = self.sampler_pair(contours)
            dist, mid, res = self.get_diameter(start, end, gray_input_fiber, input_fiber)
            radius+= dist
        mean = radius/self.sampler_num
        return mean


if __name__ == '__main__':
    # python3 radius_cnt.py --cfg '../config/radius_cnt.yml'
    def write_file(line):
        print(line)
        w.write(line)

    def get_radius(radius_cnt, input_fiber):
        mean_radius = radius_cnt.get_radius_per_fiber(input_fiber)
        res = input_fiber+','+str(mean_radius)+'\n'
        return res

    cfg = parse()
    input_path = cfg.RADIUS.INPUT
    output_path = cfg.RADIUS.OUTPUT 
    step = cfg.RADIUS.STEP
    max_iter = cfg.RADIUS.MAX_ITER
    show_point = cfg.RADIUS.SHOW_POINTS
    sampler_num = cfg.RADIUS.SAMPLER_NUM 

    radius_cnt = RadiusCnt(step, max_iter, 
        sampler_num, show_point)

    w = open(output_path, 'w')
    pool = multiprocessing.Pool(10)
    with open(input_path) as lines:
        for line in lines:
            input_fiber = line.split(',')[0]
            # res = get_radius(radius_cnt, input_fiber)
            # write_file(res)
            
            pool.apply_async(
                get_radius, 
                (radius_cnt, input_fiber,),
                callback = write_file
            )

    pool.close()
    pool.join()
    w.close()
