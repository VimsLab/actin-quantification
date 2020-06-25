import numpy as np

class Line:
    '''
    The line ax+by+c=0
    '''
    def __init__(self, a=0., b=0., c=0., change_y_axis=True):
        self.a = a
        self.b = b
        self.c = c
        self.change_y_axis = change_y_axis

    def init_by_point(self, point1, point2):
        '''
        init weights of the line ax+by+c=0, by two point
        ''' 
        diff = point2-point1
        mid = (point1+point2)/2
        if diff[0] == 0.0:
            self.a = 0.0
            self.b = 1.0
            self.c = -mid[1]
        elif diff[1] == 0.0:
            self.a = 1.0
            self.b = 0.0
            self.c = -mid[0]
        else:
            k1 = diff[1]/diff[0]
            b1 = ((point2[0]*point1[1])-(point1[0]*point2[1]))/diff[0]
            k2 = -1.0/k1
            b2 = -1.0*mid[0]*k2+mid[1]
            self.a = -1.0*k2
            self.b = 1.0
            self.c = -1*b2

    def get_point_by_xchanged(self, point, steps):
        '''
        for function ax+by+c=0, get the point 
        when steps is happen in x value
        '''
        if self.a == 0.0:
            x = point[0]
            y = point[1]
        elif self.b == 0.0:
            x = point[0]-steps
            y = point[1]+steps
        else:
            x = point[0]
            y = (-self.c-self.a*x)/self.b 

        return np.array([x,y])

    def get_point_by_ychanged(self, point, steps):
        '''
        for function ax+by+c=0, get the point 
        when steps is happen in y value
        '''
        if self.a == 0.0:
            x = point[0]+steps
            y = point[1]-steps
        elif self.b == 0.0:
            x = point[0]
            y = point[1]
        else:
            y = point[1]
            x = (-self.c-self.b*y)/self.a
            

        return np.array([x,y])

    def get_next_point(self, cur_steps, tmp_point):
        '''
        for function ax+by+c=0 get the next point
        '''
        tmp_point[0]+=(self.change_y_axis!=True)*cur_steps
        tmp_point[1]+=(self.change_y_axis==True)*cur_steps
        if self.change_y_axis:
            next_point_puls = self.get_point_by_ychanged(tmp_point,cur_steps)
        else:
            next_point_puls = self.get_point_by_xchanged(tmp_point,cur_steps)

        return next_point_puls

    def judge_axis(self, point1, point2):
        '''decide where to go, step by y_axis or x_axis'''
        self.change_y_axis = False
        diff = point1-point2
        if abs(diff[0])>abs(diff[1]):
            self.change_y_axis = True