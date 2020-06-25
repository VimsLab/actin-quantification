import cv2
import numpy as np

class FiberFilter:
    def __init__(self, thres):
        self.thres = thres

    def __call__(self, img_path):
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2 .COLOR_BGR2GRAY)
        score = self.detect(gray_image)
        if score>self.thres:
            return False
        return True

    def detect(self, image):
        return NotImplemented

class SizeFilter(FiberFilter):
    def detect(self, image):
        gray_bool_fig = image!=0
        exist_num = np.sum(gray_bool_fig)
        exist_rate = exist_num/(gray_bool_fig.shape[0]*
            gray_bool_fig.shape[1])

        return exist_rate

class BlurFilter(FiberFilter):
    def detect(self, image):
        x, y = image.shape
        image = image/1.0 #change uint8 to float64 or the sum op in line 40 will overflow
        x_diff = np.abs(image[:-1,:-1]-image[1:,:-1])
        y_diff = np.abs(image[:-1,:-1]-image[:-1,1:])
        mask = (image[:-1,:-1]!=0)&(image[1:,:-1]!=0)&(image[:-1,1:]!=0)
        num = float(np.sum(mask))
        score = np.sum((x_diff+y_diff)*mask)/num
        
        return score

class StripeFilter:
    def __init__(self, blur_thres=10.0, side_blur_thres=2.5):
        self.blur_thres =  blur_thres
        self.side_blur_thres = side_blur_thres
        
    def __call__(self, img_path):
        # fiber_path = '/opt/FTE/users/chrgao/datasets/Textile/single_fiber/2019.5.6/2号/苎麻/115_0.png'
        # if img_path == fiber_path:
        #     import pdb;pdb.set_trace()
        # else:
        #     return
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2 .COLOR_BGR2GRAY)

        x, y = image.shape
        image = image/1.0 #change uint8 to float64 or the sum op in line 40 will overflow
        x_diff = np.abs(image[:-1,:-1]-image[1:,:-1])
         
        y_diff = np.abs(image[:-1,:-1]-image[:-1,1:])
        
        mask = (image[:-1,:-1]!=0)&(image[1:,:-1]!=0)
        x_num = float(np.sum(mask))
        x_score = np.sum((x_diff)*mask)/x_num

        mask = (image[:-1,:-1]!=0)&(image[:-1,1:]!=0)
        y_num = float(np.sum(mask))
        y_score = np.sum((y_diff)*mask)/y_num

        mask = (image[:-1,:-1]!=0)&(image[1:,:-1]!=0)&(image[:-1,1:]!=0)
        num = float(np.sum(mask))
        score = np.sum((x_diff+y_diff)*mask)/num
        
        # side_blur_thres is 4.0 before
        if score>self.blur_thres and \
            (x_score/y_score>self.side_blur_thres or 
            y_score/x_score>self.side_blur_thres):
            return True

        return False

# def get(comp_fiber_path):
#     blur_thres = 10
#     comp_fiber = cv2.imread(comp_fiber_path)
#     # import pdb;pdb.set_trace()
#     comp_fiber = cv2.cvtColor(comp_fiber, cv2 .COLOR_BGR2GRAY) 
#     # comp_fiber = cv2.GaussianBlur(comp_fiber,(3,3),1)
#     # comp_fiber = cv2.medianBlur(comp_fiber,5)
#     # cv2.imshow('comp_fiber.png', comp_fiber)
#     # cv2.waitKey(0)   
#     # b_filter = Size_Filter(blur_thres)
#     # b_filter = Blur_Filter(blur_thres)
#     b_filter = Stripe_Filter(blur_thres)
#     b_filter(comp_fiber_path)
#     # score = b_filter.detect(comp_fiber_path, comp_fiber)
#     # print(score)
#     # return score

# if __name__ == "__main__":
#     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/yongxingdata/test_fiber_pre/1/Image0087_2.png'
#     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/20190513_160616/苎麻/mian/025_0_0.5619938,0.40816948,0.012250177,0.009086391,0.008500201.png'
#     # get(comp_fiber_path87)

# #     # comp_fiber_path88 = '/opt/FTE/users/chrgao/datasets/Textile/yongxingdata/test_fiber_pre/1/Image0088_2.png'
# #     # get(comp_fiber_path88)

# #     # comp_fiber_path86 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻1/yama/014_0_2.6650396e-06,9.602409e-07,0.99999547,6.623952e-07,2.7559784e-07.png'
# #     # get(comp_fiber_path86)
#     # path_name = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻2/yama/'
#     path_name = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/20190513_160616/苎麻/mian'
#     if os.path.exists(path_name):
#         fileList = os.listdir(path_name)
#         for f in fileList:
#             cur_dir = os.path.join(path_name,f)

#             if os.path.isdir(cur_dir):   
#                 dir_all(cur_dir)              
#             else:
#                 # import pdb;pdb.set_trace()
#                 score = get(cur_dir)
# #                 # print(cur_dir, score)
# #     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻2/mian/029_0_0.6710679,0.00028091163,0.3125907,6.7567663e-07,0.01605988.png'
# #     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻2/mian/014_0_0.9983967,0.00044377477,0.0002930907,6.4423496e-08,0.0008661697.png'
# #     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻2/mian/189_0_0.47690743,0.29497322,0.22560287,0.002469863,4.665249e-05.png'
# #     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻2/mian/009_0_0.7019644,0.079237,0.018598352,0.0012193287,0.19898096.png'
# #     # comp_fiber_path87 = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/error_data/human_seg_mix_fiber/bad_case/亚麻2/yama/123_1_4.740725e-08,0.0004911666,0.9995079,6.3913546e-07,1.9691161e-07.png'
# #     # print(get(comp_fiber_path87))