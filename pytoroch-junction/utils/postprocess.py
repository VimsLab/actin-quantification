import numpy as np
import cv2
import skfmm

from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_dilation
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy import interpolate
from skimage.draw import line

from skimage.morphology import skeletonize
from test_config import cfg
import pylab as pl
import copy
from skimage import io, color, morphology
from skimage import measure

import xlwt

from math import sin, cos, radians

def rotate_point(input_points, angle, center_point=(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    test = np.zeros((1,2))
    test[0,0] = 1
    points = input_points
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin

    points_temp = copy.deepcopy(points)
    points_temp[:,0] = points[:,0] - center_point[0]
    points_temp[:,1] = points[:,1] - center_point[1]
    # c, s = np.cos(radians), np.sin(radians)
    # j = np.matrix([[c, s], [-s, c]])
    # m = np.dot(j, [x, y])
    points[:,0] = points_temp[:,1] * sin(angle_rad) + points_temp[:,0] * cos(angle_rad)
    points[:,1] = points_temp[:,1] * cos(angle_rad) - points_temp[:,0] * sin(angle_rad)
    # Reverse the shifting we have done
    points[:, 0] = points[:,0] + center_point[0]
    points[:, 1] = points[:,1] + center_point[1]
    # import pdb; pdb.set_trace()
    return points

def normalize_include_neg_val(tag):

    # Normalised [0,255]
    normalised = 1.*(tag - np.min(tag))/np.ptp(tag).astype(np.float32)

    return normalised

def get_angle(x, y):
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle_radius=np.arccos(cos_angle)
    angle_degree=angle_radius*360/2/np.pi
    return angle_degree

def accumulate_votes(votes, shape):
    xs = votes[:,0]
    ys = votes[:,1]
    ps = votes[:,2]
    tl = [np.floor(ys).astype('int32'), np.floor(xs).astype('int32')]
    tr = [np.floor(ys).astype('int32'), np.ceil(xs).astype('int32')]
    bl = [np.ceil(ys).astype('int32'), np.floor(xs).astype('int32')]
    br = [np.ceil(ys).astype('int32'), np.ceil(xs).astype('int32')]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps*(1.-dx)*(1.-dy)
    tr_vals = ps*dx*(1.-dy)
    bl_vals = ps*dy*(1.-dx)
    br_vals = ps*dy*dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    heatmap = np.asarray(coo_matrix( (data[good_inds], (I[good_inds],J[good_inds])), shape=shape ).todense())

    return heatmap

def iterative_bfs(graph, start, path=[]):
    '''iterative breadth first search from start'''
    q=[(None,start)]
    visited = []
    while q:
        v=q.pop(0)
        if not v[1] in visited:
            visited.append(v[1])
            path=path+[v]
            q=q+[(v[1], w) for w in graph[v[1]]]
    return path

def compute_heatmaps(kp_maps, short_offsets):
    map_shape = kp_maps.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))


    this_kp_map = kp_maps
    votes = idx + short_offsets.transpose((1,2,0))
    votes = np.reshape(np.concatenate([votes, this_kp_map[:,:,np.newaxis]], axis=-1), (-1, 3))
    heatmap = accumulate_votes(votes, shape=map_shape) / (np.pi*cfg.disc_radius**2)

    return heatmap

def get_keypoints(heatmaps):
    keypoints = []
    peaks = maximum_filter(heatmaps, footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmaps
    # import pdb;pdb.set_trace()
    peaks = zip(*np.nonzero(peaks))
    keypoints.extend([{'xy': np.array(peak[::-1]), 'conf': heatmaps[peak[0], peak[1]]} for peak in peaks])
    keypoints = [kp for kp in keypoints if kp['conf'] > cfg.PEAK_THRESH * 2]

    return keypoints

def get_curvature(points): #input numpy arrary, [[x,y]]


    dx_dt = np.gradient(points[:, 0])
    dy_dt = np.gradient(points[:, 1])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / ((dx_dt * dx_dt + dy_dt * dy_dt)**1.5 + 0.000001) # avoid div by zero
    # import pdb;pdb.set_trace()
    points_pad = np.vstack((points, points[-1] + (points[-1] - points[-2]), points[-1] + 2 * (points[-1] - points[-2])))
    # import pdb;pdb.set_trace()
    angle = np.zeros_like(curvature)
    for i in range(len(points)):
        x1, y1 = points_pad[i]
        x2, y2 = points_pad[i + 1]
        x3, y3 = points_pad[i + 2]

        vec1 = (y2 - y1, x2 - x1)
        vec2 = (y3 - y2, x3 - x2)
        cos_angle = np.dot(vec1,vec2)
        norm_ = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = cos_angle / (norm_ + 0.000001)
        angle_rad = np.arccos(cos_angle)
        # print(angle_rad)
        if cos_angle < 0:
            angle_rad = np.pi - angle_rad

        angle[i] = angle_rad
        angle[i] = cos_angle

    angle[0] = 0.000001
    angle[1] = 0.000001
    angle[2] = 0.000001
    angle[-1] = 0.000001
    angle[-2] = 0.000001
    angle[-3] = 0.000001
        # import pdb;pdb.set_trace()


    cos_angle = dx_dt * np.pad(dx_dt, (0, 1), 'edge')[1:] + dy_dt * np.pad(dy_dt, (0, 1), 'edge')[1:]
    cos_angle /= ((np.sqrt(dx_dt ** 2 + dy_dt**2) * np.sqrt( np.pad(dx_dt, (0, 1), 'edge')[1:]**2, np.pad(dy_dt, (0, 1), 'edge')[1:]**2)) + 0.000001)

    cos_angle[np.where(cos_angle>1)] = 1.0

    angle_value = np.arccos(cos_angle)

    # import pdb;pdb.set_trace()
    for i in range(len(angle_value)):
        if angle_value[i] < 0:
            angle_value[i] = np.pi - angle_value[i]


    # tangent_theta = dy_dt / dx_dt
    # angle = np.arctan(tangent_theta) * 180 / np.pi


    return curvature, angle

def get_curvature_v2(points, step): #input numpy arrary, [[x,y]]


    dx_dt = np.gradient(points[:, 0])
    dy_dt = np.gradient(points[:, 1])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / ((dx_dt * dx_dt + dy_dt * dy_dt)**1.5 + 0.000001) # avoid div by zero
    # import pdb;pdb.set_trace()
    points_pad = points
    for i in range(step):
        points_pad = np.vstack((points_pad, points[-1] + (i + 1) * (points[-1] - points[-2])))

    # import pdb;pdb.set_trace()
    angle = np.zeros_like(curvature)
    for i in range(len(points)):
        x1, y1 = points_pad[i - 1*step]
        x2, y2 = points_pad[i]
        x3, y3 = points_pad[i + 1*step]

        vec1 = (y2 - y1, x2 - x1)
        vec2 = (y3 - y2, x3 - x2)
        cos_angle = np.dot(vec1,vec2)
        norm_ = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = cos_angle / (norm_ + 0.000001)
        angle_rad = np.arccos(cos_angle)
        # print(angle_rad)
        if cos_angle < 0:
            angle_rad = np.pi - angle_rad

        angle_rad = angle_rad / np.pi
        angle[i] = angle_rad


    angle[0:step] = 0.0001
    # angle[1] = 0.000001
    # angle[2] = 0.000001
    # angle[-1] = 0.000001
    # angle[-2] = 0.000001
    angle[-step:] = 0.0001
        # import pdb;pdb.set_trace()


    cos_angle = dx_dt * np.pad(dx_dt, (0, 1), 'edge')[1:] + dy_dt * np.pad(dy_dt, (0, 1), 'edge')[1:]
    cos_angle /= ((np.sqrt(dx_dt ** 2 + dy_dt**2) * np.sqrt( np.pad(dx_dt, (0, 1), 'edge')[1:]**2, np.pad(dy_dt, (0, 1), 'edge')[1:]**2)) + 0.000001)

    cos_angle[np.where(cos_angle>1)] = 1.0

    angle_value = np.arccos(cos_angle)

    # import pdb;pdb.set_trace()
    for i in range(len(angle_value)):
        if angle_value[i] < 0:
            angle_value[i] = np.pi - angle_value[i]

    return curvature, angle

def fast_march(keypoints, seg_mask, filename, write=True):
    if write:
        book = xlwt.Workbook()
        length__ = book.add_sheet('length')
        angle__ = book.add_sheet('angle')
        curvature__ = book.add_sheet('curvature')
        curvature__.write(0, 1,'max_curvaures')
        curvature__.write(0, 2,'min_curvaures')
        curvature__.write(0, 3,'mean_curvatures')
        summary__ = book.add_sheet('summary')
        summary__.write(0, 0,'num_of_objects')
        summary__.write(0, 1,'average_length')
        summary__.write(0, 2,'average_angle')
        summary__.write(0, 3,'average_curvatures')

        for_k_mean_clustering_book = xlwt.Workbook()
        for_k_mean_clustering = for_k_mean_clustering_book.add_sheet('para')
        for_k_mean_clustering.write(0,0,'orientation')
        for_k_mean_clustering.write(0,1,'2nd order')
        for_k_mean_clustering.write(0,2,'1st order')
        for_k_mean_clustering.write(0,3,'constant')
    step = 5
    map_shape = seg_mask.shape
    mask = ~(seg_mask>0)

    phi = 1.0 * np.ones(map_shape)
    phi  = np.ma.MaskedArray(phi, mask)




    for i in range(len(keypoints)):
        corr_xy = keypoints[i]['xy']
        phi[corr_xy[1], corr_xy[0]] = -1.
    ok = skfmm.distance(phi, dx=1)
    distance_values = ok.data


    ##############individual
    import copy
    inter_mask = copy.deepcopy(distance_values)
    inter_mask = (inter_mask < 5)
    phi_for_individual = 1.0 * np.ones(map_shape)

    seg_mask_v2 = copy.deepcopy(seg_mask)
    seg_mask_v2 = morphology.skeletonize(seg_mask_v2)
    phi_for_individual  = np.ma.MaskedArray(phi_for_individual, mask)
    # phi_for_individual  = np.ma.MaskedArray(phi_for_individual, inter_mask)

    normalized_distance_map = ok.data

    normalized_distance_map = normalize_include_neg_val(ok.data) # skfmm distances are negative value
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(normalized_distance_map)

    rgb_org = copy.deepcopy(rgba_img)

    # cv2.imwrite('rgba_img__.png',rgba_img*1.0)
    # cv2.imshow('t.png', normalized_distance_map*1.0)
    # cv2.waitKey(0)
    abs_value_map = normalized_distance_map * (np.max(ok.data) - np.min(ok.data))
    abs_value_map = abs_value_map * seg_mask

    # abs_value_map
    abs_value_map_gaussed = gaussian_filter(abs_value_map, sigma=2)
    rgba_img_2 = cmap(abs_value_map)
    # cv2.imshow('rgba_img_gauss.png',rgba_img_2*1.0)
    # cv2.waitKey(0)
    #peaks = maximum_filter(abs_value_map, footprint=[[0,1,0],[1,1,1],[0,1,0]]) == abs_value_map
    # peaks = maximum_filter(abs_value_map, size = 10)== abs_value_map
    peaks = maximum_filter(abs_value_map_gaussed, size = 3)== abs_value_map_gaussed
    peaks = zip(*np.nonzero(peaks))
    num_of_objects = 0
    max_point = []
    # values = []
    total_length_march = 0
    keypoints_data_to_save = []
    for peak in peaks:
        if abs_value_map[peak[0], peak[1]] > 10:
            max_point.append({'ind' : num_of_objects, 'xy': np.array(peak[::-1]), 'conf': abs_value_map[peak[0], peak[1]]})
            rgba_img = cv2.circle(rgba_img, (peak[1], peak[0]), 4, (255,0,255), 1)
            keypoints_data_to_save.append((peak[1], peak[0]))
            keypoints_data_to_save.append(distance_values[peak[0], peak[1]])
            # total_length_march = total_length_march + (distance_values[peak[0], peak[1]])
            # num_of_objects = num_of_objects + 1

    list_object = np.zeros((len(keypoints_data_to_save),), dtype=np.object)
    list_object[:] = keypoints_data_to_save
    import scipy.io
    scipy.io.savemat('../../keypoints_data.mat', mdict={'list_object': list_object})

#######################################################################################\
    row = 0
    length_total = 0
    angle_total = 0
    curvatures_total = 0

    curvature_map = np.zeros_like(seg_mask)
    curvature_map_2 = np.zeros_like(seg_mask)
    curvature_map_mask = np.zeros_like(seg_mask)
    curve_reconstruction = np.zeros_like(seg_mask)
    canvas = (seg_mask)
    obj_para = []
    import time
    start_time = time.time()
    from tqdm import tqdm

    data_points = open(r"../data_mask.txt","w+")
    data_to_save = []

    for ii in tqdm(range(len(max_point))):

        # if ii > 150:
        #     break



        local_center = (max_point[ii]['xy'][1], max_point[ii]['xy'][0])
        # phi_for_individual  = np.ma.MaskedArray(phi_for_individual, inter_mask)
        phi_for_individual_temp = copy.deepcopy(phi_for_individual)
        phi_for_individual_temp[max_point[ii]['xy'][1], max_point[ii]['xy'][0]] = -1.
        # print ([max_point[ii]['xy'][1], max_point[ii]['xy'][0]])
        length_value = distance_values[max_point[ii]['xy'][1], max_point[ii]['xy'][0]]

        if length_value < 10:
            continue
        dist_map_from_mid_point = skfmm.distance(phi_for_individual_temp, dx=1, narrow= length_value)
        # if(length_value < 20):
        #     continue

        # if ii != 19:
        #     continue


        individual_segment_map = dist_map_from_mid_point.data



        seg_map_dilate = cv2.dilate(np.float32(1.0 * individual_segment_map > 0), np.ones((5,5)))
        seg_map_erode = cv2.erode(seg_map_dilate, np.ones((5,5)))
        # seg_map_erode = np.float32(1.0 * individual_segment_map > 0)

        # from skimage.morphology import medial_axis
        # individual_segment_map_skel, distance = medial_axis(seg_map_erode>0, return_distance=True)
        # cv2.imshow('t', 1.0 * (individual_segment_map_skel>0))
        # cv2.waitKey(0)
        # import pdb; pdb.set_trace()


        individual_segment_map_skel = skeletonize(1.0 * (seg_map_erode > 0))
        # import pdb; pdb.set_trace()
        intersections, _ = get_skeleton_intersection_and_endpoint(individual_segment_map_skel)
        for p in intersections:
            individual_segment_map_skel[p[1],p[0]] = 0
        # import pdb; pdb.set_trace()
        contours, hierarchy = cv2.findContours(np.uint8(individual_segment_map_skel),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        selected_contour = contours[0]
        for a_contour in contours:
            if len(a_contour) > len(selected_contour):
                selected_contour = a_contour

        canvas_tmp = np.zeros_like(1.0 * individual_segment_map_skel)
        selected_contour = selected_contour.squeeze()
        canvas_tmp[selected_contour[:,1],selected_contour[:,0]] = 1.0


        # canvas_tmp = cv2.dilate(np.float32(canvas_tmp), np.ones((3,3)))
        contours_, _ = cv2.findContours(np.uint8(canvas_tmp),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (x,y),(MA,ma),angle = cv2.fitEllipse(contours_[0])

        center_elipse = (y,x)
        print('id: %3d, angle: %6.2f'%(ii, angle))
        y_, x_= np.where(seg_map_dilate > 0)

        row_local = np.max(y_) - np.min(y_) + 1
        col_local = np.max(x_) - np.min(x_) + 1

        center = (int(col_local/2 + np.min(y_)), int(row_local/2 + np.min(x_)))
        local_y = y_ #- center[0]
        local_x = x_ #- center[1]
        data_to_save.append(local_x)
        data_to_save.append(local_y)
        #
        # # import pdb; pdb.set_trace()
        #
        # fig = plt.figure(1)
        # plt.plot(local_y, local_x, 'bo')
        #
        # # plt.show()
        # fig.savefig('individual/'+ str(ii) +'_before_switch' + str(angle) +'.png')
        #
        # plt.clf()
        # # seg_map_erode = cv2.ellipse(seg_map_erode,((x,y),(MA,ma),angle),1,2)
        # if (angle > 135 or angle < 45):
        #
        #     # rotate with angle
        #     # import pdb; pdb.set_trace()
        #     stack_yx = np.vstack((local_y, local_x)).transpose()
        #
        #     # import pdb; pdb.set_trace()
        #     # if (angle>135 and angle < 45):
        #     #     rotation_angle = 0
        #     # stack_yx_rotated = rotate_point(stack_yx, -angle, center_point=(0, 0))
        #     # import pdb; pdb.set_trace()
        #     # local_y = stack_yx_rotated[:,0]
        #     # local_x = stack_yx_rotated[:,1]
        #     # import pdb; pdb.set_trace()
        #     curve_fitted_para = np.polyfit(local_y,local_x,7)
        #
        #     curve_fitted = np.poly1d(curve_fitted_para)
        #     input_for_curve = np.sort(np.unique(local_y), axis=None)
        #
        #     output_coor = curve_fitted(input_for_curve)
        #     poly_d = np.polyder(curve_fitted)
        #     poly_dd = np.polyder(poly_d)
        #     curvatures = np.abs(poly_dd(input_for_curve)) / ((1 + poly_d(input_for_curve)**2) ** 3/2 + 0.00001)
        #     # import pdb; pdb.set_trace()
        #     obj_entry = [1, poly_d[2], poly_d[1], poly_d[0]]
        #     obj_para.append(obj_entry)
        #     fig = plt.figure(1, figsize=(4, 4))
        #
        #     plt.plot(input_for_curve, output_coor)
        #     # plt.show()
        #     fig.savefig('individual/'+ str(ii) +'_fitted.png')
        #
        #     plt.clf()
        #     plt.plot(local_y, local_x, 'bo')
        #     fig.savefig('individual/'+ str(ii) +'_origin.png')
        #     plt.clf()
        #     try:
        #         curve_reconstruction[input_for_curve.astype(int) + center[0], output_coor.astype(int) + center[1] ] = curvatures
        #     except:
        #         import pdb; pdb.set_trace()
        # elif (angle < 135 or angle > 45):
        #     # cv2.imshow('t', seg_map_erode)
        #     # cv2.waitKey(0)
        #
        #
        #     # rotate with angle
        #     # import pdb; pdb.set_trace()
        #     # stack_yx = np.vstack((local_y, local_x)).transpose()
        #
        #     # import pdb; pdb.set_trace()
        #     # if (angle>135 and angle < 45):
        #     #     rotation_angle = 0
        #     # stack_yx_rotated = rotate_point(stack_yx, -angle, center_point=(0, 0))
        #     # import pdb; pdb.set_trace()
        #     # local_y = stack_yx_rotated[:,0]
        #     # local_x = stack_yx_rotated[:,1]
        #     # import pdb; pdb.set_trace()
        #     curve_fitted_para = np.polyfit(local_x,local_y,7)
        #
        #     curve_fitted = np.poly1d(curve_fitted_para)
        #     input_for_curve = np.sort(np.unique(local_x), axis=None)
        #
        #     output_coor = curve_fitted(input_for_curve)
        #     poly_d = np.polyder(curve_fitted)
        #
        #     obj_entry = [0, poly_d[2], poly_d[1], poly_d[0]]
        #     obj_para.append(obj_entry)
        #
        #     poly_dd = np.polyder(poly_d)
        #     curvatures = np.abs(poly_dd(input_for_curve)) / ((1 + poly_d(input_for_curve)**2) ** 3/2 + 0.00001)
        #     # import pdb; pdb.set_trace()
        #     fig = plt.figure(1, )
        #     plt.plot(input_for_curve, output_coor)
        #     # plt.show()
        #     fig.savefig('individual/'+ str(ii) +'_fitted_switch.png')
        #
        #     plt.clf()
        #     plt.plot(local_x, local_y, 'bo')
        #     fig.savefig('individual/'+ str(ii) +'_origin_switch.png')
        #     plt.clf()
        #     curve_reconstruction[output_coor.astype(int)+ center[0],input_for_curve.astype(int) + center[1]] = curvatures

#
#
#
#         curve_yy = poly_yy(t)

#
#         individual_segment_map_skel = skeletonize(1.0 * (seg_map_erode > 0))
#
#         # Compute gradients
#         # GX = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=5, scale=1)
#         # GY = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=5, scale=1)
#
#
#         curvature_map_mask = curvature_map_mask + individual_segment_map_skel
#         # individual_segment_map_skel = cv2.cvtColor(np.float32(individual_segment_map_skel), cv2.COLOR_BGR2GRAY)
#         # ret, thresh = cv2.threshold(np.uint8(individual_segment_map_skel), 0.1, 1, 0)
#
#         intersections, _ = get_skeleton_intersection_and_endpoint(individual_segment_map_skel)
#         for p in intersections:
#             individual_segment_map_skel[p[1],p[0]] = 0
#
#         contours, hierarchy = cv2.findContours(np.uint8(individual_segment_map_skel),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         selected_contour = contours[0]
#         for a_contour in contours:
#             if len(a_contour) > len(selected_contour):
#                 contours[0] = a_contour
#                 selected_contour = a_contour
#
#         individual_segment_map_temp = 1.0 * (individual_segment_map > 0)
#
#         (x,y),(MA,ma),angle = cv2.fitEllipse(selected_contour)
#         if len(selected_contour) / 2 < 2 * step:
#
#             # import pdb;pdb.set_trace()
#             curvature_map[selected_contour.squeeze()[:,1],selected_contour.squeeze()[:,0]] = 0.001
#             continue
#
#         sorted_countour_like_array = selected_contour[:int(len(selected_contour)/2)]
#
# #####################################
#         # only pick one point
#         one_end_point = (np.where(individual_segment_map == np.max(individual_segment_map))[0][0],np.where(individual_segment_map == np.max(individual_segment_map))[1][0])
#
#         phi_segment = 1.0 * np.ones(map_shape)
#         phi_segment[one_end_point] = -1
#         phi_segment  = np.ma.MaskedArray(phi_segment, ~(individual_segment_map_temp>0))
#         individual_segment_dist_skfmm = skfmm.distance(phi_segment, dx=1)
#         individual_segment_dist_map = individual_segment_dist_skfmm.data
#
#
#         y_, x_= np.where(individual_segment_map_skel > 0)
#         y_ = selected_contour.squeeze()[:,1]
#         x_ = selected_contour.squeeze()[:,0]
#         row_local = np.max(y_) - np.min(y_) + 1
#         col_local = np.max(x_) - np.min(x_) + 1
#
#
#
#         center = (int(row_local/2), int(col_local/2))
#
#         local_center_y = local_center[0] # curve center
#         local_center_x = local_center[1] # curve center
#         shift_local_curve_center_y = int(local_center_y) - np.min(y_)
#         shift_local_curve_center_x = int(local_center_x) - np.min(x_)
#
#         global_center_y = np.min(y_)
#         global_center_x = np.min(x_)
#
#         coordinates_center_y = np.min(y_) + int(row_local/2)
#         coordinates_center_x = np.min(x_) + int(col_local/2)
#
#         local_y = y_ - global_center_y
#         local_x = x_ - global_center_x
#
#         print(angle)
#         import pdb; pdb.set_trace()
#
#
#         local_segment_map = np.zeros((row_local, col_local))
#         local_segment_map[local_y, local_x] = individual_segment_dist_map[y_,x_]
#
#         sorted_index = np.unravel_index(np.argsort(local_segment_map, axis=None), local_segment_map.shape)
#         # # import pdb;pdb.set_trace()
#         # # for ind in range(len(sorted_index[0])):
#
#         num_pixels_in_seg = len(np.where(local_segment_map>0)[0])
#         num_pixels_in_seg = int(len(y_) / 2)
#         if num_pixels_in_seg < 2 * step:
#             continue
#         filtered_sorted_index = [sorted_index[0][-num_pixels_in_seg:],sorted_index[1][-num_pixels_in_seg:]]
#         local_segment_map_test = np.zeros((row_local, col_local))
#
#
#         ############################################################
#         t = np.arange(num_pixels_in_seg) - int(num_pixels_in_seg / 2)
#         step_t = 1
#         t = t[step_t:-step_t]
#         # print(t)
#         # import pdb; pdb.set_trace()
#         curve_y = np.polyfit(t, sorted_index[0][-num_pixels_in_seg+step_t: -step_t] - shift_local_curve_center_y, 3)
#         curve_x = np.polyfit(t, sorted_index[1][-num_pixels_in_seg+step_t: -step_t] - shift_local_curve_center_x, 3)
#         poly_xx = np.poly1d(curve_x)
#         poly_yy = np.poly1d(curve_y)
#
#
#         curve_xx = poly_xx(t)
#         curve_yy = poly_yy(t)
#
#
#         poly_xx_d = np.polyder(poly_xx)
#         poly_yy_d = np.polyder(poly_yy)
#
#         poly_xx_dd = np.polyder(poly_xx_d)
#         poly_yy_dd = np.polyder(poly_yy_d)
#
#         k_curvature = np.abs(poly_xx_d(t) * poly_yy_dd(t) - poly_yy_d(t) * poly_xx_dd(t)) / \
#                      ((poly_xx_d(t) ** 2 + poly_yy_d(t) ** 2) ** (3/2))
#
#     #     def reject_outliers(data, m=2):
#     # return data[abs(data - np.mean(data)) < m * np.std(data)]
#         print ('-------')
#
#         print(np.mean(k_curvature))
#         print(np.std(k_curvature))
#
#         # k_curvature[abs(k_curvature - np.mean(k_curvature)) > np.std(k_curvature)] = np.mean(k_curvature)
#         # import pdb; pdb.set_trace()
#
#         fig = plt.figure(1)
#         # import pdb; pdb.set_trace()
#         plt.plot(curve_yy, curve_xx)
#         fig.savefig('individual/'+ str(ii) +'_fitted.png')
#         # plt.show()
#         print(k_curvature)
#         print ('-------')
#
#         plt.clf()
#         fig_2 = plt.figure(2)
#         plt.plot(sorted_index[1][-num_pixels_in_seg:] - int(col_local/2), sorted_index[0][-num_pixels_in_seg:] - int(row_local/2))
#         fig_2.savefig('individual/'+ str(ii) +'_origin.png')
#         plt.clf()
#         # if ii == 35:
#         #     import pdb; pdb.set_trace()
#         # plt.show()
#
#         local_segment_map_curve = np.zeros((row_local + 1, col_local + 1))
#         # local_segment_map_curve[curve_xx.astype(int) + int(row_local/2), curve_yy.astype(int) + int(col_local/2)] = 1.0
#         # import pdb; pdb.set_trace()
#         curve_reconstruction[curve_yy.astype(int) + int(row_local/2) + global_center_y,\
#                             curve_xx.astype(int) + int(col_local/2) + global_center_x] = k_curvature
#         # import pdb; pdb.set_trace()


        # cv2.imshow('tt',seg_map_erode)
        # cv2.imshow('ttt',individual_segment_map_temp)
        # cv2.imshow('t',local_segment_map_toshow)
        # cv2.imshow('ttttt',curve_reconstruction)
        # cv2.imwrite('individual/' + str(ii) + '_origin.png',local_segment_map_toshow)
        # cv2.waitKey(0)

        # # tck, u = interpolate.splprep([filtered_sorted_index[0], filtered_sorted_index[1]], s=0)
        # # unew = np.arange(0,len(filtered_sorted_index[0]))
        # # out = interpolate.splev(unew, tck)
        # # import pdb;pdb.set_trace()
        # # plt.figure()
        # # plt.plot(out[0], out[1], 'r')
        # # plt.plot(filtered_sorted_index[0], filtered_sorted_index[1], 'g')
        # # plt.show()
        # # import pdb;pdb.set_trace()

        # sorted_countour_like_array = np.expand_dims(np.vstack((filtered_sorted_index[1] + global_center_x, filtered_sorted_index[0] + global_center_y)).transpose(), axis=1)
        # number of points X 1 X 2
        # pfit = np.polyfit(x_, y_, 4);


        # import pdb;pdb.set_trace()
        # sx_ = np.polyfit(np.arange(len(x_)), x_, 3)
        # sy_ = np.polyfit(np.arange(len(y_)), y_, 3)

        # sx_p = np.poly1d(sx_)
        # sy_p = np.poly1d(sy_)

        # print(sx_p(np.arange(len(x_))) - x_)
        # print(sy_p(np.arange(len(y_))) - y_)

        # canvas[sx_p(np.arange(len(x_))).astype(int), sy_p(np.arange(len(y_))).astype(int)] = 0
        # cv2.imshow('test', canvas * 255.)
        # cv2.waitKey(0)
        # import pdb;pdb.set_trace()


        # if(len(selected_contour)/2 < 2 * step):
        #     continue

        # import pdb;pdb.set_trace()
        # half_selected_contour = selected_contour[: int(len(selected_contour)/2)].squeeze()
        # x_ = half_selected_contour[:,0]
        # y_ = half_selected_contour[:,1]

        # x_, y_ = np.where(individual_segment_map_skel)
        # import pdb;pdb.set_trace()
        # tck, u = interpolate.splprep([x_, y_], s=0)
        # unew = np.arange(0,len(x_))
        # out = interpolate.splev(unew, tck)
        # import pdb;pdb.set_trace()
        # plt.figure()
        # plt.plot(out[0], out[1])
        # plt.show()
        # import pdb;pdb.set_trace()
        # pfit = np.polyfit(x_, y_, 4);

        # dx = np.polyder(pfit);
        # ddx = np.polyder(dx);
        # curvature = np.polyval(ddx, x_) / np.power((1 + np.power(np.polyval(dx, y_), 2)), 1.5);
        # import pdb;pdb.set_trace()
        # curvature_map_2[y_, x_] = curvature
        # print(curvature)

        # cv2.imshow('curture', 1.0 * (np.abs(curvature_map)>0))
        # cv2.waitKey(0)


        # tck, u = interpolate.splprep(selected_contour.squeeze().transpose(), s=0)
        # unew = np.arange(0,leng(selected_contour))







        # curvatures, angle_cur = get_curvature( np.squeeze(sorted_countour_like_array[:int(len(sorted_countour_like_array)):step], axis=1))
#        curvatures, angle_cur = get_curvature_v2(np.squeeze(sorted_countour_like_array, axis = 1), step = 5)

        # for sample_point_index in range(len(curvatures)):

#        swap_for_cv2 = np.squeeze(sorted_countour_like_array)[:,[1,0]].transpose()
#        curvature_map[swap_for_cv2[0,:],swap_for_cv2[1,:]] = angle_cur
        # try:
            # curvatures, angle_cur = get_curvature( np.squeeze(contours[0][:int(len(contours[0])/2):step], axis=1))
        # except:
        #     curvature_map[contours[0].squeeze()[:,1],contours[0].squeeze()[:,0]]=0
        #     curvatures = [0.0]
        #     angle_cur = [0.000001]

        # print(angle_cur)


        # import pdb;pdb.set_trace()
        ######################################################
        # contour_position_index = sorted_countour_like_array.squeeze()# contour used opencv, so index is reversed
        contour_position_index = selected_contour# contour used opencv, so index is reversed
        contour_position_index_trans = (contour_position_index[:,1], contour_position_index[:,0])
        # curvatures, angle_cur = get_curvature( np.squeeze(contour_position_index[:int(len(sorted_countour_like_array)):step], axis=1))
        curvatures, angle_cur = get_curvature_v2(contour_position_index, 5)

        curvature_map[contour_position_index_trans] = angle_cur
        #
        # import pdb; pdb.set_trace()
        # start_index = 0
        # for sample_point_index in range(len(curvatures)):
        #     if (start_index + int(step / 2) + step) > int(len(contours[0])):
        #         start_index = start_index + step - 1
        #
        #     else:
        #         if np.isnan(curvatures[sample_point_index]):
        #             insertion = 0.0001
        #             insertion_angle_cur = 0.00001
        #         elif curvatures[sample_point_index] >= 0.2:
        #             insertion = 0.0001
        #             insertion_angle_cur = 0.00001
        #         else:
        #             insertion = curvatures[sample_point_index]
        #             # import pdb;pdb.set_trace()
        #             insertion_angle_cur = angle_cur[sample_point_index]
        #
        #
        #         # curvature_map[contour_position_index[start_index:start_index+step, 1],contour_position_index[start_index:start_index+step, 0]]= insertion
        #         draw_line_fill_gap = np.zeros_like(seg_mask)
        #         # draw_line_index = (contour_position_index[start_index:start_index+step, 1],contour_position_index[start_index:start_index+step, 0])
        #
        #         # import pdb;pdb.set_trace()
        #         # for i in range(len(draw_line_index[0])):
        #         #     if i > len(draw_line_index) - 2:
        #         #         break
        #         #     rr, cc = line(draw_line_index[0][i],draw_line_index[1][i],draw_line_index[0][i + 1],draw_line_index[1][i + 1])
        #         #     # print(draw_line_index)
        #         #     # print(rr)
        #         #     # print(cc)
        #         #     curvature_map[rr,cc] = insertion_angle_cur + 0.0001
        #         try:
        #             curvature_map[contour_position_index[start_index + int(step / 2): start_index + int(step / 2) + step, 1],contour_position_index[start_index + int(step / 2): start_index+ int(step / 2)+ step, 0]]= insertion_angle_cur + 0.1
        #             curvature_map[contour_position_index[start_index + int(step / 2): start_index + int(step / 2) + step, 1],contour_position_index[start_index + int(step / 2): start_index+ int(step / 2)+ step, 0]]= insertion_angle_cur + 0.1
        #         except:
        #             import pdb;pdb.set_trace()
        #         if np.max(angle_cur) > 1.4:
        #             row_here = np.max(contour_position_index[:, 1]) - np.min(contour_position_index[:, 1])
        #             col_here = np.max(contour_position_index[:, 0]) - np.min(contour_position_index[:, 0])
        #             canvas = np.zeros((row_here + 1,col_here + 1))
        #             # import pdb;pdb.set_trace()
        #             canvas[contour_position_index[:,1] - np.min(contour_position_index[:, 1]), contour_position_index[:,0]- np.min(contour_position_index[:, 0])] = 1
        #             # cv2.imshow(' canvas', canvas)
        #             # cv2.waitKey(0)
        #             # import pdb;pdb.set_trace()
        #         start_index = start_index + step - 1
        # ########################################################################################################################################
        # # cv2.imshow('t', 255. * (curvature_map > 0))
        # cv2.waitKey()

#############################################3
###################################################
#########################################################
        #
        # for_curvature = np.where(individual_segment_map_skel>0)
        # points = np.vstack((for_curvature[0][:int(len(for_curvature[0])):3], for_curvature[1][:int(len(for_curvature[0])):3]))
        # # import pdb;pdb.set_trace()
        # try:
        #     curvatures = get_curvature(points.transpose())
        # except:
        #     curvature_map[for_curvature]=0.00001
        #     curvatures = 0.0
        # contour_position_index = points
        # start_index = 0
        # for sample_point_index in range(len(curvatures)):
        #     if (start_index) >= int(len(curvatures)):
        #         start_index = start_index + 3
        #         continue
        #     else:
        #         if np.isnan(curvatures[sample_point_index]):
        #             insertion = 0.00001
        #         else:
        #             insertion = curvatures[sample_point_index]
        #
        #         curvature_map[contour_position_index[0, start_index:start_index+3],contour_position_index[1, start_index:start_index+3]]= insertion
        #         start_index = start_index + 3
        # cv2.imshow('curvature_mapblack_w', (curvature_map>0)*255.0)
        # cv2.waitKey(0)


        #
        # max_curvaures = np.max(np.nan_to_num(curvatures))
        # min_curvaures = np.min(np.nan_to_num(curvatures))
        # mean_curvatures = np.mean(np.nan_to_num(curvatures))
        #
        # length_total = length_total + length_value
        # angle_total = angle_total + angle
        #
        # curvatures_total = curvatures_total +  mean_curvatures
        #
        # if write:
        #     length__.write(row, 0, row)
        #     length__.write(row, 1, float(length_value))
        #
        #     angle__.write(row, 0, row)
        #     angle__.write(row, 1, float(angle))
        #
        #     curvature__.write(row+1, 0,row)
        #     curvature__.write(row+1, 1, float(max_curvaures))
        #     curvature__.write(row+1, 2, float(min_curvaures))
        #     curvature__.write(row+1, 3, float(mean_curvatures))
        #
        #     for_k_mean_clustering.write(row+1, 0, float(obj_entry[0]))
        #     for_k_mean_clustering.write(row+1, 1, float(obj_entry[1]))
        #     for_k_mean_clustering.write(row+1, 2, float(obj_entry[2]))
        #     for_k_mean_clustering.write(row+1, 3, float(obj_entry[3]))
        # row = row + 1
    for_k_mean_clustering_book.save('objec_para.xlsx')
    list_object = np.zeros((len(data_to_save),), dtype=np.object)
    list_object[:] = data_to_save
    import scipy.io
    scipy.io.savemat('../../data_seg.mat', mdict={'list_object': list_object})
    # import pdb; pdb.set_trace()
    # curvature_map_mask = (curvature_map_mask > 0)
    # curvature_map = grey_dilation((curvature_map * 100).astype(int), np.ones((5,5)))
    # curvature_map = cv2.GaussianBlur(curvature_map, (7, 7), curvature_map.max())
    # k_curvature[abs(k_curvature - np.mean(k_curvature)) > np.std(k_curvature)] = np.mean(k_curvature)

    valid_pixel = curvature_map[np.where(curvature_map>0)]
    valid_pixel_indx = np.where(curvature_map>0)

    mean_val = np.mean(valid_pixel)
    std_val = np.std(valid_pixel)

    curve_reconstruction_diff = np.abs(curvature_map - mean_val)
    curve_reconstruction_diff[np.where(curve_reconstruction<=0)] = 0
    ind_outlier = np.where(curve_reconstruction_diff > std_val)

    curvature_map[ind_outlier] = mean_val
    curvature_map = curvature_map

    # curvature_map = cv2.medianBlur(curvature_map,5)
    # curvature_map = maximum_filter(curvature_map, size = 3)
    # curvature_map = cv2.GaussianBlur(curvature_map, (5, 5), curvature_map.max())
    # curvature_map_2 = cv2.GaussianBlur(curvature_map_2, (3, 3), curvature_map_2.max())

    # seg_mask = 1.0 * (seg_mask > 0)

    seg_mask = 1.0 * (curvature_map > 0)
    # import pdb;pdb.set_trace()
    show_curvature_map = normalize_include_neg_val(curvature_map)
    # show_curvature_map = curvature_map

    # curvature_map = curve_reconstruction
    # cmap = plt.get_cmap('hot')
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=curvature_map.min(), vmax=curvature_map.max())
    show_curvature_map_rgb = cmap(norm(curvature_map))

    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    # show_curvature_map_rgb = show_curvature_map_rgb * np.repeat(seg_mask[:, :, np.newaxis], 4, axis=2)
    show_curvature_map_r = show_curvature_map_rgb[:,:, 0]
    show_curvature_map_b = show_curvature_map_rgb[:,:, 1]
    show_curvature_map_c = show_curvature_map_rgb[:,:, 2]

    show_curvature_map_r[np.where(1 - seg_mask)] = 0
    show_curvature_map_rgb[:,:,0] = show_curvature_map_r
    show_curvature_map_b[np.where(1 - seg_mask)] = 0
    show_curvature_map_rgb[:,:,1] = show_curvature_map_b
    show_curvature_map_c[np.where(1- seg_mask)] = 0
    show_curvature_map_rgb[:,:,2] = show_curvature_map_c


    # cv2.imshow('curvature_map', show_curvature_map_rgb)
    plt.imsave('visualization/curvature_map_new.png', show_curvature_map_rgb)
    cv2.imwrite('visualization/curve_reconstruction.png', (curve_reconstruction>0) * 255.0)

    # cv2.imwrite('visualization/curvature_map_new.png', rgb_curvature)
    # cv2.waitKey(0)
    fig = plt.figure(figsize=tuple(map(lambda x : x/100, map_shape[::-1])), dpi=100)


    imgplot = plt.imshow(show_curvature_map_rgb)
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7, 0.9, 1.2], orientation='horizontal', fraction=0.046, pad=0.04)
    plt.savefig('curvature_map.png')
    # cv2.imshow('curvature_map', show_curvature_map_rgb)

    # cv2.waitKey(0)
    # if write:
    #     num = row
    #     summary__.write(1, 0,num)
    #     summary__.write(1, 1,float(length_total/num))
    #     summary__.write(1, 2,float(angle_total/num))
    #     summary__.write(1, 3,float(curvatures_total/num))
    #     book.save(filename + '.xlsx')

    print("--- %s seconds ---" % (time.time() - start_time))
    import pdb;pdb.set_trace()

####################################################################################
#####
    seg_mask_for_skel = seg_mask

    mask_keypoint = np.zeros_like(seg_mask)

    skel = skeletonize(seg_mask_for_skel)
    intersections, _ = get_skeleton_intersection_and_endpoint(skel)
    for p in intersections:
        mask_keypoint[p[1],p[0]] = 1


    kernel = np.ones((10,10))
    mask_keypoint = cv2.dilate(mask_keypoint, kernel)

    phi2 = copy.deepcopy(phi)

    for p in intersections:
        skel[p[1],p[0]] = 0
        phi2[p[1],p[0]] = -1

    dist_map = skfmm.distance(phi2, dx=1)
    distance_values_for_skel = dist_map.data

    normalized_distance_map_skel = dist_map.data

    normalized_distance_map_skel = normalize_include_neg_val(dist_map.data) # skfmm distances are negative value

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('jet')
    rgba_img_skel = cmap(normalized_distance_map_skel)
    rgb_org_skel = copy.deepcopy(rgba_img_skel)
    skel = np.asarray(skel, dtype='float32')
    skel = skel * (1-mask_keypoint)
    blobs_labels = measure.label(skel, background=0)
    labels = np.unique(blobs_labels)
    total_length_skel = 0
    num_of_labels = 0
    for _, i in enumerate(labels):

        if i == 0:
            continue
        else:
            total_length_skel = total_length_skel + np.max(distance_values_for_skel[np.where(blobs_labels==i)])
            num_of_labels = num_of_labels + 1
            # results_of_skel.append(np.max(distance_values_for_skel[np.where(blobs_labels==i)]))

    # import matplotlib.pyplot as plt
    # plt.imwrite(blobs_labels, cmap='nipy_spectral')
    # plt.show()

    # book = xlwt.Workbook()
    # sh = book.add_sheet('Sheet 1')



    # for row, length in enumerate(values):
    #     sh.write(row, 0, float(length))
    #     row = row + 1


    # for row, length in enumerate(results_of_skel):
    #     sh.write(row, 1, float(length))


    # book.save('test.xlsx')

    # import pylab as plt

    # fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    # (or if you have an existing figure)

    #####################################################################################333
    # fig = plt.gcf()
    # ax = fig.gca()

    # plt.title('Distance from the boundary')
    # canvas = (1 - seg_mask) * 255.0
    # color = (5)

    # import copy


    # for ii in range(len(max_point)):
    #     rgba_img = cv2.circle(rgba_img, (max_point[ii]['xy'][0], max_point[ii]['xy'][1]), 2, (255,0,0), -1)

    # for i in range(len(keypoints)):
    #     corr_xy = keypoints[i]['xy']
    #     rgb_org = cv2.circle(rgb_org, (corr_xy[0], corr_xy[1]), 1, (255,255,0), -1)

    # for p in intersections:
    #     rgb_org_skel = cv2.circle(rgb_org_skel, (p[0], p[1]), 1, (255,255,0), -1)


    # cv2.imshow('0_contour.png', rgb_org*1.0)
    # cv2.imshow('0_contour_skel.png', rgb_org_skel*1.0)
    # cv2.waitKey(0)
    # cv2.imshow('tttt.png', rgba_img*1.0)
    # cv2.waitKey(0)
    # # plt.show()
    # plt.contour(phi.mask, [0], linewidths=(1), colors='red')
    # plt.contour(phi,[0], linewidths=(3), colors='black')
    # plt.contour(ok, 15)
    # plt.colorbar()
    # plt.savefig('2d_phi_distance.png')
    # # plt.show()
    skel_len = total_length_skel
    num_skel = num_of_labels
    march_len = total_length_march
    num_march = num_of_objects
    return skel_len, num_skel, march_len, num_march

def compute_pointness(I, n=5):
    # Compute gradients
    # GX = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=5, scale=1)
    # GY = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=5, scale=1)
    GX = cv2.Scharr(I, cv2.CV_32F, 1, 0, scale=1)
    GY = cv2.Scharr(I, cv2.CV_32F, 0, 1, scale=1)
    GX = GX + 0.0001  # Avoid div by zero

    # Threshold and invert image for finding contours

    _, I = cv2.threshold(I, 100, 255, cv2.THRESH_BINARY_INV)

    # Pass in copy of image because findContours apparently modifies input.
    C, H = cv2.findContours(np.uint8(I.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    heatmap = np.zeros_like(I, dtype=np.float)
    pointed_points = []
    for contour in C:
        contour = contour.squeeze()
        measure = []
        N = len(contour)
        for i in range(N):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + n) % N]

            # Angle between gradient vectors (gx1, gy1) and (gx2, gy2)
            gx1 = GX[y1, x1]
            gy1 = GY[y1, x1]
            gx2 = GX[y2, x2]
            gy2 = GY[y2, x2]
            cos_angle = gx1 * gx2 + gy1 * gy2
            cos_angle /= (np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2)))
            angle = np.arccos(cos_angle)
            if cos_angle < 0:
                angle = np.pi - angle

            x1, y1 = contour[((2*i + n) // 2) % N]  # Get the middle point between i and (i + n)
            heatmap[y1, x1] = angle  # Use angle between gradient vectors as score
            measure.append((angle, x1, y1, gx1, gy1))

        _, x1, y1, gx1, gy1 = max(measure)  # Most pointed point for each contour

        # Possible to filter for those blobs with measure > val in heatmap instead.
        pointed_points.append((x1, y1, gx1, gy1))

    heatmap = cv2.GaussianBlur(heatmap, (3, 3), heatmap.max())
    return heatmap, pointed_points

def get_skeleton_intersection_and_endpoint(skeleton):

    def neighbour(x,y,image):
        """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
        return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];

    validEndpoint = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]];
    image = skeleton.copy();
    image[0,:] = 0;
    image[len(image) - 1,:]=0
    image[:,0] = 0
    image[:, len(image[0]) - 1] = 0
    intersections = list();
    endpoints = list();
    indexes = np.where(image>0);

    # import pdb; pdb.set_trace()
    for i in range(len(indexes[0])):
        neighbours = neighbour(indexes[0][i], indexes[1][i],image);
        valid = True;
        # import pdb;pdb.set_trace()
        if len(np.where(neighbours)[0]) > 2:
            intersections.append((indexes[1][i],indexes[0][i]))
        # if neighbours in validIntersection:
        #     intersections.append((indexes[1][i],indexes[0][i]))
        if neighbours in validEndpoint:
            endpoints.append((indexes[1][i], indexes[0][i]))

    ####### Filter intersections to make sure we don't count them twice or ones that are very close together
    # for point1 in intersections:
    #     for point2 in intersections:
    #         if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
    #             intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections, endpoints;


def resize_back_output_shape(input_map, output_shape):

    input_map_shape = input_map.shape
    output_map = cv2.resize(input_map, (int(output_shape[1]), int(output_shape[0])),interpolation = cv2.INTER_NEAREST )
    return output_map

def refine_next(keypoints, short_offsets, mid_offsets, num_steps):
    # short_offsets shape: C * H * w

    # x = keypoints['xy'][0]
    # y = keypoints['xy'][1]
    x = keypoints[0]
    y = keypoints[1]

    y_v = short_offsets.shape[1] # height
    x_h = short_offsets.shape[2] # width

    mid_offsets_h = mid_offsets[0]
    mid_offsets_v = mid_offsets[1]

    short_offsets_h = short_offsets[0]
    short_offsets_v = short_offsets[1]

    y = min(y_v - 1, int(y))
    if int(y) < 0:
        y = 0
    x = min(x_h - 1, int(x))

    if int(x) < 0:
        x = 0
    for i in range(num_steps):
        curr = (x, y)



        offset_x = mid_offsets_h[y, x]
        offset_y = mid_offsets_v[y, x]

        tmp_y = min(y_v - 1, int(y + offset_y))
        if int(y + offset_y) < 0:

            tmp_y = 0
        tmp_x = min(x_h - 1, int(x + offset_x))
        if int(x + offset_x) < 0:
            tmp_x = 0

        offset_x_n = offset_x + short_offsets_h[tmp_y, tmp_x]
        offset_y_n = offset_y + short_offsets_v[tmp_y, tmp_x]

        new_x = int(x + offset_x_n)
        new_y = int(y + offset_y_n)

        x = new_x
        y = new_y


        y = min(y_v - 1, int(y))
        if int(y) < 0:
            y = 0
        x = min(x_h - 1, int(x))
        if int(x) < 0:
            x = 0

    return (x, y)

def group_one_skel(st, endpoints, keypoints, short_offsets, mid_offsets_next):

    x = st[0]
    y = st[1]
    skel = []
    curr = (x,y)
    canvas = np.zeros(short_offsets[0].shape)
    cv2.circle(canvas, curr, 5, 1, 3)

    skel.append(curr)
    continue_flag = True
    count = 1
    count_2 = 1
    double_mid_offsets_next = mid_offsets_next * 2
    mid_offsets_next_to_use = mid_offsets_next
    length = 0
    # import pdb; pdb.set_trace()
    while (continue_flag):


        proposal_next = refine_next(curr, short_offsets, mid_offsets_next_to_use, 1)
        proposal_next_next = refine_next(proposal_next, short_offsets, mid_offsets_next, 1) # always mid offsets.
        curr_dir = [proposal_next[1] - curr[1], proposal_next[0] - curr[0]]
        next_dir = [proposal_next_next[1] - proposal_next[1], proposal_next_next[0] -  proposal_next[0]]
        curr_dir = np.asarray(curr_dir)
        next_dir = np.asarray(next_dir)
        angle = get_angle(curr_dir, next_dir)
        print (np.linalg.norm(curr_dir))
        print (np.linalg.norm(next_dir))
        print (angle)

        #   1 check if it is end point
        for i in range(len(endpoints)):
            if np.linalg.norm(np.asarray(proposal_next)-np.asarray(endpoints[i]['xy'])) <= 12:
                print ('endpoints_added')
                skel.append(proposal_next)
                cv2.circle(canvas, proposal_next, 5, 1, 3)
                return skel, canvas

        if np.linalg.norm(np.asarray(proposal_next)-np.asarray(proposal_next_next)) <= 10:
            print (np.linalg.norm(np.asarray(proposal_next)-np.asarray(proposal_next_next)))
            if count_2 == 0:
                skel.append(proposal_next)
                return skel, canvas
            else:
                skel.append(proposal_next)
                curr = proposal_next
                count_2 = 0

        count_2 = 1
        if angle < 45:
            tmp_ind = 0
            dist_min = np.inf
            for i in range(len(keypoints)):
                dist = np.linalg.norm(np.asarray(proposal_next)-np.asarray(keypoints[i]['xy']))
                if dist <= dist_min:
                    dist_min = dist
                    tmp_ind = i
            if dist_min < 12:
                proposal_next = (keypoints[tmp_ind]['xy'][0],keypoints[tmp_ind]['xy'][1])


            cv2.circle(canvas, proposal_next, 5, 1, 1)
            skel.append(proposal_next)
            curr = proposal_next

            mid_offsets_next_to_use = mid_offsets_next
            count = 1
            length += 1
            print(length)
            if length > 50:
                return skel, canvas
        else:
            print ('bigger than 45')
            if count == 1:
                mid_offsets_next_to_use = double_mid_offsets_next
                count = 0
            else:
                return skel, canvas


def group_skeletons(keypoints, end_points, mid_offsets_pre, mid_offsets_next, short_offsets):
    end_points.sort(key=(lambda kp: kp['conf']), reverse=True)
    skeletons = []
    # dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

    # skeleton_graph = {i:[] for i in range(config.NUM_KP)}

    # for i in range(config.NUM_KP):
    #     for j in range(config.NUM_KP):
    #         if (i,j) in config.EDGES or (j,i) in config.EDGES:
    #             skeleton_graph[i].append(j)
    #             skeleton_graph[j].append(i)

    while len(end_points) > 0:
        kp = end_points.pop(0)
        # if any([np.linalg.norm(kp['xy']-s[kp['id'], :2]) <= 10 for s in skeletons]):
        #     continue
        this_skel = np.zeros((3,3))
        this_skel[0, :2] = kp['xy']
        this_skel[0, 2] = kp['conf']

        from_kp = tuple(np.round(this_skel[0,:2]).astype('int32'))


        proposal_next = refine_next(kp, short_offsets, mid_offsets_next, 1)
        # proposal_prev = refine_next(kp, short_offsets, mid_offsets_pre, 1)

        # proposal_next = kp['xy'] + mid_offsets_next[kp['xy'][1], kp['xy'][0]]
        # proposal_prev = kp['xy'] + mid_offsets_pre[kp['xy'][1], kp['xy'][0]]

        matches_next = []
        matches_prev = []

        for i in range(len(keypoints)):

            if np.linalg.norm(proposal_next-keypoints[i]['xy']) <= 16:
                matches_next.extend[{'ind': i, 'kp':keypoints[i], 'dist':np.linalg.norm(proposal_next-match[1]['xy'])}]

            if np.linalg.norm(proposal_prev-keypoints[i]['xy']) <= 16:
                matches_prev.extend[{'ind': i, 'kp':keypoints[i], 'dist':np.linalg.norm(proposal_next-match[1]['xy'])}]

        if len(matches_next) == 0:
            continue

        matches_next.sort(key=(lambda kp:kp['dist']))
        next_kp = np.round(matches_next[0]['xy']).astype('int32')

        matches = [(i, keypoints[i]) for i in range(len(keypoints)) if np.linalg.norm(proposal_next-match[1]['xy']) <= 32]



        this_skel = np.zeros((config.NUM_KP, 3))

    return skeletons
