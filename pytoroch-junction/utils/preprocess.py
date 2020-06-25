import numpy as np
import config as cfg
import cv2

from utils.color_map import GenColorMap

def draw_mask(im, mask, color):

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    mask = mask>0
    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.uint8)

def visualize_label_map(im, label_map):
    CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)


    ids = np.unique(label_map)

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)
    # import pdb; pdb.set_trace()

    for idx in ids:
        if idx == 0:
            continue
        color = CLASS_2_COLOR[int(idx) + 1]

        r[label_map==idx] = color[0]
        g[label_map==idx] = color[1]
        b[label_map==idx] = color[2]

    combined = cv2.merge([r, g, b]) * 0.75+ im.astype(np.float32) * 0.25
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.uint8)

def visualize_offset(canvas, offset_h, offset_v):
    for x in range(0, canvas.shape[1], 5):
        for y in range(0, canvas.shape[0], 5):
            curr = (x, y)
            offset_x = offset_h[y, x]
            offset_y = offset_v[y, x]

            next_pt = (int(x + offset_x), int(y + offset_y))
            cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

    return canvas

def visualize_points(canvas, points_map):

    canvas[np.where(points_map>0)] = 1
    return canvas

def visualize_keypoint(canvas, keypoints):
    for i in range(len(pred_st_kp)):
        curr = (pred_st_kp[i]['xy'][0],pred_st_kp[i]['xy'][1])

        cv2.circle(canvas, curr, 3, 1, -1)
    return canvas

def create_position_index(height, width):
    """
    create width x height x 2  pixel position indexes
    each position represents (x,y)
    """
    position_indexes = np.rollaxis(np.indices(dimensions=(width, height)), 0, 3).transpose((1,0,2))
    return position_indexes


def get_keypoint_discs_offset(all_keypoints, offset_map_point, img_shape, radius):

    #WHY NOT JUST USE IMDILATE
    #TO DO: USE discs, then use the offsets map(single point), find the value. then Map back to discs.
    map_shape = (img_shape[0], img_shape[1])
    offset_map_circle = np.zeros(map_shape)
    offset_map_circle_debug = np.zeros(map_shape)
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    discs = [[] for _ in range(len(all_keypoints))]
    # centers is the same with all keypoints.
    # Will change later.
    centers = all_keypoints
    dists = np.zeros(map_shape+(len(centers),))
    for k, center in enumerate(centers):
        dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1)) #Return the distance map to the point.
    # import pdb; pdb.set_trace()
    if len(centers) > 0:
        try:
            inst_id = dists.argmin(axis=-1)   #To which points its the closest
        except:
            print ('argmin fail')
            import pdb; pdb.set_trace()
    count = 0
    for i in range(len(all_keypoints)):

        discs[i].append(np.logical_and(inst_id == count, dists[:,:,count]<= radius))
        # offset_map_circle_debug[discs[i][0]] = 1.0
        offset_map_circle[discs[i][0]] = offset_map_point[dists[:,:,count] == 0]
        count +=1
    # tmp = np.asarray(offset_map_circle_debug * 255.0)
    # cv2.imshow('t', tmp)
    # cv2.waitKey(0)
    return discs, offset_map_circle

def get_keypoint_discs(all_keypoints, img_shape, radius):

    map_shape = (img_shape[0], img_shape[1])
    offset_map_circle = np.zeros(map_shape)
    offset_map_circle_debug = np.zeros(map_shape)
    # import pdb; pdb.set_trace()
    idx = create_position_index(height = map_shape[0], width = map_shape[1])
    # idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    # discs = [[] for _ in range(len(all_keypoints))]
    discs = []
    # centers is the same with all keypoints.
    # Will change later.
    centers = all_keypoints
    dists = np.zeros(map_shape+(len(centers),))
    for k, center in enumerate(centers):
        dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1)) #Return the distance map to the point.
    if len(centers) > 0:
        inst_id = dists.argmin(axis=-1)   #To which points its the closest
    count = 0
    for i in range(len(all_keypoints)):

        discs.append(np.logical_and(inst_id == count, dists[:,:,count]<= radius))

        count +=1

    return discs


def compute_short_offsets(all_keypoints, discs, map_shape, radius):

    r = radius
    x = np.tile(np.arange(r, -r - 1, -1), [2 * r + 1, 1])
    y = x.transpose()
    m = np.sqrt(x*x + y*y) <= r
    kp_circle = np.stack([x, y], axis=-1) * np.expand_dims(m, axis=-1)

    def copy_with_border_check(map, center, disc):
        from_top = max(r-center[1], 0)
        from_left = max(r-center[0], 0)
        from_bottom = max(r-(map_shape[0]-center[1])+1, 0)
        from_right =  max(r-(map_shape[1]-center[0])+1, 0)
        # import pdb;pdb.set_trace()
        try:
            cropped_disc = disc[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right]
            map[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right, :][cropped_disc,:] = kp_circle[from_top:2*r+1-from_bottom, from_left:2*r+1-from_right, :][cropped_disc,:]
        except:
            print (center)
            print (from_left)
            print (from_top)
            print (from_right)
            print (from_bottom)
            import pdb;pdb.set_trace()

    offsets = np.zeros(map_shape+(2,)) #x offeset, y offset
    # import pdb;pdb.set_trace()
    for i in range(len(all_keypoints)):
        copy_with_border_check(offsets[:,:,0:2], (all_keypoints[i,0], all_keypoints[i,1]), discs[i])
                                                 # x col               # y, row
    canvas = np.zeros_like(offsets[:,:,0])

    canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
    # tmp = np.asarray(np.absolute(offsets[:,:,0] * 255.0))
    # cv2.imshow('t', canvas)
    # cv2.waitKey(0)
    return offsets, canvas

def compute_mid_offsets(all_keypoints, offset_map_h, offset_map_v, map_shape, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    # import pdb;pdb.set_trace()

    offsets = np.zeros(map_shape+(2,))
    canvas = np.zeros_like(offsets[:,:,0])
    for k, center in enumerate(all_keypoints):
        # import pdb;pdb.set_trace()
        next_point_h = center[0] - offset_map_h[(center[1],center[0])]
        next_point_v = center[1] - offset_map_v[(center[1],center[0])]

        next_point_center = (int(next_point_h), int(next_point_v))
        curr = (int(center[0]), int(center[1]))
        cv2.arrowedLine(canvas, curr, next_point_center, 1, 1)
        # m = discs[i]

        # import pdb;pdb.set_trace()
        dists = next_point_center - idx
        offsets[discs[k],0] = dists[discs[k],0]
        offsets[discs[k],1] = dists[discs[k],1]

    canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
    # cv2.imshow('t', canvas)
    # cv2.waitKey(0)
        # import pdb;pdb.set_trace()
    return offsets, canvas

def compute_closest_control_point_offset(all_keypoints, seg_mask, map_shape):
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    direct_dists = np.zeros(map_shape + (len(all_keypoints),))

    offsets = np.zeros(map_shape+(2,))
    offsets_tmp = np.zeros(map_shape+(2,))
    offsets_h = np.zeros(map_shape + (len(all_keypoints),))
    offsets_v = np.zeros(map_shape + (len(all_keypoints),))

    seg_mask_index = seg_mask > 0
    canvas = np.zeros_like(offsets[:,:,0])

    for k, center in enumerate(all_keypoints):
        curr = (int(center[0]), int(center[1]))

        dists = curr - idx
        # import pdb;pdb.set_trace()



        offsets_tmp[seg_mask_index,0] = dists[seg_mask_index,0]
        offsets_tmp[seg_mask_index,1] = dists[seg_mask_index,1]

        offsets_h[seg_mask_index,k] = dists[seg_mask_index,0]
        offsets_v[seg_mask_index,k] = dists[seg_mask_index,1]

        # canvas = visualize_offset(canvas, offsets_h[:,:,k], offsets_v[:,:,k])
        # cv2.imshow('t', canvas)
        # cv2.waitKey(0)

        direct_dists[:,:,k] = np.sqrt(np.sum(np.square(offsets_tmp), axis = 2)) # obtain the shortest dist to the control point


    try:
       closest_keypoints = np.argmin(direct_dists, axis=2)
    except:
        import pdb; pdb.set_trace()
        print ('argmin fail')



    closest_keypoints = closest_keypoints.flatten() # flatten (indices of the last dimension)

    ind = (np.arange(len(closest_keypoints)), closest_keypoints) # create indexes

    offsets_flatten_h = np.reshape(offsets_h, (-1, len(all_keypoints)))

    offsets_h_final = offsets_flatten_h[ind]

    offsets_flatten_v = np.reshape(offsets_v, (-1, len(all_keypoints)))

    offsets_v_final = offsets_flatten_v[ind]

    offsets_h_final = np.reshape(offsets_h_final, map_shape)
    offsets_v_final = np.reshape(offsets_v_final, map_shape)
    # import pdb;pdb.set_trace()

    offsets[:,:,0] = offsets_h_final
    offsets[:,:,1] = offsets_v_final
    # import pdb;pdb.set_trace()
    # canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
    # cv2.imshow('t', canvas)
    # cv2.waitKey(0)

    return offsets, canvas

def compute_mid_long_offsets(all_keypoints, offset_map_h, offset_map_v, map_shape, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    # import pdb;pdb.set_trace()

    dists = np.zeros(map_shape+(len(all_keypoints),))
    offsets = np.zeros(map_shape+(2,))
    canvas = np.zeros_like(offsets[:,:,0])
    for k, center in enumerate(all_keypoints):
        # import pdb;pdb.set_trace()
        next_point_h = center[0] - offset_map_h[(center[1],center[0])]
        next_point_v = center[1] - offset_map_v[(center[1],center[0])]
        next_next_point_h = next_point_h - offset_map_h[(int(next_point_v),int(next_point_h))]
        next_next_point_v = next_point_v - offset_map_v[(int(next_point_v),int(next_point_h))]


        next_next_point_center = (int(next_next_point_h), int(next_next_point_v))
        curr = (int(center[0]), int(center[1]))
        cv2.arrowedLine(canvas, curr, next_next_point_center, 1, 1)
        # m = discs[i]

        # import pdb;pdb.set_trace()
        dists = next_next_point_center - idx

        offsets[discs[k],0] = dists[discs[k],0]
        offsets[discs[k],1] = dists[discs[k],1]

    canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
    # cv2.imshow('t', canvas)
    # cv2.waitKey(0)
        # import pdb;pdb.set_trace()
    return offsets, canvas
