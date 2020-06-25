import numpy  as np
import math

def label_map(value, label_map):
    if label_map.has_key(value[0]):
        value = [label_map[x] for x in value]
    else:
        label_map = {v: k for k, v in label_map.items()}
        print(label_map)
        value = [label_map[x] for x in value]
    return value

def sigmoid(feature):
    return 1 / (np.exp(-np.array(feature, dtype=np.float32)) + 1)

def softmax(feature):
    return np.exp(feature)/np.sum(np.exp(feature),axis=0)


def point2angle(x, y):
    angle = math.atan2(y, x) * 360 / 2 / math.pi
    return angle

def angle_diff(x, y):
    diff = np.abs(x - y)
    need_idx = np.where(diff > 180)
    diff[np.where(diff > 180)] = 360 - diff[np.where(diff > 180)]

    return diff
