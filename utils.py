"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import utils.pycocotools.mask as mask_utils
#from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
#from maskrcnn_benchmark.structures.keypoint import make_keypoints_from_cfg

IM_EXTs = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']


def is_file_exist(file):
    if os.path.exists(file) and os.path.isfile(file):
        return True
    return False


def is_endswith_image_ext(image_path):
    for im_ext in IM_EXTs:
        if image_path.endswith(im_ext):
            return True
    return False


def is_image_exist(image_path, strictly=False):
    '''strictly determine'''
    if not is_file_exist(image_path):
        return False
    if not is_endswith_image_ext(image_path):
        return False
    if not strictly:
        return True
    im = cv2.imread(image_path)
    if im is None:
        return False
    try:
        im = Image.open(image_path)
    except (IOError,) as e:
        print("Error: {}".format(e))
        return False
    try:
        im = np.array(im, np.float32)
    except (AttributeError, TypeError) as e:
        print("Error: {}".format(e))
        return False
    return True


def convert_to_jpg_for_compressing(im, comp_pert=85):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), comp_pert]
    flag, em_im = cv2.imencode('.jpg', im, encode_param)
    assert flag == True
    im = cv2.imdecode(em_im, 1)
    return im


def get_images_from_file(image_dir, file):
    images_paths = []
    with open(file) as f:
        for x in f.readlines():
            with open(file) as f:
                image_path = os.path.join(image_dir, x.strip())
                if is_image_exist(image_path):
                    images_paths.append(image_path)
    return images_paths


def get_images_from_dir(image_dir):
    '''Recursively to find images'''
    images_paths = []
    for _dir in os.listdir(image_dir):
        image_path = os.path.join(image_dir, _dir)
        flag = is_image_exist(image_path)
        if flag:
            images_paths.append(image_path)
            continue
        if image_path[-1] != "/":
            image_path = image_path + "/"
        if os.path.isdir(image_path):
            images_paths.extend(get_images_from_dir(image_path))
    return images_paths


def get_images_from_dir_or_file(image_dir, file):
    if is_image_exist(image_dir):
        return [image_dir,]
    if is_file_exist(file):
        return get_images_from_file(image_dir, file)
    return get_images_from_dir(image_dir)


def get_images_names_from_images_paths(image_dir, images_paths):
    if not isinstance(images_paths, (list, tuple)):
        images_paths = [images_paths,]
    if not image_dir.endswith("/"):
        image_dir = image_dir + "/"
    L = len(image_dir)
    images_names = [image_path[L:] for image_path in images_paths]
    return images_names


def show_image(im, title, cx=400, cy=0):
    cv2.imshow(title, im)
    cv2.moveWindow(title, cx, cy)
    key_code = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key_code


def resize_image(im, rs_size=1056.):
    if rs_size > 0:
        max_size = max(im.shape[:-1])
        scale = rs_size / max_size
        im = cv2.resize(im,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )
    return im


def get_colors(clen):
    import matplotlib.pyplot as plt
    colors = []
    for color in plt.cm.hsv(np.linspace(0, 1, clen)).tolist():
        color = [int(c * 255) for c in color]
        colors.append(color)
    return tuple(colors)


def draw_rectange(im, boxes, labels, classes, scores=None):
    if len(boxes) == 0 or len(labels) == 0:
        print("no boxes for visualizing")
    assert len(boxes) == len(labels)
    colors = get_colors(len(boxes))
    get_colors
    for ix, (bbox, label) in enumerate(zip(boxes, labels)):
        p1 = tuple(bbox[:2])
        p2 = tuple(bbox[2:])
        score = scores[ix] if scores else 0.0
        p3 = (bbox[0], (bbox[1] + bbox[3]) // 2)
        cv2.rectangle(im, p1, p2, colors[ix], 5)
        text = "{}: {:.4f}".format(classes[label], score)
        cv2.putText(
            im, text, p3, cv2.FONT_HERSHEY_SIMPLEX, 0.81, colors[ix], 2
        )
    return im


def draw_rectange_for_image(image, boxes, classes, scores):
    if len(boxes) == 0 or len(classes) == 0 or len(scores):
        print("no boxes for visualizing")
    assert len(boxes) == len(classes)
    assert len(boxes) == len(scores)
    colors = get_colors(len(boxes))
    get_colors
    for ix, (bbox, cls, score) in enumerate(zip(boxes, classes, scores)):
        p1 = tuple(bbox[:2])
        p2 = tuple(bbox[2:])
        p3 = (bbox[0], (bbox[1] + bbox[3]) // 2)
        cv2.rectangle(image, p1, p2, colors[ix], 5)
        text = "{}: {:.4f}".format(cls, score)
        cv2.putText(
            image, text, p3, cv2.FONT_HERSHEY_SIMPLEX, 0.81, colors[ix], 2
        )
    return image


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def pologons_to_mask(polygons, size):
    height, width = size
    # formatting for COCO PythonAPI


    try:
        rles = mask_utils.frPyObjects(polygons, height, width)
    except:
        import pdb; pdb.set_trace()

    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def mask_to_edge(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask, kernel)
    edge = mask.astype(int)^eroded.astype(int)
    return edge


def pologons_to_edge(polygons, size):
    mask = pologons_to_mask(polygons, size)
    return mask_to_edge(mask)


def mask_to_contours(mask):
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    _, contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def mask_to_pologons(mask):
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    _, contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            polygons.append(contour)
    return polygons


def edge_to_pologons(edge, size):
    polygons = mask_to_pologons(edge)
    return polygons


def edge_to_mask(edge, size):
    polygons = edge_to_pologons(edge, size)
    if len(polygons) == 0:
        return None

    # _polygons = []
    # for p in polygons:
    #     _polygons.extend(p)
    # polygons = [_polygons]

    contour = []
    for p in polygons:
        xs = p[0::2]
        ys = p[1::2]
        for x, y in zip(xs, ys):
            contour.append([x, y])
    contours = np.array(contour, dtype=np.int32)
    hull = cv2.convexHull(contours, False)
    polygons = [hull.flatten().tolist(),]

    mask = pologons_to_mask(polygons, size)
    return mask
