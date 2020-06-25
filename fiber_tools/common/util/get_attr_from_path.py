'''
image_path = dir/sample_name/imagePrefix_fieldName_foucusName_exposures.jpg
fiber_path = dir/sample_name/imagePrefix_fieldName_foucusName_exposures_fiberIndex.jpg
'''
def get_exposure(fiber_path):
    exposure = int(fiber_path.split('_')[-2])
    return exposure

def get_field(image_path):
    cur_field_name = '_'.join(image_path.split('_')[:-2])
    return cur_field_name

def get_image_name(fiber_path):
    image_name = '_'.join(fiber_path.split('_')[:-1])
    return image_name

def get_sample_name(path):
    sample_name = '/'.join(path.split('/')[:-1])
    return sample_name

def get_sample_label(fiber_path):
    if '棉' in fiber_path and '麻' not in fiber_path:
        return 0
    elif '苎麻' in fiber_path and '棉' not in fiber_path:
        return 1
    elif '亚麻' in fiber_path and '棉' not in fiber_path:
        return 2
    elif '苎麻' in fiber_path and '棉' in fiber_path:
        return 3
    elif '亚麻' in fiber_path and '棉' in fiber_path:
        return 4
    elif '毛' in fiber_path and '绒' in fiber_path:
        return 5
    elif '绒' not in fiber_path and '毛' in fiber_path:
        return 6
    elif '毛' not in fiber_path and '绒' in fiber_path:
        return 7

    return -1