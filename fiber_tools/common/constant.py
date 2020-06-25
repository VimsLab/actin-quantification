SL  = {}       # single label dict
SL['MIAN'] = 0
SL['ZHUMA'] = 1
SL['YAMA'] = 2
SL['MAO'] = 3
SL['RONG'] = 4

SL_IDX_TO_LABEL  = {}       # single label dict
for label, idx in SL.items():
    SL_IDX_TO_LABEL[idx] = label

ML={}          # multiple label dict
ML['MIAN'] = 0
ML['ZHUMA'] = 1
ML['YAMA'] = 2
ML['MIAN_ZHUMA'] = 3
ML['MIAN_YAMA'] = 4
ML['MAO_RONG'] = 5
ML['MAO'] = 6
ML['RONG'] = 7

ML_IDX_TO_LABEL = {}
for label, idx in ML.items():
    ML_IDX_TO_LABEL[idx] = label

DENSITY = {}       # single label dict
DENSITY[0] = 1.0
DENSITY[1] = 1.0
DENSITY[2] = 1.0
DENSITY[3] = 1.0
DENSITY[4] = 1.0

def ml_label_trans(label_dict):
    assert len(label_dict)<=2, 'labels must less than two'

    if len(label_dict)==1:
        if SL['MIAN'] in label_dict:
            return ML['MIAN']
        if SL['ZHUMA'] in label_dict:
            return ML['ZHUMA']
        if SL['YAMA'] in label_dict:
            return ML['YAMA']
        if SL['RONG'] in label_dict:
            return ML['RONG']
        if SL['MAO'] in label_dict:
            return ML['MAO']

    if SL['MIAN'] in label_dict and SL['ZHUMA'] in label_dict:
        return ML['MIAN_ZHUMA']
    elif SL['MIAN'] in label_dict and SL['YAMA'] in label_dict:
        return ML['MIAN_YAMA']
    elif SL['MAO'] in label_dict and SL['RONG'] in label_dict:
        return ML['MAO_RONG']

TRAIN_LIST = ["fb_450_train_common", 
    "fb_second_450_train_common", 
    "fb_second_400_train_common", 
    "fb_third_common", 
    "fb_fourth_common", 
    "fb_fifth_common", 
    "fb_sixth_common"]

TEST_LIST = ["fb_second_100_test_common"]

FIBER_DICT = {
        "fb_500_train_common": {
            "img_dir": "Textile/to_label_file",
            "ann_file": "Textile/fb_coco_style.json",
        },
        "fb_450_train_common": {
            "img_dir": "Textile/train_first_data",
            "ann_file": "Textile/fb_coco_style_train.json",
        },
        "fb_50_test_common": {
            "img_dir": "Textile/test_first_data",
            "ann_file": "Textile/fb_coco_style_test.json",
        },
        "fb_second_450_train_common": {
            "img_dir": "Textile/exp_data/to_label_file_second",
            "ann_file": "Textile/second_data_annotation/fb_coco_style_train_second1.json",
        },
        "fb_second_400_train_common": {
            "img_dir": "Textile/exp_data/to_label_file_second",
            "ann_file": "Textile/second_data_annotation/fb_coco_style_train_second2.json",
        },
        "fb_second_100_test_common": {
            "img_dir": "Textile/exp_data/to_label_file_second",
            "ann_file": "Textile/second_data_annotation/fb_coco_style_test_second.json",
        },
        "fb_third_common": {
            "img_dir": "Textile/data_for_zfb/cls/",
            "ann_file": "Textile/data_annotation/common_instance_json_format/fb_coco_style_third.json",
        },
        "fb_fourth_common": {
            "img_dir": "Textile/data_for_zfb/cls/",
            "ann_file": "Textile/data_annotation/common_instance_json_format/fb_coco_style_fourth.json",
        },
        "fb_fifth_common": {
            "img_dir": "Textile/data_for_zfb/new_data/mian_zhuma_hard_data/selected_img/",
            "ann_file": "Textile/data_annotation/common_instance_json_format/fb_coco_style_fifth.json",
        },
        "fb_sixth_common": {
            "img_dir": "Textile/data_for_zfb/new_data/mao_rong_multi_data/cls/",
            "ann_file": "Textile/data_annotation/common_instance_json_format/fb_coco_style_sixth.json",
        }
    }
