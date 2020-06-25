import sys
sys.path.insert(0, '../../')

from fiber_tools.common.util.file_operator import get_files_from_dir
from fiber_tools.common.util.get_attr_from_path import get_sample_label
from fiber_tools.common.util.init_util import parse 

def main(input_dir, output_file):
    w = open(output_file, 'w')
    file_names = get_files_from_dir(input_dir)
    for file_name in file_names:
        label = get_sample_label(file_name)
        w.write(file_name+','+str(label)+'\n')
    w.close()
    print('write file {} complete'.format(output_file))
 
if __name__ == '__main__':
    # python3 get_path_and_label.py --cfg ../config/preprocess.yml
    cfg = parse()
    input_dir = cfg.PREPROCCESS.INPUT_DIR
    output_file = cfg.PREPROCCESS.OUTPUT_File
    main(input_dir, output_file)

