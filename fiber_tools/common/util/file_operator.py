# -*-coding:utf-8-*-
import os

def get_files_from_dir(dir_name):

    def recur_dir(dir_name):
        file_list = os.listdir(dir_name)
        for file in file_list:
            cur_path = os.path.join(dir_name, file)
            if os.path.isdir(cur_path):   
                recur_dir(cur_path)
            else:
                file_names.append(cur_path)

    file_names = []
    recur_dir(dir_name)
    return file_names