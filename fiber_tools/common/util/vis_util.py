import os
import pandas as pd
import numpy as np

def save_to_html(judge_results, label_name, 
    file_names, out_file, get_table):
    f = open(out_file, "w")

    for judge_result, file_name in zip(
        judge_results, file_names):
        if len(judge_result)==0:
            continue

        if not isinstance(judge_result, np.ndarray) and \
            isinstance(judge_result, list):
            judge_result = np.array(judge_result)
        titles, contents = get_table(
            judge_result, label_name, file_name)
        html_table = convert_to_html(contents, titles)
        f.write(html_table)
    f.close()
    print('complete confuse table {} saved.'.format(out_file))

def get_common_table(judge_res, label_names, file_names):
    
    titles = ['type']
    for i in range(len(label_names)):
        titles.append(label_names[i])
    
    contents = []
    name_line = []
    for i in range(len(file_names)):    
        name_line.append(file_names[i])
    contents.append(name_line)

    for i in range(len(label_names)):
        # import pdb;pdb.set_trace()
        try:
            # tmp_line = judge_res[:,i].astype(np.int64).tolist()
            tmp_line = judge_res[:,i].tolist()
        except:
            import pdb;pdb.set_trace()

        contents.append(tmp_line)

    return titles, contents

def convert_to_html(result, titles):
    table_dict = {}
    for idx, title in enumerate(titles):
        table_dict[title] = result[idx]

    df = pd.DataFrame(table_dict)
    df = df[titles]
    html_table =df.to_html(index=False)

    return html_table

def get_confuse_table(judge_res, label_names, file_names):
    titles, contents = get_common_table(
        judge_res, label_names, file_names
    )
    
    titles.append('all')
    titles.append('acc')
    acc_line = []
    sum_line = []
    corrects = 0.0
    for i in range(len(judge_res)):
        tmp_acc = judge_res[i][i]/(sum(judge_res[i])+1e-10)
        corrects+=judge_res[i][i]
        sum_line.append(sum(judge_res[i]))
        acc_line.append(tmp_acc)

    contents.append(sum_line)
    contents.append(acc_line)
    return titles, contents


    