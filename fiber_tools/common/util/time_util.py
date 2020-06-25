import datetime

def get_cur_time():
    cur_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return cur_time