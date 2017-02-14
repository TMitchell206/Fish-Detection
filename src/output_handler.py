import pandas as pd
import numpy as np

import os
import sys
import csv
import time
import datetime

def get_header():
    return ['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

def get_timestap():
    time_stamp = time.time()
    return datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M')

def gen_output_name():
    time_stamp = get_timestap()
    return '../output/OUTPUT'+ time_stamp + '.csv'

def get_filename_from_path(path):
    return path.rsplit('/',1)[1]

def gen_filenames_from_paths(file_paths):
    file_names = []
    for fp in file_paths:
        file_names.append(get_filename_from_path(fp))
    return file_names

def gen_output_frame(classifications, file_paths):
    output_file = []
    file_names = gen_filenames_from_paths(file_paths)
    for i in range(len(classifications)):
        output_file.append(np.append(file_names[i], classifications[i]))
    return pd.DataFrame(output_file)

def gen_output_csv(classifications, files_paths):
    output_file_name = gen_output_name()
    output_frame = gen_output_frame(classifications, files_paths)
    output_frame.to_csv(output_file_name, index=False, header=get_header())
