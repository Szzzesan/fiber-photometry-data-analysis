import os
import func
from matplotlib import pyplot as plt
import pandas as pd
from single_session_analysis_v2 import single_session_analysis
import warnings


def animal_analysis(animal_str):
    print('----------' + "Start analyzing animal " + animal_str + '----------')
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
    raw_dir = os.path.join(animal_dir, "raw_data")
    FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    for i in range(len(FP_file_list)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            single_session_analysis(animal_dir, FP_file_list[i], TTL_file_list[i], behav_file_list[i])
    print('----------' + "Finish analyzing animal " + animal_str + '----------')


if __name__ == '__main__':
    animal_analysis(animal_str='SZ034')
