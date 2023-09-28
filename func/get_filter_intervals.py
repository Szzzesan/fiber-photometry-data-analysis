import numpy as np


def get_filter_intervals(structured_events, left_col_name, right_col_name):
    left_arr = structured_events[left_col_name].to_numpy()
    right_arr = structured_events[right_col_name].to_numpy()
    intervals = np.stack((left_arr, right_arr), axis=1)
    return intervals