import numpy as np


def find_closest_value(s, x):
    idx = (np.abs(s - x)).argmin()
    return idx