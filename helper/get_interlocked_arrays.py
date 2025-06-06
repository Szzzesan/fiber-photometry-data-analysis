import numpy as np

def get_interlocked_arrays(array_left, array_right, direction = 'widest'):
    outer = np.subtract.outer(array_right, array_left)
    arr_l_set = set()
    arr_r_set = set()
    if direction == 'widest':
        arr_r_idx = [
            np.nan if (np.where(outer[:, col] < 0)[0].size == 0) else np.where(outer[:, col] < 0)[0].max()
            for col in list(range(array_left.size))]
        arr_l_idx = [
            np.nan if (np.where(outer[row, :] < 0)[0].size == 0) else np.where(outer[row, :] < 0)[0].min()
            for row in list(range(array_right.size))]
        arr_l_set.add(0)
        arr_r_set.add(array_right.size - 1)
    elif direction == 'narrowest':
        arr_r_idx = [
            np.nan if (np.where(outer[:, col] > 0)[0].size == 0) else np.where(outer[:, col] > 0)[0].min()
            for col in list(range(array_left.size))]
        arr_l_idx = [
            np.nan if (np.where(outer[row, :] > 0)[0].size == 0) else np.where(outer[row, :] > 0)[0].max()
            for row in list(range(array_right.size))]
    else:
        print(f'Direction can only be widest or narrowest.')
    arr_l_set.update([x for x in set(arr_l_idx) if x == x]) # = {x for x in set(arr_l_idx) if x == x}
    arr_r_set.update([x for x in set(arr_r_idx) if x == x]) # = {x for x in set(arr_r_idx) if x == x}
    if len(arr_l_set) - len(arr_r_set) == 1:
        arr_l_set.discard(max(arr_l_set))
    elif len(arr_r_set) != len(arr_l_set):
        print(f'The interlocked arrays are not the same size. Need further debugging.')
    arr_r_idx = list(arr_r_set)
    arr_l_idx = list(arr_l_set)
    array_left = array_left[arr_l_idx]
    array_right = array_right[arr_r_idx]
    if array_left.size != array_right.size:
        print('The sizes of arrays do not match! Debug the function get_interlocked_arrays().')
    else:
        return array_left, array_right