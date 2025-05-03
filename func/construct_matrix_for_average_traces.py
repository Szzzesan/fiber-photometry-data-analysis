import numpy as np
import pandas as pd
import math
import func


def construct_matrix_for_average_traces(signal, branch, time_filterleft, time_filterright, trial, phase,
                                        interval_name=None):
    if (time_filterleft.shape[0]==0) or (time_filterright.shape[0]==0):
        raise ValueError("Time filter arrays must not be empty.")
    else:
        filter_time_dist = time_filterright - time_filterleft
        filter_frame_dist = [math.ceil(item / 0.025) for item in filter_time_dist]
        mat_col_num = max(filter_frame_dist)
        mat_row_num = trial.size
        np_mat = np.empty((mat_row_num, mat_col_num))
        np_mat[:] = np.nan

        for i in range(mat_row_num):
            time0_idx = func.find_closest_value(signal['time_recording'], time_filterleft[i])
            np_mat[i, 0:filter_frame_dist[i]] = signal[branch].iloc[time0_idx:time0_idx + filter_frame_dist[i]].to_numpy()

        assert np_mat.shape[1] == len(np.arange(0, np.round(mat_col_num * 0.025, decimals=3), 0.025))
        df_for_average = pd.DataFrame(columns=np.arange(0, np.round(mat_col_num * 0.025, decimals=3), 0.025),
                                      index=list(range(trial.size)),
                                      data=np_mat)
        df_for_average.branch = branch
        df_for_average.interval_name = interval_name
        df_trial_info = pd.DataFrame({'trial': trial, 'phase': phase})
        return df_for_average, df_trial_info
