from func.find_closest_value import find_closest_value
import numpy as np
import pandas as pd


def trial_average(pi_events, dFF0, branch, aligner_condition, peri_event_range):
    aligner_times = pi_events.time_recording[aligner_condition].reset_index(drop=True)
    aligner_ids_in_dFF0 = np.empty(len(aligner_times))
    aligner_ids_in_dFF0.fill(np.nan)
    for i in range(len(aligner_times)):
        aligner_time = aligner_times[i]
        aligner_ids_in_dFF0[i] = find_closest_value(dFF0.time_recording, aligner_time)

    aligner_ids_in_dFF0 = aligner_ids_in_dFF0.astype(int)

    num_datapoint_prior = peri_event_range[0] * 40
    num_datapoint_post = peri_event_range[1] * 40

    dFF0_trial_average = pd.DataFrame(columns=['time_aligned', 'average', 'std_l', 'std_h'])
    sample_range = np.arange(num_datapoint_prior, num_datapoint_post)
    dFF0_trial_average['time_aligned'] = np.arange(num_datapoint_prior, num_datapoint_post) / 40
    col_name_in_dFF0 = 'green_' + branch
    for i in range(len(dFF0_trial_average.time_aligned)):
        dFF0_trial_average['average'].iloc[i] = dFF0[col_name_in_dFF0].iloc[
                                                    aligner_ids_in_dFF0 + sample_range[i]].mean() * 100
        dFF0_trial_average['std_l'].iloc[i] = dFF0_trial_average['average'].iloc[i] - dFF0[col_name_in_dFF0].iloc[
            aligner_ids_in_dFF0 + sample_range[i]].std() * 100
        dFF0_trial_average['std_h'].iloc[i] = dFF0_trial_average['average'].iloc[i] + dFF0[col_name_in_dFF0].iloc[
            aligner_ids_in_dFF0 + sample_range[i]].std() * 100

    return dFF0_trial_average
