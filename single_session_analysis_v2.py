import os

import numpy as np

import func
from matplotlib import pyplot as plt
import pandas as pd
import sys
import warnings
from scipy.signal import find_peaks, peak_widths, peak_prominences


def single_session_analysis(animal_dir, signal_filename, arduino_filename, behav_filename):
    behav_dir = os.path.join(animal_dir, 'raw_data', behav_filename)
    signal_dir = os.path.join(animal_dir, 'raw_data', signal_filename)
    arduino_dir = os.path.join(animal_dir, 'raw_data', arduino_filename)
    fp_export_dir = os.path.join(animal_dir, 'processed_data', signal_dir[-23:-4])
    event_export_dir = os.path.join(animal_dir, 'processed_data', behav_dir[-23:-4])
    fig_export_dir = os.path.join(animal_dir, 'figures', signal_dir[-21:-7])
    # print(os.path.exists(behav_dir))
    # print(os.path.exists(signal_dir))
    # print(os.path.exists(arduino_dir))
    print("Start analyzing" + " session " + signal_filename[-23:-7])
    pi_events, neural_events = func.data_read_sync(behav_dir, signal_dir, arduino_dir)

    # region Process neural data and export them
    # neural_events[neural_events.signal_type == 'actual'].green_right.to_numpy()
    func.check_framedrop(neural_events)
    raw_separated = func.de_interleave(neural_events, session_label=signal_dir[-23:-7], plot=0, save=0,
                                       save_path=fig_export_dir)
    dFF0 = func.calculate_dFF0(raw_separated, session_label=signal_dir[-23:-7], plot=0,
                               plot_middle_steps=0, save=0, save_path=fig_export_dir)
    dFF0.name = 'dFF0'
    func.export_df_to_csv(dFF0, fp_export_dir)
    # endregion

    # region Process behavior data and export them
    pi_events["time_recording"] = (pi_events['time'] - neural_events.timestamps[0]) / 1000
    pi_events = func.data_reduction(pi_events, lick_tol=.01, head_tol=0.2)
    pi_events = func.add_2ndry_properties_to_pi_events(pi_events)
    pi_events.reset_index(drop=True, inplace=True)


    # region Extract behavior events without trial structures
    non_trial_events = func.extract_behavior_events(pi_events)
    non_trial_events.name = "nontrial_event_sec"
    func.export_df_to_csv(non_trial_events, event_export_dir)
    # endregion

    # # region Extract behavior events in regard to trial structures
    pi_trials = func.extract_trial(pi_events)
    structured_events = func.get_structured_events(pi_trials)
    # endregion
    # endregion

    # region I. plot the heatmap with the histogram
    # region plot the heatmaps but divided by leave time
    # condition = (pi_events['port'] == 1) & (pi_events['key'] == 'reward') & (pi_events['value'] == 1) & (
    #             pi_events['reward_order_in_trial'] == 1)
    # for b in {'left', 'right'}:
    #     big_interval_for_reward1 = func.get_filter_intervals(
    #         structured_events[structured_events.exp_leave_time < structured_events.exp_leave_time.quantile(0.5)],
    #         'exp_entry', 'exp_exit')
    #     pi_df = pi_events
    #     pi_trial_df = pi_trials[pi_trials.leave_time < pi_trials.leave_time.quantile(0.5)].reset_index(drop=True)
    #
    #     func.sensor_raster_plot(dFF0, pi_df, pi_trial_df, condition,
    #                             branch=b,
    #                             port='exp', aligned_by='rewards', sequence=0, plot_markers=1,
    #                             filter_intervals=big_interval_for_reward1,
    #                             plot_interval=[-2, 10], save=0, save_path=fig_export_dir)
    # endregion
    # # region plot the reward-to-reward heatmaps
    # filter_for_reward1 = func.get_filter_intervals(structured_events, 'exp_entry', 'exp_reward_2')
    # filter_for_reward2 = func.get_filter_intervals(structured_events, 'exp_reward_1', 'exp_reward_3')
    # filter_for_reward3 = func.get_filter_intervals(structured_events, 'exp_reward_2', 'exp_reward_4')
    # for branch in {'left', 'right'}:
    #     func.sensor_raster_plot(dFF0, pi_events, pi_trials,
    #                             branch=branch,
    #                             port='exp',
    #                             aligned_by='rewards',
    #                             sequence=0,
    #                             filter_intervals=filter_for_reward1,
    #                             bin_size=1 / 30,
    #                             plot_interval=[-2, 2],
    #                             fig_size=(5, 10),
    #                             plot_markers=0,
    #                             save=1, save_path=fig_export_dir, sort=1, sort_direction='before')
    #     func.sensor_raster_plot(dFF0, pi_events, pi_trials,
    #                             branch=branch,
    #                             port='exp',
    #                             aligned_by='rewards',
    #                             sequence=1,
    #                             filter_intervals=filter_for_reward2,
    #                             bin_size=1 / 30,
    #                             plot_interval=[-2, 2],
    #                             fig_size=(5, 10),
    #                             plot_markers=0,
    #                             save=1, save_path=fig_export_dir, sort=1, sort_direction='before')
    #     func.sensor_raster_plot(dFF0, pi_events, pi_trials,
    #                             branch=branch,
    #                             port='exp',
    #                             aligned_by='rewards',
    #                             sequence=2,
    #                             filter_intervals=filter_for_reward3,
    #                             bin_size=1 / 30,
    #                             plot_interval=[-2, 2],
    #                             fig_size=(5, 10),
    #                             plot_markers=0,
    #                             save=1, save_path=fig_export_dir, sort=1, sort_direction='before')
    # #endregion
    # # region basic, with different behavior events at time 0
    # bp_grid = [(branch, [port, ratio]) for branch in ['left', 'right'] for [port, ratio] in [['bg', 1], ['exp', 2]]]
    # for b, [p, ratio] in bp_grid:
    #     for [aligner, sequence, interval_on, interval_off] in [['entries', 0, -1, 5 * ratio],
    #                                                            ['rewards', 0, -2, 4 * ratio],
    #                                                            ['exits', -1, -1 * ratio, 1]]:
    #         func.sensor_raster_plot(dFF0, pi_events, pi_trials, branch=b, port=p, aligned_by=aligner, sequence=sequence,
    #                                 plot_interval=[interval_on, interval_off], save=True, save_path=fig_export_dir)
    #
    # # endregion

    # endregion
    # region II. trial-by-trial analysis
    transient_left = func.extract_transient_info('green_left', dFF0, pi_events, plot=0)
    transient_right = func.extract_transient_info('green_right', dFF0, pi_events, plot=0)
    r_left = func.visualize_trial_by_trial(transient_left, dFF0, 'green_left')
    r_right = func.visualize_trial_by_trial(transient_right, dFF0, 'green_right')
    # endregion
    print("Finish analyzing" + " session " + signal_filename[-23:-4])


if __name__ == '__main__':
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    animal_str = 'SZ035'
    animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
    raw_dir = os.path.join(animal_dir, 'raw_data')
    FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    for session in range(8, 9):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            single_session_analysis(animal_dir, FP_file_list[session], TTL_file_list[session], behav_file_list[session])
