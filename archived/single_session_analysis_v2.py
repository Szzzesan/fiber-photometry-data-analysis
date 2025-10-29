import os

import numpy as np

import helper
from matplotlib import pyplot as plt
import pandas as pd
import sys
import warnings
from scipy.signal import find_peaks, peak_widths, peak_prominences


def single_session_analysis(animal_dir, signal_filename, arduino_filename, behav_filename, task='single_reward'):
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
    pi_events, neural_events = helper.data_read_sync(behav_dir, signal_dir, arduino_dir)

    # region Process neural data and export them
    # neural_events[neural_events.signal_type == 'actual'].green_right.to_numpy()
    helper.check_framedrop(neural_events)
    raw_separated = helper.de_interleave(neural_events, session_label=signal_dir[-23:-7], plot=0, save=0,
                                         save_path=fig_export_dir)
    dFF0 = helper.calculate_dFF0_Hamilos(raw_separated, session_label=signal_dir[-23:-7], plot=0,
                                         plot_middle_steps=0, save=0, save_path=fig_export_dir)
    dFF0.name = 'dFF0'
    # helper.export_df_to_csv(dFF0, fp_export_dir)
    # endregion

    # region Process behavior data and export them
    pi_events["time_recording"] = (pi_events['time'] - neural_events.timestamps[0]) / 1000
    pi_events = helper.data_reduction(pi_events, lick_tol=.01, head_tol=0.2)
    pi_events = helper.add_2ndry_properties_to_pi_events(pi_events)
    pi_events.reset_index(drop=True, inplace=True)

    # region Extract behavior events without trial structures
    non_trial_events = helper.extract_behavior_events(pi_events)
    non_trial_events.name = "nontrial_event_sec"
    # helper.export_df_to_csv(non_trial_events, event_export_dir)
    # endregion

    # region Extract behavior events in regard to trial structures
    # pi_trials = helper.extract_trial(pi_events)
    # structured_events = helper.get_structured_events(pi_trials)
    # endregion
    # endregion

    # region I. plot the heatmap with the histogram
    # region plot the heatmaps but divided by leave time
    # condition = (pi_events['port'] == 1) & (pi_events['key'] == 'reward') & (pi_events['value'] == 1) & (
    #         pi_events['reward_order_in_trial'] == 1)
    # for b in {'left', 'right'}:
    #     big_interval_for_reward1 = helper.get_filter_intervals(
    #         structured_events,
    #         'exp_entry', 'exp_exit')
    #
    #     helper.sensor_raster_plot(dFF0, pi_events, condition,
    #                             branch=b,
    #                             port='exp',
    #                             aligned_by='rewards',
    #                             sequence=0,
    #                             filter_intervals=big_interval_for_reward1,
    #                             bin_size=1 / 30,
    #                             plot_interval=[-1, 5],
    #                             fig_size=(10, 10),
    #                             plot_markers=True,
    #                             save=False, save_path=fig_export_dir,
    #                             sort=False, sort_direction='before')
    # endregion
    #
    # # # region plot the reward-to-reward heatmaps
    # # filter_for_reward1 = helper.get_filter_intervals(structured_events, 'exp_entry', 'exp_reward_2')
    # # filter_for_reward2 = helper.get_filter_intervals(structured_events, 'exp_reward_1', 'exp_reward_3')
    # # filter_for_reward3 = helper.get_filter_intervals(structured_events, 'exp_reward_2', 'exp_reward_4')
    # # for branch in {'left', 'right'}:
    # #     helper.sensor_raster_plot(dFF0, pi_events, pi_trials,
    # #                             branch=branch,
    # #                             port='exp',
    # #                             aligned_by='rewards',
    # #                             sequence=0,
    # #                             filter_intervals=filter_for_reward1,
    # #                             bin_size=1 / 30,
    # #                             plot_interval=[-2, 2],
    # #                             fig_size=(5, 10),
    # #                             plot_markers=0,
    # #                             save=1, save_path=fig_export_dir, sort=1, sort_direction='before')
    # # # endregion

    # endregion

    intervals_df, single = helper.make_intervals_df(pi_events, report_single=True)


    # region Ib. trial average neural signals
    bg_slow = (pi_events.is_valid_trial) & (pi_events.phase == '0.4') & (pi_events.port == 2) & (
            pi_events.key == 'trial') & (pi_events.value == 1)
    bg_fast = (pi_events.is_valid_trial) & (pi_events.phase == '0.8') & (pi_events.port == 2) & (
            pi_events.key == 'trial') & (pi_events.value == 1)
    bg_slow_trial_average_r = helper.trial_average(pi_events, dFF0, 'right', bg_slow, [-1, 11])
    bg_fast_trial_average_r = helper.trial_average(pi_events, dFF0, 'right', bg_fast, [-1, 11])
    bg_slow_trial_average_l = helper.trial_average(pi_events, dFF0, 'left', bg_slow, [-1, 11])
    bg_fast_trial_average_l = helper.trial_average(pi_events, dFF0, 'left', bg_fast, [-1, 11])
    helper.plot_average(bg_slow_trial_average_r, bg_fast_trial_average_r, save_path=fig_export_dir,
                        save_name='dFF0_all_averaged', left_or_right='right', save=1)
    helper.plot_average(bg_slow_trial_average_l, bg_fast_trial_average_l, save_path=fig_export_dir,
                        save_name='dFF0_all_averaged', left_or_right='left', save=1)
    # endregion

    # region II. trial-by-trial analysis
    transient_right = helper.extract_transient_info('green_right', dFF0, pi_events, plot_zscore=0, plot=0)
    transient_left = helper.extract_transient_info('green_left', dFF0, pi_events, plot_zscore=0, plot=0)
    r_left = helper.visualize_trial_by_trial(transient_left, dFF0, 'green_left', session_label=signal_dir[-23:-7],
                                             plot=1, save=1, save_path=fig_export_dir, left_or_right='left', task=task)
    r_right = helper.visualize_trial_by_trial(transient_right, dFF0, 'green_right', session_label=signal_dir[-23:-7],
                                              plot=1, save=1, save_path=fig_export_dir, left_or_right='right', task=task)

    # endregion

    # region III. slow component analysis
    # helper.slow_timescale_analysis(dFF0, pi_events, timescale_in_sec=30)
    # endregion
    print("Finish analyzing" + " session " + signal_filename[-23:-4])
    print("----------------------------------")
    median_transient_rt = transient_right['height'].median()
    median_transient_lft = transient_left['height'].median()
    return r_right, r_left, median_transient_rt, median_transient_lft


if __name__ == '__main__':
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    animal_str = 'SZ047'
    animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
    raw_dir = os.path.join(animal_dir, 'raw_data')
    FP_file_list = helper.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = helper.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = helper.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    num_session = len(TTL_file_list)
    df_across_session_right = pd.DataFrame(index=np.arange(num_session), columns=['median_peak',
                                                                                  'r2_R', 'r2_N', 'r2_X'])
    df_across_session_left = pd.DataFrame(index=np.arange(num_session), columns=['median_peak',
                                                                                 'r2_R', 'r2_N', 'r2_X'])
    reentry_slow_list = np.zeros(num_session)
    reentry_fast_list = np.zeros(num_session)

    for session in range(16, 17):
        # try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_right, r_left, median_ph_rt, median_ph_lft, reentry_index_slow, reentry_index_fast = single_session_analysis(
                animal_dir,
                FP_file_list[session],
                TTL_file_list[session],
                behav_file_list[session], task='multi_reward')
            df_across_session_right.iloc[session, 0] = median_ph_rt
            df_across_session_right.iloc[session, -3:] = r_right
            df_across_session_left.iloc[session, 0] = median_ph_lft
            df_across_session_left.iloc[session, -3:] = r_left
            reentry_slow_list[session] = reentry_index_slow
            reentry_fast_list[session] = reentry_index_fast

        # except:
        #     print("skipped session " + FP_file_list[session][-23:-7] + "!!!")
        #     print("----------------------------------")
        #     continue

    df_across_session_right = df_across_session_right.astype(float)
    df_across_session_left = df_across_session_left.astype(float)

    across_session_save_path = os.path.join(animal_dir, 'figures')
    fig_name_reentry = os.path.join(across_session_save_path, 'bg_reentry_across_session' + '.png')
    fig, ax = plt.subplots()
    plt.plot(reentry_slow_list, label='0.4 block')
    plt.plot(reentry_fast_list, label='0.8 block')
    plt.ylim([0.9, 2.5])
    plt.title(animal_str + ' Background re-entry index across session')
    ax.set_xlabel('Session')
    ax.set_facecolor("white")
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    fig.show()
    fig.savefig(fig_name_reentry)

    fig_name_r = os.path.join(across_session_save_path, 'r2_comparison_right' + '.png')
    fig_name_l = os.path.join(across_session_save_path, 'r2_comparison_left' + '.png')
    fig, ax = plt.subplots()
    df_across_session_right.boxplot(column=['r2_IRI_exc1',
                                            'r2_NRI_exc1',
                                            'r2_RXI_exc1',
                                            'r2_R', 'r2_N', 'r2_X', 'r2_RXI_end'],
                                    grid=False, rot=15)
    ax.set_ylabel('$R^{2}$')
    ax.set_xlabel('Type of Interval')
    plt.ylim([-0.02, 0.53])
    plt.title(animal_str + ' ' + 'Right ' + '$R^{2}$' + ' Comparison')
    ax.set_facecolor("white")
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    fig.show()
    fig.savefig(fig_name_r)

    fig, ax = plt.subplots()
    df_across_session_left.boxplot(column=['r2_IRI_exc1',
                                           'r2_NRI_exc1',
                                           'r2_RXI_exc1',
                                           'r2_R', 'r2_N', 'r2_X', 'r2_RXI_end'],
                                   grid=False, rot=15)
    ax.set_ylabel('$R^{2}$')
    ax.set_xlabel('Type of Interval')
    plt.ylim([-0.02, 0.53])
    plt.title(animal_str + ' ' + 'Left ' + '$R^{2}$' + ' Comparison')
    ax.set_facecolor("white")
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    fig.show()
    fig.savefig(fig_name_l)

    fig_name = os.path.join(across_session_save_path, 'r2_IRI_midreward' + '.png')
    fig, ax = plt.subplots()
    plt.plot(df_across_session_right.r2_IRI_exc1, label='right')
    plt.plot(df_across_session_left.r2_IRI_exc1, label='left')
    plt.legend()
    plt.xlabel('Session')
    plt.ylabel('$R^{2}$')
    plt.title('DA-IRI Correlation across session in ' + animal_str)
    fig.show()
    fig.savefig(fig_name)

    # fig_name = os.path.join(across_session_save_path, 'signal_strength_across_session' + '.png')
    # fig, ax = plt.subplots()
    # plt.plot(df_across_session_right['median_peak'] * 100, label='right')
    # plt.plot(df_across_session_left['median_peak'] * 100, label='left')
    # plt.ylim([0, 3.5])
    # plt.xlim([-1, 39])
    # plt.xlabel('Session')
    # plt.ylabel('Median dF/F0 (%)')
    # plt.title('Signal strength across session in ' + animal_str)
    # plt.legend()
    # fig.show()
    # fig.savefig(fig_name)
