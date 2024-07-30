import os
import numpy as np

from matplotlib import pyplot as plt

import func


class OneSession:
    def __init__(self, animal_str, session, include_branch='both'):
        self.include_branch = include_branch

        lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
        animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
        raw_dir = os.path.join(animal_dir, 'raw_data')
        FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
        behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
        TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)

        self.behav_dir = os.path.join(animal_dir, 'raw_data', behav_file_list[session])
        self.signal_dir = os.path.join(animal_dir, 'raw_data', FP_file_list[session])
        self.arduino_dir = os.path.join(animal_dir, 'raw_data', TTL_file_list[session])
        self.fig_export_dir = os.path.join(animal_dir, 'figures', self.signal_dir[-21:-7])
        print("This class is" + " session " + self.signal_dir[-23:-7])
        self.pi_events, self.neural_events = func.data_read_sync(self.behav_dir, self.signal_dir, self.arduino_dir)
        if self.pi_events['task'].iloc[0] == 'single_reward':
            self.task = 'single_reward'
        elif self.pi_events['task'].iloc[0] == 'cued_no_forgo_forced':
            self.task = 'multi_reward'
        self.dFF0 = None
        self.transient_r = None
        self.transient_l = None
        self.transient_occur_r = None
        self.transient_occur_l = None
        self.transient_midmag_r = None
        self.transient_midmag_l = None
        self.corr_l = None
        self.corr_r = None
        self.intervals_df = None

    def examine_raw(self, save=0):
        func.check_framedrop(self.neural_events)
        raw_separated = func.de_interleave(self.neural_events, session_label=self.signal_dir[-23:-7], plot=1,
                                           save=save,
                                           save_path=self.fig_export_dir)

    def calculate_dFF0(self, plot=0, plot_middle_step=0, save=0):
        raw_separated = func.de_interleave(self.neural_events, session_label=self.signal_dir[-23:-7],
                                           plot=plot_middle_step,
                                           save=0, save_path=self.fig_export_dir)
        self.dFF0 = func.calculate_dFF0_Hamilos(raw_separated, session_label=self.signal_dir[-23:-7], plot=plot,
                                                plot_middle_steps=plot_middle_step, save=save,
                                                save_path=self.fig_export_dir)
        self.dFF0.name = 'dFF0'

    def process_behavior_data(self):
        self.pi_events["time_recording"] = (self.pi_events['time'] - self.neural_events.timestamps[0]) / 1000
        self.pi_events = func.data_reduction(self.pi_events, lick_tol=.01, head_tol=0.2)
        self.pi_events = func.add_2ndry_properties_to_pi_events(self.pi_events)
        self.pi_events.reset_index(drop=True, inplace=True)

    # this function would depend on the execution of self.process_behavior_data()
    def actual_leave_vs_adjusted_optimal(self, save=0):
        self.intervals_df = func.make_intervals_df(self.pi_events)
        func.visualize_adjusted_optimal(self.intervals_df, save=save, save_path=self.fig_export_dir)

    def plot_bg_heatmaps(self, save=0):
        cbarmin_l = int(np.percentile(self.dFF0['green_left'].values, 1.5) * 100)
        cbarmin_r = int(np.percentile(self.dFF0['green_right'].values, 1.5) * 100)
        cbarmax_l = self.dFF0['green_left'].max() * 100
        cbarmax_r = self.dFF0['green_right'].max() * 100

        if self.include_branch == 'both':
            branch_list = [['green_right', cbarmin_r, cbarmax_r], ['green_left', cbarmin_l, cbarmax_l]]
        elif self.include_branch == 'only_right':
            branch_list = [['green_right', cbarmin_r, cbarmax_r]]
        elif self.include_branch == 'only_left':
            branch_list = [['green_left', cbarmin_l, cbarmax_l]]
        else:
            print('Error: You need signals from at least one branch to plot a heatmap')

        condition_bg_reward = (self.pi_events['port'] == 2) & (self.pi_events['value'] == 1) & (
                self.pi_events['key'] == 'reward') & (self.pi_events['reward_order_in_trial'] == 1)
        condition_bg_entry = (self.pi_events['port'] == 2) & (self.pi_events['value'] == 1) & (
                self.pi_events['key'] == 'trial') & (self.pi_events.is_valid_trial)
        condition_bg_exit = (self.pi_events['port'] == 2) & (self.pi_events['value'] == 0) & (
                self.pi_events['key'] == 'head') & (self.pi_events.is_valid_trial)
        condition_exp_reward = (self.pi_events['port'] == 1) & (self.pi_events['value'] == 1) & (
                self.pi_events['key'] == 'reward') & (self.pi_events['reward_order_in_trial'] == 1)
        condition_exp_entry = (self.pi_events['port'] == 1) & (self.pi_events['value'] == 1) & (
                self.pi_events['key'] == 'head') & (self.pi_events.is_valid)
        condition_exp_exit = (self.pi_events['port'] == 1) & (self.pi_events['value'] == 0) & (
                self.pi_events['key'] == 'head') & (self.pi_events.is_valid)
        for branch in branch_list:
            df_for_heatmap_bg, df_bg_r_trial_info = func.construct_matrix_for_heatmap(self.pi_events, self.dFF0,
                                                                                      branch=branch[0],
                                                                                      vmin=-3, vmax=12,
                                                                                      time0_condition=condition_bg_entry,
                                                                                      filterleft_condition=condition_exp_exit,
                                                                                      filterright_condition=condition_bg_exit,
                                                                                      orderleft_condition=None,
                                                                                      orderright_condition=None,
                                                                                      time0_name=f'BgEntry {branch[0]}')
            func.plot_heatmap_from_matrix(df_for_heatmap_bg, df_bg_r_trial_info, cbarmin=branch[1],
                                          cbarmax=branch[2] * 0.37, split_block=1, save=save,
                                          save_path=self.fig_export_dir)


    def plot_heatmaps(self, save=0):
        cbarmin_l = int(np.percentile(self.dFF0['green_left'].values, 1.5) * 100)
        cbarmin_r = int(np.percentile(self.dFF0['green_right'].values, 1.5) * 100)
        cbarmax_l = self.dFF0['green_left'].max() * 100
        cbarmax_r = self.dFF0['green_right'].max() * 100
        if self.include_branch == 'both':
            func.plot_heatmap(self.pi_events, self.dFF0, 'green_left', cbarmin=cbarmin_l, cbarmax=cbarmax_l, save=save,
                              save_path=self.fig_export_dir)
            func.plot_heatmap(self.pi_events, self.dFF0, 'green_right', cbarmin=cbarmin_r, cbarmax=cbarmax_r, save=save,
                              save_path=self.fig_export_dir)
        elif self.include_branch == 'only_right':
            func.plot_heatmap(self.pi_events, self.dFF0, 'green_right', cbarmin=cbarmin_r, cbarmax=cbarmax_r, save=save,
                              save_path=self.fig_export_dir)
        elif self.include_branch == 'only_left':
            func.plot_heatmap(self.pi_events, self.dFF0, 'green_left', cbarmin=cbarmin_l, cbarmax=cbarmax_l, save=save,
                              save_path=self.fig_export_dir)
        elif self.include_branch == 'neither':
            print("No branch is available.")
        else:
            print("Error: The 'include_branch' argument can only be 'both', 'only_right', 'only_left', or 'neither'.")

    def extract_transient(self, plot_zscore=0, plot_dff0=0):
        if self.include_branch == 'both':
            self.transient_r = func.extract_transient_info('green_right', self.dFF0, self.pi_events,
                                                           plot_zscore=plot_zscore,
                                                           plot=plot_dff0)
            self.transient_l = func.extract_transient_info('green_left', self.dFF0, self.pi_events,
                                                           plot_zscore=plot_zscore,
                                                           plot=plot_dff0)
        elif self.include_branch == 'only_right':
            self.transient_r = func.extract_transient_info('green_right', self.dFF0, self.pi_events,
                                                           plot_zscore=plot_zscore,
                                                           plot=plot_dff0)
        elif self.include_branch == 'only_left':
            self.transient_l = func.extract_transient_info('green_left', self.dFF0, self.pi_events,
                                                           plot_zscore=plot_zscore,
                                                           plot=plot_dff0)
        elif self.include_branch == 'neither':
            pass
        else:
            print("Error: The 'include_branch' argument can only be 'both', 'only_right', 'only_left', or 'neither'.")

        # how many peaks/min in the first 9 minutes of recording & how large are the transients in the whole session
        if self.transient_l is not None:
            self.transient_occur_l = len(self.transient_l.index[self.transient_l['peak_time'] < 540]) / 9
            self.transient_midmag_l = self.transient_l['height'].median()
        if self.transient_r is not None:
            self.transient_occur_r = len(self.transient_r.index[self.transient_r['peak_time'] < 540]) / 9
            self.transient_midmag_r = self.transient_r['height'].median()

    # this function depends on the execution of function self.extract_transient()
    def visualize_correlation_scatter(self, save=0):
        if len(self.transient_l.index > 10):
            self.corr_l = func.visualize_trial_by_trial(self.transient_l, self.dFF0, 'green_left',
                                                        session_label=self.signal_dir[-23:-7],
                                                        plot=1, save=save, save_path=self.fig_export_dir,
                                                        left_or_right='left',
                                                        task=self.task)
        if len(self.transient_r.index > 10):
            self.corr_r = func.visualize_trial_by_trial(self.transient_r, self.dFF0, 'green_right',
                                                        session_label=self.signal_dir[-23:-7],
                                                        plot=1, save=save, save_path=self.fig_export_dir,
                                                        left_or_right='right',
                                                        task=self.task)


if __name__ == '__main__':
    test_session = OneSession('SZ047', 4, include_branch='only_right')
    # test_session.examine_raw(save=1)
    test_session.calculate_dFF0(plot=0, plot_middle_step=0, save=0)
    test_session.process_behavior_data()
    test_session.plot_bg_heatmaps()
    # test_session.actual_leave_vs_adjusted_optimal()
    # test_session.extract_transient(plot_zscore=0)
    # test_session.visualize_correlation_scatter()
    print("Hello")
