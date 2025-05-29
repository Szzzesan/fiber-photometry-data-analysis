import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import func


class OneSession:
    def __init__(self, animal_str, session, include_branch='both', port_swap=0):
        self.include_branch = include_branch

        lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
        self.animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
        raw_dir = os.path.join(self.animal_dir, 'raw_data')
        # lab_dir = os.path.join('C:\\', 'Users', 'Valued Customer', 'Shichen')
        # self.animal_dir = os.path.join(lab_dir, animal_str)
        # raw_dir = os.path.join(self.animal_dir, 'raw_data')
        FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
        behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
        TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)

        self.animal = animal_str
        self.behav_dir = os.path.join(self.animal_dir, 'raw_data', behav_file_list[session])
        self.signal_dir = os.path.join(self.animal_dir, 'raw_data', FP_file_list[session])
        self.arduino_dir = os.path.join(self.animal_dir, 'raw_data', TTL_file_list[session])
        self.fig_export_dir = os.path.join(self.animal_dir, 'figures', self.signal_dir[-21:-7])
        self.processed_dir = os.path.join(self.animal_dir, 'processed_data')
        print("This class is" + " session " + self.signal_dir[-23:-7])
        self.pi_events, self.neural_events = func.data_read_sync(self.behav_dir, self.signal_dir, self.arduino_dir)
        if self.pi_events['task'].iloc[10] == 'single_reward':
            self.task = 'single_reward'
        elif self.pi_events['task'].iloc[10] == 'cued_no_forgo_forced':
            self.task = 'multi_reward'

        self.block_palette = sns.color_palette('Set2')
        self.port_swap = port_swap
        if self.port_swap:
            self.pi_events.loc[self.pi_events['port'] == 2, 'port'] = 3
            self.pi_events.loc[self.pi_events['port'] == 1, 'port'] = 2
            self.pi_events.loc[self.pi_events['port'] == 3, 'port'] = 1
        self.dFF0 = None
        self.zscore = None
        self.idx_taskbegin = None
        self.idx_taskend = None
        self.transient_r = None
        self.transient_l = None
        self.transient_occur_r = None
        self.transient_occur_l = None
        self.transient_midmag_r = None
        self.transient_midmag_l = None
        self.corr_l = None
        self.corr_r = None
        self.intervals_df = None
        self.NRI_amplitude_r = None
        self.NRI_amplitude_l = None
        self.IRI_amplitude_r = None
        self.IRI_amplitude_l = None
        self.NRI_amplitude, self.IRI_amplitude = {}, {}
        self.reward_features_DA = pd.DataFrame()
        self.DA_NRI_block_priorrewards = pd.DataFrame()
        self.DA_vs_NRI_IRI = pd.DataFrame()
        self.bg_behav_by_trial = pd.DataFrame()

    def examine_raw(self, save=0):
        func.check_framedrop(self.neural_events)
        raw_separated = func.de_interleave(self.neural_events, session_label=self.signal_dir[-23:-7], plot=1,
                                           save=save,
                                           save_path=self.fig_export_dir)

    def calculate_dFF0(self, plot=0, plot_middle_step=0, save=0):
        raw_separated = func.de_interleave(self.neural_events, session_label=self.signal_dir[-23:-7],
                                           plot=plot_middle_step,
                                           save=0, save_path=self.fig_export_dir)
        self.dFF0 = func.calculate_dFF0_Hamilos(raw_separated,
                                                session_label=f'{self.animal}: {self.signal_dir[-23:-7]}',
                                                plot=plot,
                                                plot_middle_steps=plot_middle_step, save=save,
                                                save_path=self.fig_export_dir)
        self.dFF0.name = 'dFF0'
        self.zscore = pd.DataFrame({'time_recording': self.dFF0.time_recording})
        self.zscore['green_right'] = stats.zscore(self.dFF0['green_right'].tolist(), nan_policy='omit')
        self.zscore['green_left'] = stats.zscore(self.dFF0['green_left'].to_list(), nan_policy='omit')

    def remove_outliers_dFF0(self):
        col_name_obj = self.dFF0.columns
        for branch in {col_name_obj[i] for i in range(1, col_name_obj.size)}:
            IQR = self.dFF0[branch].quantile(0.75) - self.dFF0[branch].quantile(0.25)
            upper_bound = self.dFF0[branch].quantile(0.75) + 3 * IQR
            lower_bound = self.dFF0[branch].quantile(0.25) - 1.5 * IQR
            self.dFF0.loc[self.dFF0[branch] >= upper_bound, branch] = np.nan
            self.dFF0.loc[self.dFF0[branch] < lower_bound, branch] = np.nan

    def process_behavior_data(self, save=0):
        self.pi_events["time_recording"] = (self.pi_events['time'] - self.neural_events.timestamps[0]) / 1000
        self.pi_events = func.data_reduction(self.pi_events, lick_tol=.01, head_tol=0.2)
        self.pi_events = func.add_2ndry_properties_to_pi_events(self.pi_events)
        # self.pi_events.reset_index(drop=True, inplace=True)
        self.idx_taskbegin = self.dFF0.index[
            self.dFF0['time_recording'] >= self.pi_events['time_recording'].min()].min()
        self.idx_taskend = self.dFF0.index[self.dFF0['time_recording'] <= self.pi_events['time_recording'].max()].max()
        if save:
            self.pi_events.to_csv(f"{self.animal_dir}/{self.signal_dir[-21:-7]}_pi_events.csv")

    # this function would depend on the execution of self.process_behavior_data()
    def actual_leave_vs_adjusted_optimal(self, save=0):
        self.intervals_df = func.make_intervals_df(self.pi_events)
        func.visualize_adjusted_optimal(self.intervals_df, save=save, save_path=self.fig_export_dir)

    def extract_bg_behav_by_trial(self):
        is_trial_start = (self.pi_events['key'] == 'trial') & (self.pi_events['value'] == 1)
        is_lick = (self.pi_events['key'] == 'lick') & (self.pi_events['value'] == 1)
        is_reward = (self.pi_events['key'] == 'reward') & (self.pi_events['value'] == 1)
        is_exit = (self.pi_events['key'] == 'head') & (self.pi_events['value'] == 0) & (self.pi_events['is_valid'])
        is_context = self.pi_events['port'] == 2
        bg_entries_df = self.pi_events[is_trial_start & self.pi_events['is_valid']]
        bg_entry_times = bg_entries_df['time_recording'].values
        bg_entry_trial_ids = bg_entries_df['trial'].values
        bg_exits_df = self.pi_events[is_exit & is_context]
        bg_exit_times = bg_exits_df['time_recording'].values
        bg_exit_trial_ids = bg_exits_df['trial'].values

        bg_licks_df = self.pi_events[is_lick & is_context]
        bg_lick_times = bg_licks_df['time_recording'].values
        bg_rewards_df = self.pi_events[is_reward & is_context]
        bg_reward_times = bg_rewards_df['time_recording'].values

        # examine if entries and exits match and correspond to the same trials
        are_trials_aligned_and_match = False  # Flag to indicate successful validation
        if len(bg_entry_trial_ids) == 0:
            print("No valid entry trials found. Cannot perform alignment check.")
            if len(bg_exit_trial_ids) == 0:
                print("No valid exit trials found either. Considering this 'aligned' as both are empty.")
                are_trials_aligned_and_match = True  # Or False, based on desired strictness
        elif len(bg_entry_trial_ids) != len(bg_exit_trial_ids):
            print(f"VALIDATION FAILED: Mismatch in the number of detected entries and exits.")
            print(f"Number of entry trials: {len(bg_entry_trial_ids)}")
            print(f"Number of exit trials: {len(bg_exit_trial_ids)}")
            # For debugging, you might want to see the trial IDs:
            # print(f"Entry trial IDs: {bg_entry_trial_ids}")
            # print(f"Exit trial IDs: {bg_exit_trial_ids}")
        elif 'trial' not in bg_exits_df.columns:  # Second check if exits_df was problematic
            print("VALIDATION SKIPPED for trial ID sequence: 'trial' column was missing in exit data.")
        else:
            # Lengths are the same, and both have trial IDs, now check if the sequences are identical
            if np.array_equal(bg_entry_trial_ids, bg_exit_trial_ids):
                print(f"VALIDATION PASSED: Entries and exits are aligned for {len(bg_entry_trial_ids)} trials.")
                print(f"Trial ID sequence: {bg_entry_trial_ids[:10]}..." if len(
                    bg_entry_trial_ids) > 10 else bg_entry_trial_ids)  # Print a sample
                are_trials_aligned_and_match = True
            else:
                print("VALIDATION FAILED: Entry and exit counts match, but their trial ID sequences differ.")
                # Find the first mismatch for a more specific error message
                mismatches = bg_entry_trial_ids != bg_exit_trial_ids
                first_mismatch_index = np.where(mismatches)[0]
                if len(first_mismatch_index) > 0:
                    idx = first_mismatch_index[0]
                    print(f"First mismatch occurs at index {idx}:")
                    print(f"  Entry trial ID: {bg_entry_trial_ids[idx]}, Exit trial ID: {bg_exit_trial_ids[idx]}")
                # Optionally, count total mismatches:
                # print(f"Total number of mismatched trial IDs: {np.sum(mismatches)}")
        # define the time window around port entry to plot
        time_window_before = -0.3
        time_window_after = 15
        aligned_licks_by_trial = []
        aligned_rewards_by_trial = []
        aligned_exits_by_trial = []
        if len(bg_entry_times) == 0:
            print("No port 2 entry events found. Cannot create raster plot.")
        else:
            for trial_idx, entry_time in enumerate(bg_entry_times):
                filter_left = entry_time + time_window_before
                filter_right = min(bg_exit_times[trial_idx], entry_time + time_window_after)
                trial_licks_abs = bg_lick_times[(bg_lick_times > filter_left) & (bg_lick_times < filter_right)]
                trial_rewards_abs = bg_reward_times[(bg_reward_times > filter_left) & (bg_reward_times < bg_exit_times[trial_idx])]
                trial_licks_relative = trial_licks_abs - entry_time
                trial_rewards_relative = trial_rewards_abs - entry_time
                trial_exits_relative = bg_exit_times[trial_idx] - entry_time
                aligned_licks_by_trial.append(list(trial_licks_relative))
                aligned_rewards_by_trial.append(list(trial_rewards_relative))
                aligned_exits_by_trial.append([trial_exits_relative])
        self.bg_behav_by_trial
    def bg_lick_rasterplot(self):
        is_trial_start = (self.pi_events['key'] == 'trial') & (self.pi_events['value'] == 1)
        is_lick = (self.pi_events['key'] == 'lick') & (self.pi_events['value'] == 1)
        is_reward = (self.pi_events['key'] == 'reward') & (self.pi_events['value'] == 1)
        is_exit = (self.pi_events['key'] == 'head') & (self.pi_events['value'] == 0) & (self.pi_events['is_valid'])
        is_context = self.pi_events['port'] == 2
        bg_entries_df = self.pi_events[is_trial_start & self.pi_events['is_valid']]
        bg_entry_times = bg_entries_df['time_recording'].values
        bg_exits_df = self.pi_events[is_exit & is_context]
        bg_exit_times = bg_exits_df['time_recording'].values
        bg_licks_df = self.pi_events[is_lick & is_context]
        bg_lick_times = bg_licks_df['time_recording'].values
        bg_rewards_df = self.pi_events[is_reward & is_context]
        bg_reward_times = bg_rewards_df['time_recording'].values
        # define the time window around port entry to plot
        time_window_before = -0.3
        time_window_after = 15
        aligned_licks_by_trial = []
        aligned_rewards_by_trial = []
        aligned_exits_by_trial = []
        if len(bg_entry_times) == 0:
            print("No port 2 entry events found. Cannot create raster plot.")
        else:
            for trial_idx, entry_time in enumerate(bg_entry_times):
                filter_left = entry_time + time_window_before
                filter_right = min(bg_exit_times[trial_idx], entry_time + time_window_after)
                trial_licks_abs = bg_lick_times[(bg_lick_times > filter_left) & (bg_lick_times < filter_right)]
                trial_rewards_abs = bg_reward_times[(bg_reward_times > filter_left) & (bg_reward_times < bg_exit_times[trial_idx])]
                trial_licks_relative = trial_licks_abs - entry_time
                trial_rewards_relative = trial_rewards_abs - entry_time
                trial_exits_relative = bg_exit_times[trial_idx] - entry_time
                aligned_licks_by_trial.append(list(trial_licks_relative))
                aligned_rewards_by_trial.append(list(trial_rewards_relative))
                aligned_exits_by_trial.append([trial_exits_relative])

        if not any(aligned_licks_by_trial):
            print("No licks found within the defined time window. Plot will be empty or not generated. ")
        else:
            fig, ax = plt.subplots(figsize=(10, max(6, len(bg_entry_times) * 0.3)))
            ax.eventplot(aligned_licks_by_trial, colors='darkgrey',
                         lineoffsets=np.arange(len(aligned_licks_by_trial)) + 1.5, linelengths=0.8, linewidths=1)
            ax.eventplot(aligned_rewards_by_trial, colors='blue',
                         lineoffsets=np.arange(len(aligned_licks_by_trial)) + 1.5, linelengths=1, linewidths=1.5)
            ax.eventplot(aligned_exits_by_trial, colors='red', lineoffsets=np.arange(len(aligned_licks_by_trial)) + 1.5,
                         linelengths=1, linewidths=1.5)
            ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Entry')
            # plt.eventplot(aligned_licks_by_trial, colors='black', lineoffsets=np.arange(len(aligned_licks_by_trial)),
            #               linelengths=0.8, linewidths=1)
            ax.set_ylim([1, len(aligned_licks_by_trial) + 1])
            ax.set_xlim([-0.3, 15])
            ax.set_yticks([1, 10, 20, 30, 40, 50])
            ax.invert_yaxis()
            fig.show()
            print('temporary pause here')

    def plot_bg_heatmaps(self, save=0):
        cbarmin_l = int(np.nanpercentile(self.dFF0['green_left'].values, 0.1) * 100)
        cbarmin_r = int(np.nanpercentile(self.dFF0['green_right'].values, 0.1) * 100)
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
            df_for_heatmap_bg, df_bg_r_trial_info, dfplot_entry, dfplot_exit, dfplot_reward, dfplot_lick = func.construct_matrix_for_heatmap(
                self.pi_events, self.dFF0,
                branch=branch[0],
                vmin=-1, vmax=12,
                time0_condition=condition_bg_entry,
                filterleft_condition=condition_exp_exit,
                filterright_condition=condition_bg_exit,
                orderleft_condition=None,
                orderright_condition=None,
                time0_name=f'BgEntry', order_name='RealTime')
            func.plot_heatmap_from_matrix(df_for_heatmap_bg, df_bg_r_trial_info, dfplot_entry, dfplot_exit,
                                          dfplot_reward, dfplot_lick, cbarmin=branch[1], cbarmax=branch[2] * 0.37,
                                          plot_lick=0, split_block=1, save=save, save_path=self.fig_export_dir)

    def plot_heatmaps(self, save=0):
        cbarmin_l = int(np.nanpercentile(self.dFF0['green_left'].values, 0.1) * 100)
        cbarmin_r = int(np.nanpercentile(self.dFF0['green_right'].values, 0.1) * 100)
        cbarmax_l = np.nanpercentile(self.dFF0['green_left'].iloc[self.idx_taskbegin:self.idx_taskend], 100) * 100
        cbarmax_r = np.nanpercentile(self.dFF0['green_right'].iloc[self.idx_taskbegin:self.idx_taskend], 100) * 100
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

        # how many peaks/min between 12 min and 17 min of recording & how large are the transients in the whole session
        if self.transient_l is not None:
            self.transient_occur_l = len(self.transient_l.index[(self.transient_l['peak_time'] > 720) & (
                    self.transient_l['peak_time'] < 1020)]) / 9
            self.transient_midmag_l = self.transient_l['height'].median()
        if self.transient_r is not None:
            self.transient_occur_r = len(self.transient_r.index[(self.transient_r['peak_time'] > 720) & (
                    self.transient_r['peak_time'] < 1020)]) / 9
            self.transient_midmag_r = self.transient_r['height'].median()

    # this function depends on the execution of function self.extract_transient()
    def visualize_correlation_scatter(self, plot=0, save=0):
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

    def extract_reward_features_and_DA(self, plot=0, save_dataframe=0):
        df_iri_exp = func.extract_intervals_expreward(
            self.pi_events,
            ani_str=self.animal,
            ses_str=self.signal_dir[-21:-7]
        )
        reward_times = df_iri_exp['reward_time'].to_numpy()
        peak_amps = {}

        for branch in ['green_right', 'green_left']:
            if branch not in self.dFF0.columns:
                continue  # Skip if the branch is not present

            peak_amp = np.full(reward_times.shape, np.nan)  # Initialize with NaNs

            for j, start in enumerate(reward_times):
                end = start + 0.5
                condition = (self.zscore['time_recording'] >= start) & (self.zscore['time_recording'] < end)
                signal_in_range = self.zscore.loc[condition, branch].to_numpy()

                if np.count_nonzero(~np.isnan(signal_in_range)) > 1:
                    peak_amp[j] = np.nanmax(signal_in_range) - np.nanmin(signal_in_range)

            peak_amps[branch] = peak_amp  # Store peak amplitude for this branch

        # Store processed data in the dataframe
        self.reward_features_DA = df_iri_exp.copy()
        for branch, amp_values in peak_amps.items():
            self.reward_features_DA[f'{branch}_DA_amp'] = amp_values
        if save_dataframe:
            self.reward_features_DA.to_csv(f'{self.animal_dir}/{self.signal_dir[-21:-7]}_reward_features_DA.csv')

        selected_columns = ['block', 'time_in_port', 'num_rewards_prior', 'green_right_DA_amp',
                            'green_left_DA_amp']
        df = self.reward_features_DA[selected_columns].copy()
        # Define fixed bin edges
        NRI_bins = [0, 3, 6, 9, 12]  # 4 bins: 0-3, 3-6, 6-9, 9-12
        num_reward_bins = [0, 2, 4, 6, 8, 10]  # 5 bins: 0-2, 2-4, 4-6, 6-8, 8-10
        # Assign bin labels
        df['time_in_port_bin'] = pd.cut(df['time_in_port'], bins=NRI_bins, labels=False)
        df['num_rewards_prior_bin'] = pd.cut(df['num_rewards_prior'], bins=num_reward_bins, labels=False)
        grouped = df.groupby(['block', 'time_in_port_bin', 'num_rewards_prior_bin'])
        DA_amp = grouped[['green_right_DA_amp', 'green_left_DA_amp']].agg(['mean', 'sem']).reset_index()
        # Rename columns for clarity
        DA_amp.columns = ['block', 'time_in_port_bin', 'num_rewards_prior_bin',
                          'right_DA_mean', 'right_DA_sem',
                          'left_DA_mean', 'left_DA_sem']
        self.DA_NRI_block_priorrewards = DA_amp[['block', 'time_in_port_bin', 'num_rewards_prior_bin',
                                                 'right_DA_mean', 'left_DA_mean']]

    def visualize_average_traces(self, variable='time_in_port', method='even_time', block_split=False,
                                 plot_linecharts=0, plot_histograms=0, save=0):
        # Extract interval data
        df_IRI_exp = self.reward_features_DA.copy()

        for branch in ['green_right', 'green_left']:
            if branch not in self.dFF0.columns:
                continue  # Skip if branch is missing

            # Initialize storage arrays
            avr_all = np.full((4, 2), np.nan)
            avr_low = np.full((4, 2), np.nan)
            avr_high = np.full((4, 2), np.nan)

            # Define median interval values based on method
            if method == 'even_time':
                interval_vals = [1, 3, 5, 7] if variable == 'time_in_port' else [0.75, 2.25, 3.75, 5.25]
            elif method == 'overall_quint':
                df_IRI_exp['NRI_quintile'] = pd.qcut(df_IRI_exp[variable], q=4, labels=False)
                interval_vals = df_IRI_exp.groupby('NRI_quintile')[variable].median().to_numpy()
            avr_all[:, 0] = avr_low[:, 0] = avr_high[:, 0] = interval_vals

            if plot_linecharts:
                fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)

            # Iterate over quintiles
            for i, quintile in enumerate(range(4)):
                # Filter trials based on method
                if method == 'even_time':
                    lower_bound, upper_bound = (2 * i, 2 * (i + 1)) if variable == 'time_in_port' else (
                        1.5 * i, 1.5 * (i + 1))
                    df = df_IRI_exp[(df_IRI_exp[variable] > lower_bound) & (df_IRI_exp[variable] < upper_bound)]
                else:  # 'overall_quint'
                    df = df_IRI_exp[df_IRI_exp['NRI_quintile'] == quintile]
                    lower_bound, upper_bound = df[variable].min(), df[variable].max()

                label = f'{np.round(lower_bound, 1)} - {np.round(upper_bound, 1)} sec'
                title = f'rewarded at {label} since {"entry" if variable == "time_in_port" else "last reward"}'

                # Compute average traces
                if df.loc[df['next_reward_time'].notna(), 'reward_time'].to_numpy().shape[0] == 0:
                    print(f"No valid trials for {branch} in group {i + 1}, skipping...")
                    continue
                else:
                    df_avg, df_trial_info = func.construct_matrix_for_average_traces(
                        self.zscore, branch,
                        df.loc[df['next_reward_time'].notna(), 'reward_time'].to_numpy(),
                        df.loc[df['next_reward_time'].notna(), 'next_reward_time'].to_numpy(),
                        df.loc[df['next_reward_time'].notna(), 'trial'].to_numpy(),
                        df.loc[df['next_reward_time'].notna(), 'block'].to_numpy(),
                        interval_name=None
                    )

                    # Compute peak amplitude differences
                    time_mask = df_avg.columns < 0.5
                    peak_diff = df_avg.loc[:, time_mask].mean().max() - df_avg.loc[:, time_mask].mean().min()
                    avr_all[i, 1] = peak_diff

                    # Block-based analysis
                    is_low_block = df_trial_info['phase'] == '0.4'
                    is_high_block = df_trial_info['phase'] == '0.8'
                    avr_low[i, 1] = df_avg[is_low_block].loc[:, time_mask].mean().max() - df_avg.loc[:,
                                                                                          time_mask].mean().min()
                    avr_high[i, 1] = df_avg[is_high_block].loc[:, time_mask].mean().max() - df_avg.loc[:,
                                                                                            time_mask].mean().min()

                    # Plotting (if enabled)
                    if plot_linecharts:
                        ax = axes[i % 2, i // 2]
                        if block_split:
                            for j in range(df_avg[is_high_block].shape[0]):
                                ax.plot(df_avg[is_high_block].iloc[j],
                                        c=sns.light_palette(self.block_palette[1], as_cmap=True)(j))
                            for j in range(df_avg[is_low_block].shape[0]):
                                ax.plot(df_avg[is_low_block].iloc[j],
                                        c=sns.light_palette(self.block_palette[0], as_cmap=True)(j))

                            ax.plot(df_avg[is_low_block].mean(axis=0), c=self.block_palette[0], linewidth=3,
                                    label='0.4 block')
                            ax.plot(df_avg[is_high_block].mean(axis=0), c=self.block_palette[1], linewidth=3,
                                    label='0.8 block')
                        else:
                            for j in range(df_avg.shape[0]):
                                weight = np.round(0.8 - (j + 1) / (df_avg.shape[0] + 1) * 0.4, 2)
                                ax.plot(df_avg.iloc[j], c=str(weight))
                            ax.plot(df_avg.mean(axis=0), c='b', linewidth=2.5)

                        # axes[2, 1].plot(df_avg.mean(axis=0), linewidth=2.5, label=label)  # equal time range
                        # axes[2, 1].set_title('All Average IRI Traces')
                        # axes[2, 1].legend(loc='upper right')
                        ax.set_ylim([-2, 3])
                        ax.set_title(title, fontsize=15)
                        ax.legend()

            # Summary panel in linecharts
            if plot_linecharts:
                # plt.tight_layout()
                # rect = [0.3, 0.59, 0.4, 0.4]
                # ax1 = func.add_subplot_axes(axes[2, 1], rect)
                # ax1.scatter(avr_all[:, 0], avr_all[:, 1], c='k', marker='o', s=15)
                # if block_split:
                #     ax1.scatter(avr_low[:, 0], avr_low[:, 1], c=self.block_palette[0], marker='o', s=20,
                #                 label='0.4 block')
                #     ax1.scatter(avr_high[:, 0], avr_high[:, 1], c=self.block_palette[1], marker='o', s=20,
                #                 label='0.8 block')
                # ax1.legend()
                # ax1.set_xlabel('Median Delay (sec)', fontsize=15)
                # ax1.set_ylabel('Average Amplitude', fontsize=15)
                fig.suptitle(f"{self.animal}:{self.signal_dir[-21:-7]} {branch}", fontsize=20)
                fig.show()

            # Store results in dictionary format
            # data = pd.DataFrame({
            #     'median_interval': avr_all[:, 0],
            #     'amp_all': avr_all[:, 1],
            #     'amp_low': avr_low[:, 1],
            #     'amp_high': avr_high[:, 1]
            # })
            # if variable == 'time_in_port':
            #     self.NRI_amplitude[branch] = data
            # else:
            #     self.IRI_amplitude[branch] = data

        return df_IRI_exp

    def for_pub_compare_traces_by_NRI(self, branch='green_right'):
        title = f"{self.animal}: {self.signal_dir[-23:-7]}\nDA responses are higher for rewards\ndelivered later in port"
        df = self.reward_features_DA.copy()
        interval_vals = [1.5, 4.5, 7.5, 10.5]
        fig, ax = plt.subplots()
        palette = list(reversed(sns.color_palette("Reds", n_colors=len(interval_vals))))
        for i in range(len(interval_vals)):
            lower_bound, upper_bound = (3 * i, 3 * (i + 1))
            df_in_range = df[(df['time_in_port'] > lower_bound) & (df['time_in_port'] < upper_bound)]
            df_avg, df_trial_info = func.construct_matrix_for_average_traces(
                self.zscore, branch,
                df_in_range.loc[df_in_range['next_reward_time'].notna(), 'reward_time'].to_numpy(),
                df_in_range.loc[df_in_range['next_reward_time'].notna(), 'next_reward_time'].to_numpy(),
                df_in_range.loc[df_in_range['next_reward_time'].notna(), 'trial'].to_numpy(),
                df_in_range.loc[df_in_range['next_reward_time'].notna(), 'block'].to_numpy(),
                interval_name=None
            )
            mean = df_avg.mean(axis=0, skipna=True)
            sem = df_avg.sem(axis=0, skipna=True)
            ax.plot(mean.index, mean.values, label=f'{lower_bound}-{upper_bound} sec', color=palette[i])
            ax.fill_between(mean.index, mean - sem, mean + sem, color=palette[i], alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.xlim([0, 3.5])
        plt.xticks([0, 1, 2, 3, 3.5])
        plt.xlabel('Time From Reward Delivery (sec)', fontsize=15)
        plt.ylabel('Dopamine (z-score)', fontsize=15)
        plt.tick_params(axis='both', labelsize=14)
        plt.title(title, fontsize=10)
        plt.legend()
        plt.show()

    def bg_port_in_block_reversal(self):
        df_intervals_bg = func.extract_intervals_bg_inport(self.pi_events)

        col_name_obj = self.dFF0.columns[1:]
        for branch in col_name_obj:
            # Dictionaries to store outputs
            transition_dfs, time_series_dfs, trial_info_dfs, reward_dfs = {}, {}, {}, {}

            # Unique block sequences
            unique_blocks = df_intervals_bg['block_sequence'].unique()

            # Reward columns
            reward_columns = ['reward_1', 'reward_2', 'reward_3', 'reward_4']

            # Process the first four rows of Block 1
            first_four_block1 = df_intervals_bg[df_intervals_bg['block_sequence'] == unique_blocks[0]].iloc[:4]
            time_series_block1, trial_info_block1 = func.construct_matrix_for_average_traces(
                self.zscore, branch,
                first_four_block1['entry'].to_numpy(),
                first_four_block1['exit'].to_numpy(),
                first_four_block1['trial'].to_numpy(),
                first_four_block1['block'].to_numpy()
            )
            reward_block1 = first_four_block1[reward_columns].subtract(first_four_block1['entry'], axis=0)

            # Iterate over block transitions
            for i in range(1, len(unique_blocks)):
                prev_block, current_block = unique_blocks[i - 1], unique_blocks[i]
                transition_df, time_series_df, trial_info_df, reward_df = func.process_block_transition(
                    prev_block, current_block, df_intervals_bg, reward_columns, self.zscore, branch
                )
                key = f"transition_block_{prev_block}_to_{current_block}"
                transition_dfs[key] = transition_df
                time_series_dfs[key] = time_series_df
                trial_info_dfs[key] = trial_info_df
                reward_dfs[key] = reward_df

            # Print the first four rows of Group 1
            print("First four rows of Group 1:")
            print(first_four_block1)

            # Print each transition DataFrame
            for key, transition_df in transition_dfs.items():
                print(f"\n{key}:")
                print(transition_df)

            time_series_list = list(time_series_dfs.values())
            trial_info_list = list(trial_info_dfs.values())
            reward_list = list(reward_dfs.values())

            # Determine the max number of trials (columns) dynamically
            num_cols = max(len(df) for df in time_series_list)
            num_rows = len(transition_dfs)

            # Create figure and axes
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(50, 15), sharex=True, sharey=True)
            plt.subplots_adjust(wspace=0.2, hspace=0.3)
            fig.suptitle(f"{self.animal}:{self.signal_dir[-21:-7]}\n{branch} DA Changes upon Block Switch",
                         fontsize=20, fontweight='bold', multialignment='center')
            fig.supxlabel("Time since Entering Background Port (sec)", fontsize=18, fontweight='bold')
            fig.supylabel("DA (in z-score)", fontsize=18, fontweight='bold')

            xticks = [0, 1.25, 2.5, 3.75, 5, 6]
            delivered_label_added = False  # Ensure single legend entry
            y_min, y_max = float('inf'), float('-inf')
            for i, (df, trial_info_df, reward_df) in enumerate(zip(time_series_list, trial_info_list, reward_list)):
                num_trials = len(df)
                for j in range(num_trials):
                    ax = axes[i, j]
                    trial_num = int(trial_info_df['trial'].iloc[j])
                    ax.set_title(f"trial {trial_num}")
                    color = self.block_palette[0] if trial_info_df.at[j, 'phase'] == '0.4' else self.block_palette[1]
                    ax.plot(df.iloc[j], c=color, linewidth=2.5)  # Plot time series
                    y_min = min(y_min, df.iloc[j].min())
                    y_max = max(y_max, df.iloc[j].max())
                    # Plot delivered rewards (gray dashed)
                    for x_val in reward_df.iloc[j].tolist():
                        ax.axvline(x_val, color='b', linestyle='--', zorder=1,
                                   label='Reward delivered' if not delivered_label_added else "")
                        delivered_label_added = True

                    if (i == 0) & (j == 0):
                        ax.legend()

            yticks = sorted(set([0] + [int(round(y)) for y in np.linspace(y_min, y_max, 4)]))
            for ax in axes.flatten():
                ax.set_xlim(0, 6)
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', labelsize=14)
                ax.set_yticks(yticks)
                ax.grid(axis='x', linestyle='--', alpha=0.5)  # Add grid for better x-axis clarity
                ax.grid(axis='y', linestyle='--', alpha=0.5)

            fig.show()

            selected_dfs = [time_series_list[i] for i in [0, 2, 4]]
            mean_traces = []
            sem_traces = []
            for row_idx in range(6):
                series_list = [df.iloc[row_idx].dropna() for df in selected_dfs]
                aligned_df = pd.DataFrame(series_list)
                mean_trace = aligned_df.mean(axis=0)
                sem_trace = aligned_df.sem(axis=0)
                mean_traces.append(mean_trace)
                sem_traces.append(sem_trace)
            mean_df = pd.DataFrame(mean_traces)
            sem_df = pd.DataFrame(sem_traces)
            valid_columns = ~mean_df.isna().any()
            mean_df = mean_df.loc[:, valid_columns]
            sem_df = sem_df.loc[:, valid_columns]
            fig, ax = plt.subplots()
            low_color = self.block_palette[0]
            high_color = self.block_palette[1]
            pre_map = sns.light_palette(high_color, n_colors=4, reverse=False)
            post_map = sns.light_palette(low_color, n_colors=8, reverse=True)
            colors = pre_map[2:4] + post_map[0:4]
            legends = ['trial -2', 'trial -1', 'trial 0', 'trial 1', 'trial 2', 'trial 3']
            for i in range(mean_df.shape[0]):
                ax.plot(mean_df.iloc[i], color=colors[i], linewidth=1.5, label=legends[i])
                # ax.fill_between(mean_df.columns, mean_df.iloc[i] - sem_df.iloc[i], mean_df.iloc[i] + sem_df.iloc[i],
                #                 alpha=0.2, color=colors[i], linewidth=0, zorder=0)
            ax.set_xlim(1, 4)
            ax.legend()
            ax.set_title(f"{self.animal}:{self.signal_dir[-21:-7]}\n{branch} DA Changes upon Block Switch")
            ax.set_xlabel("Time since Entering Background Port (sec)")
            ax.set_ylabel("DA (in z-score)")
            fig.show()

            print('Finished plotting block reversal traces')

    def visualize_DA_vs_NRI_IRI(self, plot_histograms=0, plot_scatters=0, save=0):
        df_IRI_exp = func.extract_intervals_expreward(self.pi_events, plot_histograms=plot_histograms,
                                                      ani_str=self.animal,
                                                      ses_str=self.signal_dir[-21:-7])
        df = df_IRI_exp

        arr_NRI = df['time_in_port'].to_numpy()
        arr_IRI = df['IRI_prior'].to_numpy()
        arr_block = df['block'].to_numpy()
        df_temp = pd.DataFrame({'NRI': arr_NRI, 'IRI': arr_IRI, 'block': arr_block})
        for branch in ['green_right', 'green_left']:
            if branch not in self.dFF0.columns:
                continue  # Skip if branch is missing
            arr_DA_amp = np.zeros(arr_IRI.shape[0])
            arr_DA_amp.fill(np.nan)
            for row_reward in range(df.shape[0]):
                start = df.loc[row_reward, 'reward_time']
                end = df.loc[row_reward, 'next_reward_time']
                condition_in_range = (self.zscore['time_recording'] >= start) & (self.zscore['time_recording'] < end)
                is_amp_calculation_range = condition_in_range & (self.zscore['time_recording'] < start + 0.5)
                soi_for_amp = self.zscore.loc[is_amp_calculation_range, branch].to_numpy()
                if np.count_nonzero(~np.isnan(soi_for_amp)) > 1:
                    arr_DA_amp[row_reward] = np.max(soi_for_amp) - np.min(soi_for_amp)
            DA_vs_NRI_IRI = pd.DataFrame({'NRI': arr_NRI, 'IRI': arr_IRI, 'DA': arr_DA_amp})
            df_temp[f'DA_{branch[6:]}'] = arr_DA_amp

            if plot_scatters:
                fig, ax = plt.subplots()
                ax.set_title(f"{self.animal}:{self.signal_dir[-21:-7]} {branch}")
                sns.scatterplot(ax=ax, x='NRI', y='IRI', hue='DA', data=DA_vs_NRI_IRI, palette='Spectral_r', s=15,
                                legend=False)
                norm = plt.Normalize(DA_vs_NRI_IRI['DA'].min(), DA_vs_NRI_IRI['DA'].max())
                sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
                sm.set_array([])  # Needed for matplotlib colorbar
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label('DA (in z-score)')

                plt.show()

        # Melt the dataframe
        df_long = df_temp.melt(id_vars=['NRI', 'IRI', 'block'],
                               value_vars=['DA_right', 'DA_left'],
                               var_name='hemisphere',
                               value_name='DA')
        df_long['hemisphere'] = df_long['hemisphere'].str.replace('DA_', '')
        df_long['session'] = self.signal_dir[-21:-7]
        df_long['animal'] = self.animal
        # df_long = df_long[['animal', 'hemisphere', 'session', 'NRI', 'IRI', 'DA']]
        df_long = df_long[['hemisphere', 'NRI', 'IRI', 'block', 'DA']]
        self.DA_vs_NRI_IRI = df_long.dropna().reset_index(drop=True)

    def scatterplot_nonreward_DA_vs_NRI(self, exclude_refractory=True, plot=0):
        df_IRI_exp = func.extract_intervals_expreward(self.pi_events, plot_histograms=0,
                                                      ani_str=self.animal,
                                                      ses_str=self.signal_dir[-21:-7])
        print('hello')

    def extract_binned_da_vs_reward_history_matrix(self, binsize=0.1, save=0):
        condition_exp_entry = (self.pi_events['key'] == 'head') & (self.pi_events['value'] == 1) & (
                self.pi_events['port'] == 1) & (self.pi_events['is_valid'])
        condition_exp_exit = (self.pi_events['key'] == 'head') & (self.pi_events['value'] == 0) & (
                self.pi_events['port'] == 1) & (self.pi_events['is_valid'])
        condition_exp_reward = (self.pi_events['key'] == 'reward') & (self.pi_events['value'] == 1) & (
                self.pi_events['port'] == 1)
        exp_entry = self.pi_events.loc[condition_exp_entry, 'time_recording'].to_numpy()
        exp_exit = self.pi_events.loc[condition_exp_exit, 'time_recording'].to_numpy()
        intervals = list(zip(exp_entry, exp_exit))
        df_signals = self.zscore
        filtered_df_signals = pd.concat(
            [df_signals.loc[(df_signals['time_recording'] >= start) & (df_signals['time_recording'] <= end)] for
             start, end in intervals])
        bin_edges = []
        for start, end in intervals:
            bin_edges.extend(np.arange(start, end, binsize))
        filtered_df_signals['bin'] = pd.cut(filtered_df_signals['time_recording'], bins=bin_edges, right=False)
        binned_means = filtered_df_signals.groupby('bin', observed=False)[
            ['green_right', 'green_left']].mean().reset_index()
        binned_means = binned_means[binned_means['bin'].apply(lambda x: x.right - x.left <= binsize * 2)]
        binned_means = binned_means.dropna().reset_index(drop=True)
        reward_num_in_bin = np.zeros(binned_means.shape[0])
        trial = np.full(binned_means.shape[0], np.nan)
        for i, bin_interval in enumerate(binned_means['bin']):
            rows_bool = (self.pi_events['time_recording'] >= bin_interval.left) & (
                    self.pi_events['time_recording'] < bin_interval.right)
            trial_array = self.pi_events.loc[rows_bool, 'trial'].to_numpy()
            if trial_array.size > 0:
                trial[i] = trial_array[0]
            else:
                trial[i] = trial[i - 1]
            reward_rows = rows_bool & condition_exp_reward
            reward_num_in_bin[i] = reward_rows.sum()
        binned_means['reward_num'] = reward_num_in_bin
        binned_means['trial'] = trial
        binned_means['animal'] = self.animal
        binned_means = func.construct_reward_history_matrix(binned_means, binsize=binsize)
        if save:
            df_to_save = binned_means[['animal', 'trial', 'bin', 'bin_idx', 'reward_num', 'green_right', 'green_left',
                                       'history_matrix_sparse']]
            if self.include_branch == 'only_left':
                df_to_save['green_right'] = np.nan
            elif self.include_branch == 'only_right':
                df_to_save['green_left'] = np.nan
            df_to_save.to_pickle(os.path.join(self.processed_dir, 'binned_DA_reward_history',
                                              f'{self.animal}_{self.signal_dir[-23:-7]}_binned_DA_vs_history.pkl'))


if __name__ == '__main__':
    test_session = OneSession('SZ043', 23, include_branch='both', port_swap=0)
    # test_session.examine_raw(save=0)
    test_session.calculate_dFF0(plot=0, plot_middle_step=0, save=0)
    # test_session.remove_outliers_dFF0()
    test_session.process_behavior_data(save=0)
    test_session.bg_lick_rasterplot()
    # test_session.extract_transient(plot_zscore=0)
    # test_session.visualize_correlation_scatter(save=0)
    test_session.plot_heatmaps(save=1)
    # test_session.plot_bg_heatmaps(save=0)
    # test_session.actual_leave_vs_adjusted_optimal(save=0)
    test_session.extract_reward_features_and_DA(plot=0, save_dataframe=0)
    df_intervals_exp = test_session.visualize_average_traces(variable='time_in_port', method='even_time',
                                                             block_split=False,
                                                             plot_histograms=0, plot_linecharts=1)
    test_session.visualize_DA_vs_NRI_IRI(plot_scatters=1, plot_histograms=1)
    # test_session.bg_port_in_block_reversal()

    # test_session.scatterplot_nonreward_DA_vs_NRI()
    # test_session.for_pub_compare_traces_by_NRI(branch='green_left')
    # test_session.extract_binned_da_vs_reward_history_matrix(binsize=0.1, save=0)

    print("Hello")
