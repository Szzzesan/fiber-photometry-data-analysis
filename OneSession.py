import os
import pickle
import config
import numpy as np
import pandas as pd
import scipy.stats as stats

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import joblib
import helper

mpl.rcParams['figure.dpi'] = 300


class OneSession:
    def __init__(self, animal_str, session, include_branch='both', port_swap=0):
        self.include_branch = include_branch

        self.animal_dir = os.path.normpath(os.path.join(config.MAIN_DATA_ROOT, animal_str))
        raw_dir = os.path.join(self.animal_dir, config.RAW_DATA_SUBDIR)

        FP_file_list = helper.list_files_by_time(raw_dir, file_type='FP', print_names=0)
        behav_file_list = helper.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
        TTL_file_list = helper.list_files_by_time(raw_dir, file_type='arduino', print_names=0)

        self.animal = animal_str
        self.behav_dir = os.path.join(self.animal_dir, 'raw_data', behav_file_list[session])
        self.signal_dir = os.path.join(self.animal_dir, 'raw_data', FP_file_list[session])
        self.arduino_dir = os.path.join(self.animal_dir, 'raw_data', TTL_file_list[session])
        self.fig_export_dir = os.path.join(self.animal_dir, config.FIGURE_SUBDIR, self.signal_dir[-21:-7])
        self.processed_dir = os.path.join(self.animal_dir, config.PROCESSED_DATA_SUBDIR)
        os.makedirs(self.processed_dir, exist_ok=True)
        print("This class is" + " session " + self.signal_dir[-23:-7])
        self.pi_events, self.neural_events = helper.data_read_sync(self.behav_dir, self.signal_dir, self.arduino_dir)
        if self.pi_events['task'].iloc[10] == 'single_reward':
            self.task = 'single_reward'
        elif self.pi_events['task'].iloc[10] == 'cued_no_forgo_forced':
            self.task = 'multi_reward'

        self.trial_df = None
        self.expreward_df = None
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
        self.last_reward_df = pd.DataFrame()
        self.bg_behav_by_trial = pd.DataFrame()
        self.nonreward_DA_vs_time = pd.DataFrame()
        self.nonreward1stmoment_DA_vs_time = pd.DataFrame()

    # --- Data Loading and Initial Processing ---
    def examine_raw(self, save=0):
        helper.check_framedrop(self.neural_events)
        raw_separated = helper.de_interleave(self.neural_events, session_label=self.signal_dir[-23:-7], plot=1,
                                             save=save,
                                             save_path=self.fig_export_dir)

    def calculate_dFF0(self, plot=0, plot_middle_step=0, save=0):
        raw_separated = helper.de_interleave(self.neural_events, session_label=self.signal_dir[-23:-7],
                                             plot=plot_middle_step,
                                             save=0, save_path=self.fig_export_dir)
        self.dFF0 = helper.calculate_dFF0_Hamilos(raw_separated,
                                                  session_label=f'{self.animal}: {self.signal_dir[-23:-7]}',
                                                  plot=plot,
                                                  plot_middle_steps=plot_middle_step, save=save,
                                                  save_path=self.fig_export_dir)
        self.dFF0.name = 'dFF0'
        self.zscore = pd.DataFrame({'time_recording': self.dFF0.time_recording})
        self.zscore['green_right'] = stats.zscore(self.dFF0['green_right'].tolist(), nan_policy='omit')
        self.zscore['green_left'] = stats.zscore(self.dFF0['green_left'].to_list(), nan_policy='omit')
        self.zscore['F0_right_zscore'] = stats.zscore(self.dFF0['F0_right'].tolist(), nan_policy='omit')
        self.zscore['F0_left_zscore'] = stats.zscore(self.dFF0['F0_left'].tolist(), nan_policy='omit')

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
        self.pi_events = helper.data_reduction(self.pi_events, lick_tol=.015, head_tol=0.2)
        self.pi_events = helper.add_2ndry_properties_to_pi_events(self.pi_events)
        # self.pi_events.reset_index(drop=True, inplace=True)
        self.idx_taskbegin = self.dFF0.index[
            self.dFF0['time_recording'] >= self.pi_events['time_recording'].min()].min()
        self.idx_taskend = self.dFF0.index[self.dFF0['time_recording'] <= self.pi_events['time_recording'].max()].max()
        if save:
            self.pi_events.to_csv(f"{self.animal_dir}/{self.signal_dir[-21:-7]}_pi_events.csv")

    def construct_trial_df(self):
        [head, trial, cue, reward, lick, off, on, port1, port2, valid_head] = helper.get_bools(self.pi_events)
        bg_entries = self.pi_events.loc[trial & on & valid_head, 'time_recording'].to_list()
        bg_exits = self.pi_events.loc[port2 & head & off & valid_head, 'time_recording'].to_list()
        exp_entries = self.pi_events.loc[port1 & head & on & valid_head, 'time_recording'].to_list()
        exp_exits = self.pi_events.loc[port1 & head & off & valid_head, 'time_recording'].to_list()
        trials = self.pi_events.loc[port2 & head & off & valid_head, 'trial'].to_list()
        phase = self.pi_events.loc[port2 & head & off & valid_head, 'phase'].to_list()
        rewards = [[] for _ in range(len(trials))]
        licks = [[] for _ in range(len(trials))]
        excess_bg_exits = [[] for _ in range(len(trials))]
        excess_exp_entries = [[] for _ in range(len(trials))]
        excess_exp_exits = [[] for _ in range(len(trials))]
        is_valid_trial = [False] * len(trials)
        for i, trial_id in enumerate(trials):
            is_in_trial = self.pi_events['trial'] == trial_id
            is_valid_trial[i] = self.pi_events.loc[is_in_trial, 'is_valid_trial'].to_list()[0]
            rewards[i] = self.pi_events.loc[reward & on & is_in_trial, 'time_recording'].to_list()
            licks[i] = self.pi_events.loc[lick & on & is_in_trial, 'time_recording'].to_list()
            excess_bg_exits[i] = self.pi_events.loc[
                port2 & head & off & is_in_trial & ~valid_head, 'time_recording'].to_list()
            excess_exp_entries[i] = self.pi_events.loc[
                port1 & head & on & is_in_trial & ~valid_head, 'time_recording'].to_list()
            excess_exp_exits[i] = self.pi_events.loc[
                port1 & head & off & is_in_trial & ~valid_head, 'time_recording'].to_list()
        self.trial_df = pd.DataFrame(
            {'trial': trials, 'phase': phase, 'is_valid_trial': is_valid_trial,
             'rewards': rewards, 'licks': licks,
             'bg_entry': bg_entries, 'bg_exit': bg_exits,
             'exp_entry': exp_entries, 'exp_exit': exp_exits,
             'excess_bg_exits': excess_bg_exits,
             'excess_exp_exits': excess_exp_exits,
             'excess_exp_entries': excess_exp_entries
             })

    def add_trial_info_to_recording(self):
        self.dFF0 = helper.match_recording_time_to_trial_time(self.dFF0, self.trial_df)
        self.zscore = helper.match_recording_time_to_trial_time(self.zscore, self.trial_df)

    def construct_expreward_interval_df(self):
        [head, trial, cue, reward, lick, off, on, port1, port2, valid_head] = helper.get_bools(self.pi_events)
        valid_trial = self.pi_events['is_valid_trial']
        arr_reward_exp = self.pi_events.loc[port1 & reward & on & valid_trial, 'time_recording'].to_numpy()
        arr_reward_bg = self.pi_events.loc[port2 & reward & on & valid_trial, 'time_recording'].to_numpy()
        arr_lastreward_exp = np.insert(arr_reward_exp[:-1], 0, np.nan)
        arr_nextreward_exp = np.append(arr_reward_exp[1:], np.nan)
        arr_NRI = self.pi_events.loc[port1 & reward & on & valid_trial, 'time_in_port'].to_numpy()
        arr_trial_exp = self.pi_events.loc[port1 & reward & on & valid_trial, 'trial']
        arr_block_exp = self.pi_events.loc[port1 & reward & on & valid_trial, 'phase']
        df_intervals = pd.DataFrame(
            {'trial': arr_trial_exp, 'block': arr_block_exp, 'last_reward_time': arr_lastreward_exp,
             'reward_time': arr_reward_exp, 'next_reward_time': arr_nextreward_exp, 'time_in_port': arr_NRI})
        first_rewards_idx = df_intervals.groupby('trial').head(1).index
        last_rewards_idx = df_intervals.groupby('trial').tail(1).index
        df_intervals.loc[first_rewards_idx, 'last_reward_time'] = np.nan
        df_intervals.loc[last_rewards_idx, 'next_reward_time'] = np.nan
        first_rewards_times = df_intervals.loc[first_rewards_idx, 'reward_time'].to_numpy()
        last_bg_rewards = np.searchsorted(arr_reward_bg, first_rewards_times, side='right') - 1
        valid_bg_mask = (last_bg_rewards > 0)
        df_intervals.loc[first_rewards_idx[valid_bg_mask], 'last_reward_time'] = arr_reward_bg[
            last_bg_rewards[valid_bg_mask]]
        df_intervals['IRI_prior'] = df_intervals['reward_time'] - df_intervals['last_reward_time']
        df_intervals['IRI_post'] = df_intervals['next_reward_time'] - df_intervals['reward_time']
        df_intervals.reset_index(inplace=True, drop=True)
        arr_recent_reward_rate = np.zeros(df_intervals.shape[0])
        arr_recent_reward_rate[:] = np.nan
        arr_recent_reward_rate_exp = np.zeros(df_intervals.shape[0])
        arr_recent_reward_rate_exp[:] = np.nan
        arr_local_reward_rate_1sec = np.zeros(df_intervals.shape[0])
        arr_local_reward_rate_1sec[:] = np.nan

        arr_exp_exits = np.zeros(df_intervals.shape[0])
        arr_exp_exits[:] = np.nan
        arr_exp_entries = np.full(df_intervals.shape[0], np.nan)

        for i in range(df_intervals.shape[0]):
            search_begin = max(0, df_intervals.loc[i, 'reward_time'] - 30)
            search_end = df_intervals.loc[i, 'reward_time']
            is_in_range = (self.pi_events['time_recording'] >= search_begin) & (
                    self.pi_events['time_recording'] < search_end)
            onesec_begin = max(0, df_intervals.loc[i, 'reward_time'] - 1)
            onesec_end = df_intervals.loc[i, 'reward_time']
            is_in_1sec = (self.pi_events['time_recording'] >= onesec_begin) & (
                    self.pi_events['time_recording'] < onesec_end)
            arr_recent_reward_rate[i] = self.pi_events.loc[is_in_range & reward & on].shape[0] / (
                    search_end - search_begin)
            arr_recent_reward_rate_exp[i] = self.pi_events.loc[is_in_range & reward & on & port1].shape[0] / (
                    search_end - search_begin)
            arr_local_reward_rate_1sec[i] = self.pi_events.loc[is_in_1sec & reward & on].shape[0]

            # find the exponential port entries and exits for each trial
            def extract_single_time(time_values, trial, event_type):
                if time_values.shape[0] == 1:
                    return time_values[0]
                elif time_values.shape[0] > 1:
                    raise ValueError(
                        f"Found {time_values.shape[0]} valid time investment port {event_type}s for trial {trial}!")

            is_trial = (self.pi_events['trial'] == df_intervals.loc[i, 'trial'])

            time_entry = self.pi_events.loc[is_trial & port1 & head & on & valid_head, 'time_recording'].to_numpy()
            time_exit = self.pi_events.loc[is_trial & port1 & head & off & valid_head, 'time_recording'].to_numpy()
            arr_exp_entries[i] = extract_single_time(time_entry, df_intervals.loc[i, 'trial'], "entry")
            arr_exp_exits[i] = extract_single_time(time_exit, df_intervals.loc[i, 'trial'], "exit")

        df_intervals['entry_time'] = arr_exp_entries
        df_intervals['exit_time'] = arr_exp_exits
        df_intervals['recent_reward_rate'] = arr_recent_reward_rate
        df_intervals['recent_reward_rate_exp'] = arr_recent_reward_rate_exp
        df_intervals['local_reward_rate_1sec'] = arr_local_reward_rate_1sec
        df_intervals['num_rewards_prior'] = df_intervals.groupby('trial')['reward_time'].cumcount()
        self.expreward_df = df_intervals

    # --- Exploratory Analysis and Visualization ---
    def actual_leave_vs_adjusted_optimal(self, save=0):
        # this function would depend on the execution of self.process_behavior_data()
        self.intervals_df = helper.make_intervals_df(self.pi_events)
        helper.visualize_adjusted_optimal(self.intervals_df, save=save, save_path=self.fig_export_dir)

    def extract_bg_behav_by_trial(self):
        is_trial_start = (self.pi_events['key'] == 'trial') & (self.pi_events['value'] == 1)
        is_lick = (self.pi_events['key'] == 'lick') & (self.pi_events['value'] == 1)
        is_reward = (self.pi_events['key'] == 'reward') & (self.pi_events['value'] == 1)
        is_entry = (self.pi_events['key'] == 'head') & (self.pi_events['value'] == 1)
        is_exit = (self.pi_events['key'] == 'head') & (self.pi_events['value'] == 0)
        is_valid = (self.pi_events['is_valid'] == 1)
        is_context = self.pi_events['port'] == 2
        bg_entries_df = self.pi_events[is_trial_start & self.pi_events['is_valid']]
        bg_entry_times = bg_entries_df['time_recording'].values
        bg_entry_trial_ids = bg_entries_df['trial'].values
        bg_exits_df = self.pi_events[is_exit & is_valid & is_context]
        bg_exit_times = bg_exits_df['time_recording'].values
        bg_exit_trial_ids = bg_exits_df['trial'].values

        bg_licks_df = self.pi_events[is_lick & is_context]
        bg_lick_times = bg_licks_df['time_recording'].values
        bg_rewards_df = self.pi_events[is_reward & is_context]
        bg_reward_times = bg_rewards_df['time_recording'].values
        bg_excessive_entries_df = self.pi_events[is_entry]
        bg_excessive_entry_times = bg_excessive_entries_df['time_recording'].values
        bg_excessive_exits_df = self.pi_events[is_exit & (~is_valid)]
        bg_excessive_exit_times = bg_excessive_exits_df['time_recording'].values
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
        # --- Decision point based on validation ---
        if not are_trials_aligned_and_match:
            print("CRITICAL WARNING: Entry and exit trial data are NOT aligned as expected. "
                  "Using 'bg_exit_times[i]' to correspond to 'bg_entry_times[i]' "
                  "based on index will likely lead to incorrect trial pairings.")
        else:
            print("Proceeding with analysis assuming aligned entry and exit data.")
            licks_by_trial = []
            rewards_by_trial = []
            excess_entries_by_trial = []
            excess_exits_by_trial = []
            if len(bg_entry_times) == 0:
                print("No port 2 entry events found. Cannot create raster plot.")
            else:
                for trial_idx, entry_time in enumerate(bg_entry_times):
                    trial_licks_abs = bg_lick_times[
                        (bg_lick_times > entry_time) & (bg_lick_times < bg_exit_times[trial_idx])]
                    trial_rewards_abs = bg_reward_times[
                        (bg_reward_times > entry_time) & (bg_reward_times < bg_exit_times[trial_idx])]
                    trial_excess_entries_abs = bg_excessive_entry_times[
                        (abs(bg_excessive_entry_times - entry_time) > 0.01) & (
                                bg_excessive_entry_times > entry_time) & (
                                bg_excessive_entry_times < bg_exit_times[trial_idx])]
                    trial_excess_exits_abs = bg_excessive_exit_times[
                        (bg_excessive_exit_times > entry_time) & (bg_excessive_exit_times < bg_exit_times[trial_idx])]
                    licks_by_trial.append(list(trial_licks_abs))
                    rewards_by_trial.append(list(trial_rewards_abs))
                    excess_entries_by_trial.append(list(trial_excess_entries_abs))
                    excess_exits_by_trial.append(list(trial_excess_exits_abs))
            self.bg_behav_by_trial['trial'] = bg_exits_df['trial']
            self.bg_behav_by_trial['phase'] = bg_exits_df['phase']
            self.bg_behav_by_trial['entry'] = bg_entry_times
            self.bg_behav_by_trial['exit'] = bg_exit_times
            self.bg_behav_by_trial['licks'] = licks_by_trial
            self.bg_behav_by_trial['rewards'] = rewards_by_trial
            self.bg_behav_by_trial['excess_entries'] = excess_entries_by_trial
            self.bg_behav_by_trial['excess_exits'] = excess_exits_by_trial
            print('Dataframe bg_behav_by_trial have been prepared. ')

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
                trial_rewards_abs = bg_reward_times[
                    (bg_reward_times > filter_left) & (bg_reward_times < bg_exit_times[trial_idx])]
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

    def plot_reward_aligned_lick_histograms(self,
                                            phases_to_analyze=['0.4', '0.8'],
                                            reward_indices_to_align=[0, 1, 2, 3],
                                            window_r123=(-0.9, 0.3),
                                            window_r4=(-1.0, 0.2),
                                            target_bin_width=0.1):

        if not hasattr(self, 'bg_behav_by_trial') or self.bg_behav_by_trial.empty:
            print("DataFrame 'bg_behav_by_trial' is not available or is empty.")
            return
        if not hasattr(self, 'block_palette'):
            print("Warning: self.block_palette not found. Using default colors.")
            cmap = plt.cm.get_cmap('viridis', len(phases_to_analyze) if len(phases_to_analyze) > 0 else 1)
            self.block_palette = [cmap(i) for i in range(len(phases_to_analyze))]

        num_phases = len(phases_to_analyze)
        num_reward_alignments = len(reward_indices_to_align)

        if num_phases == 0 or num_reward_alignments == 0:
            print("No phases or reward alignment points specified.")
            return

        fig, axes = plt.subplots(num_phases, num_reward_alignments,
                                 figsize=(2 * num_reward_alignments, 3 * num_phases),
                                 sharex=False, sharey=True, squeeze=False)

        def get_reward_label(idx):
            if idx == 0: return "1st"
            if idx == 1: return "2nd"
            if idx == 2: return "3rd"
            return f"{idx + 1}th"

        for i, current_phase in enumerate(phases_to_analyze):
            phase_df = self.bg_behav_by_trial[self.bg_behav_by_trial['phase'] == current_phase]
            bar_color_for_phase = self.block_palette[i % len(self.block_palette)]

            if phase_df.empty:
                for k_col, reward_idx_col_loop in enumerate(reward_indices_to_align):
                    ax = axes[i, k_col]
                    temp_t_before, temp_t_after = window_r123 if reward_idx_col_loop < 3 else window_r4
                    reward_label_str_col = get_reward_label(reward_idx_col_loop)
                    ax.text(0.5, 0.5, f"P{current_phase}|{reward_label_str_col}\nNo Data", fontsize=7)
                    ax.axvline(0, color='gray', linestyle='--', linewidth=1, label=f'{reward_label_str_col} Reward')
                    ax.set_xlim(temp_t_before, temp_t_after)
                    ax.legend(fontsize='x-small')
                continue

            for k, reward_idx_to_align in enumerate(reward_indices_to_align):
                ax = axes[i, k]
                current_reward_label = get_reward_label(reward_idx_to_align)

                current_t_before, current_t_after = window_r123
                if reward_idx_to_align == 3:  # 4th reward (0-indexed)
                    current_t_before, current_t_after = window_r4

                current_hist_range = (current_t_before, current_t_after)  # Still useful for np.histogram if needed

                # --- Calculate bin edges based on target_bin_width ---
                current_range_width = current_t_after - current_t_before
                if current_range_width <= 0 or target_bin_width <= 0:  # Safety checks
                    num_actual_bins = 1  # Fallback to a single bin
                    current_bin_edges = np.array([current_t_before, current_t_after])
                else:
                    # Calculate the number of bins needed to cover the range with the target width
                    num_actual_bins = int(np.ceil(current_range_width / target_bin_width))
                    # Use np.linspace to create precise bin edges. num_actual_bins + 1 edges for num_actual_bins bins.
                    current_bin_edges = np.linspace(current_t_before, current_t_after, num_actual_bins + 1)

                # --- End of bin edges calculation ---

                ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label=f'{current_reward_label} Reward')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_xlim(current_t_before, current_t_after)

                all_relative_licks_for_subplot = []
                trials_contributing_after_filter = 0

                # --- Data preparation and filtering ---
                for _, trial_data in phase_df.iterrows():
                    reward_times_in_trial = trial_data['rewards']
                    lick_times_in_trial = trial_data['licks']
                    if not (isinstance(reward_times_in_trial, list) and len(
                            reward_times_in_trial) > reward_idx_to_align):
                        continue
                    alignment_reward_time_abs = reward_times_in_trial[reward_idx_to_align]
                    data_collection_window_start = alignment_reward_time_abs + current_t_before
                    data_collection_window_end = alignment_reward_time_abs + current_t_after
                    stayed_in_port_continuously = True  # Assuming this logic is complete from previous steps
                    primary_trial_entry = trial_data['entry']
                    primary_trial_exit = trial_data['exit']
                    if not (
                            primary_trial_entry <= data_collection_window_start and primary_trial_exit >= data_collection_window_end):
                        stayed_in_port_continuously = False
                    if stayed_in_port_continuously:
                        trial_excess_exits = trial_data.get('excess_exits', [])
                        if isinstance(trial_excess_exits, list):
                            for t_excess_exit in trial_excess_exits:
                                if data_collection_window_start < t_excess_exit < data_collection_window_end:
                                    stayed_in_port_continuously = False
                                    break
                    if not stayed_in_port_continuously:
                        continue
                    trials_contributing_after_filter += 1
                    if isinstance(lick_times_in_trial, list) and len(lick_times_in_trial) > 0:
                        for lick_time_abs in lick_times_in_trial:
                            relative_lick_time = lick_time_abs - alignment_reward_time_abs
                            # Licks must be within the precise histogram range for binning
                            if current_bin_edges[0] <= relative_lick_time <= current_bin_edges[-1]:
                                all_relative_licks_for_subplot.append(relative_lick_time)
                # --- End of data preparation ---
                text_main = f"Phase {current_phase} | {current_reward_label} Reward"
                if trials_contributing_after_filter == 0:
                    print(f"Phase {current_phase}: No trials with at least {current_reward_label} reward.")
                    ax.text(0.5, 0.5, f"{text_main}\nNo such rewards in trials", fontsize=9)
                elif not all_relative_licks_for_subplot:
                    print(
                        f"Phase {current_phase}, {current_reward_label} Reward: No licks in window ({trials_contributing_after_filter} trials had this reward).")
                    ax.text(0.5, 0.5, f"{text_main}\n({trials_contributing_after_filter} trials)\nNo licks in window",
                            fontsize=9)
                else:
                    raw_counts, _ = np.histogram(all_relative_licks_for_subplot, bins=current_bin_edges)
                    weights = np.full(len(all_relative_licks_for_subplot),
                                      (1 / trials_contributing_after_filter) / target_bin_width)  # Percentage

                    ax.hist(all_relative_licks_for_subplot, bins=current_bin_edges,  # Use edges
                            weights=weights, color=bar_color_for_phase,
                            edgecolor='grey', alpha=0.75)

        for reward_idx in reward_indices_to_align:
            reward_label_str = get_reward_label(reward_idx)
            axes[0, reward_idx].set_title(f"{reward_label_str} Reward")
        fig.supxlabel("Time from Reward (sec)")
        fig.supylabel('Lick Freq. (licks/sec/trial)')
        fig.suptitle(f"{self.animal}: {self.signal_dir[-21:-7]} ")
        fig.tight_layout()
        fig.show()

    def calculate_lick_rates_around_bg_reward(self, reward_idx_to_align=3, plot_comparison=0):
        """
        Calculates average lick frequencies (licks/sec/trial) for specific windows
        around the 2nd reward, for phases '0.4' and '0.8'.
        Filters trials based on continuous port stay during a defined overall window.
        """
        if not hasattr(self, 'bg_behav_by_trial') or self.bg_behav_by_trial.empty:
            print("DataFrame 'bg_behav_by_trial' is not available or is empty.")
            return None

        calculated_rates = {}
        reward_label_str_list = ['1st Reward', '2nd Reward', '3rd Reward', '4th Reward']
        reward_label = reward_label_str_list[reward_idx_to_align]

        # Define phase-specific configurations
        # Windows are [start, end) relative to the 2nd reward time
        phase_configs = {
            '0.4': {
                'overall_port_stay_window': (-1.4, 0.3),
                'analysis_windows': {
                    'baseline': (-1.4, -1.1),
                    'anticipation': (-0.3, 0.0),
                    'consumption': (0.0, 0.3)
                }
            },
            '0.8': {
                'overall_port_stay_window': (-0.8, 0.3),
                'analysis_windows': {
                    'baseline': (-0.8, -0.5),
                    'anticipation': (-0.3, 0.0),
                    'consumption': (0.0, 0.3)
                }
            }
        }

        for phase, config in phase_configs.items():
            phase_df = self.bg_behav_by_trial[
                self.bg_behav_by_trial['phase'] == str(phase)]  # Ensure phase is string for lookup

            if phase_df.empty:
                print(f"No data found for phase '{phase}'.")
                calculated_rates[phase] = {f"{name}_freq_hz": 0.0 for name in config['analysis_windows']}
                calculated_rates[phase]['num_valid_trials'] = 0
                continue

            valid_trials_for_phase_count = 0
            # Initialize total licks for each analysis window for this phase
            total_licks_in_analysis_windows = {key: 0 for key in config['analysis_windows']}

            for _, trial_data in phase_df.iterrows():
                reward_times_in_trial = trial_data['rewards']

                # Check 1: Does the trial have at least the 2nd reward?
                if not (isinstance(reward_times_in_trial, list) and len(reward_times_in_trial) > reward_idx_to_align):
                    continue

                alignment_reward_time_abs = reward_times_in_trial[reward_idx_to_align]

                # Define the overall window for port stay check, in absolute time
                port_stay_win_start_rel, port_stay_win_end_rel = config['overall_port_stay_window']
                abs_port_stay_window_start = alignment_reward_time_abs + port_stay_win_start_rel
                abs_port_stay_window_end = alignment_reward_time_abs + port_stay_win_end_rel

                # Check 2: Continuous Port Occupancy Filter
                stayed_in_port_continuously = True
                primary_trial_entry = trial_data['entry']
                primary_trial_exit = trial_data['exit']

                if not (primary_trial_entry <= abs_port_stay_window_start and \
                        primary_trial_exit >= abs_port_stay_window_end):
                    stayed_in_port_continuously = False

                if stayed_in_port_continuously:
                    trial_excess_exits = trial_data.get('excess_exits', [])  # Use .get for safety
                    if isinstance(trial_excess_exits, list):
                        for t_excess_exit in trial_excess_exits:
                            # If an excess exit occurs strictly WITHIN the port stay window boundaries
                            if abs_port_stay_window_start < t_excess_exit < abs_port_stay_window_end:
                                stayed_in_port_continuously = False
                                break

                if not stayed_in_port_continuously:
                    continue  # Skip this trial due to port exit during the overall validity window

                # If we reach here, trial is valid (has 2nd reward & passed port stay filter)
                valid_trials_for_phase_count += 1

                lick_times_in_trial = trial_data.get('licks', [])  # Use .get for safety
                if isinstance(lick_times_in_trial, list) and len(lick_times_in_trial) > 0:
                    for lick_time_abs in lick_times_in_trial:
                        relative_lick_time = lick_time_abs - alignment_reward_time_abs

                        for window_name, (win_start_rel, win_end_rel) in config['analysis_windows'].items():
                            # Check if lick falls into the analysis window [start, end)
                            if win_start_rel <= relative_lick_time < win_end_rel:
                                total_licks_in_analysis_windows[window_name] += 1

            # Calculate average frequencies for the current phase
            phase_results_dict = {}
            if valid_trials_for_phase_count > 0:
                for window_name, total_licks in total_licks_in_analysis_windows.items():
                    win_start_rel, win_end_rel = config['analysis_windows'][window_name]
                    window_duration_sec = win_end_rel - win_start_rel

                    if window_duration_sec > 0:  # Should always be true with defined windows
                        avg_freq_hz = total_licks / (valid_trials_for_phase_count * window_duration_sec)
                    else:
                        avg_freq_hz = 0.0
                    phase_results_dict[f"{window_name}_freq_hz"] = avg_freq_hz
            else:  # No valid trials found for this phase
                for window_name in config['analysis_windows'].keys():
                    phase_results_dict[f"{window_name}_freq_hz"] = 0.0

            phase_results_dict['num_valid_trials'] = valid_trials_for_phase_count
            calculated_rates[phase] = phase_results_dict

        if plot_comparison:
            # Define the order of periods for the x-axis
            period_labels = ['baseline\n(middle 0.3 sec)', 'anticipation\n((-0.3, 0) sec)',
                             'consumption\n((0, 0.3) sec)']
            periods = ['baseline', 'anticipation', 'consumption']
            x_coords = np.arange(len(periods))  # Numerical positions for x-axis

            fig, ax = plt.subplots(figsize=(4, 6))

            # Determine which phases are in the data, or use a fixed list
            phases_to_plot_from_data = sorted(calculated_rates.keys())

            # Use a consistent order and only plot requested phases if self.block_palette is indexed
            # For this example, we'll use the phases available in calculated_rates and map to palette
            palette_indices = {'0.4': 0, '0.8': 1}  # Assuming these are the main phases of interest

            for phase_key in phases_to_plot_from_data:
                if phase_key not in calculated_rates or phase_key not in palette_indices:
                    print(f"Skipping phase {phase_key}: not in data or no defined palette index.")
                    continue

                phase_data = calculated_rates[phase_key]
                rates_for_phase = [
                    phase_data.get(f"{p}_freq_hz", 0.0) for p in periods  # Use .get for safety
                ]
                num_trials = phase_data.get('num_valid_trials', 0)

                # Get color from palette using predefined mapping
                color_idx = palette_indices[phase_key]
                color = self.block_palette[color_idx % len(self.block_palette)]

                ax.plot(x_coords, rates_for_phase, marker='o', markersize=8, linestyle='-', linewidth=2,
                        color=color, label=f"Phase {phase_key} (n={num_trials} trials)")

            ax.set_xticks(x_coords)
            ax.set_xticklabels(period_labels, fontsize=11)
            ax.set_ylabel("Avg. Lick Rate (licks/sec/trial)", fontsize=12)
            ax.set_title(f"{self.animal}: {self.signal_dir[-23:-7]}\nLick Rate Comparison around {reward_label}",
                         fontsize=14, pad=20)

            ax.legend(fontsize=10, title="Phase Information")
            ax.grid(True, linestyle=':', alpha=0.7, axis='y')  # Light grid on y-axis

            # Improve y-axis limits for better visualization
            current_ylim_bottom, current_ylim_top = ax.get_ylim()
            if current_ylim_top <= 1.0 and current_ylim_top > 0:  # If max Y is small
                ax.set_ylim(0, max(1.0, current_ylim_top + 0.2))
            elif current_ylim_top == 0:  # If all rates are zero
                ax.set_ylim(0, 1.0)
            else:  # Add some padding
                ax.set_ylim(max(0, current_ylim_bottom - 0.1),
                            current_ylim_top + (current_ylim_top - current_ylim_bottom) * 0.1)

            plt.tight_layout()
            plt.show()

        epsilon = 1e-9
        anticipatory_lick_modulation_low = (
                                                   calculated_rates['0.4']['anticipation_freq_hz'] -
                                                   calculated_rates['0.4']['baseline_freq_hz']) / (
                                                   calculated_rates['0.4']['anticipation_freq_hz'] +
                                                   calculated_rates['0.4']['baseline_freq_hz'] + epsilon)
        anticipatory_lick_modulation_high = (
                                                    calculated_rates['0.8']['anticipation_freq_hz'] -
                                                    calculated_rates['0.8']['baseline_freq_hz']) / (
                                                    calculated_rates['0.8']['anticipation_freq_hz'] +
                                                    calculated_rates['0.8']['baseline_freq_hz'] + epsilon)
        return calculated_rates, (anticipatory_lick_modulation_low, anticipatory_lick_modulation_high)

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
            df_for_heatmap_bg, df_bg_r_trial_info, dfplot_entry, dfplot_exit, dfplot_reward, dfplot_lick = helper.construct_matrix_for_heatmap(
                self.pi_events, self.dFF0,
                branch=branch[0],
                vmin=-1, vmax=12,
                time0_condition=condition_bg_entry,
                filterleft_condition=condition_exp_exit,
                filterright_condition=condition_bg_exit,
                orderleft_condition=None,
                orderright_condition=None,
                time0_name=f'BgEntry', order_name='RealTime')
            helper.plot_heatmap_from_matrix(df_for_heatmap_bg, df_bg_r_trial_info, dfplot_entry, dfplot_exit,
                                            dfplot_reward, dfplot_lick, cbarmin=branch[1], cbarmax=branch[2] * 0.37,
                                            plot_lick=0, split_block=1, save=save, save_path=self.fig_export_dir)

    def plot_heatmaps(self, save=0):
        cbarmin_l = int(np.nanpercentile(self.dFF0['green_left'].values, 0.1) * 100)
        cbarmin_r = int(np.nanpercentile(self.dFF0['green_right'].values, 0.1) * 100)
        cbarmax_l = np.nanpercentile(self.dFF0['green_left'].iloc[self.idx_taskbegin:self.idx_taskend], 100) * 100
        cbarmax_r = np.nanpercentile(self.dFF0['green_right'].iloc[self.idx_taskbegin:self.idx_taskend], 100) * 100
        if self.include_branch == 'both':
            helper.plot_heatmap(self.pi_events, self.dFF0, 'green_left', cbarmin=cbarmin_l, cbarmax=cbarmax_l,
                                save=save,
                                save_path=self.fig_export_dir)
            helper.plot_heatmap(self.pi_events, self.dFF0, 'green_right', cbarmin=cbarmin_r, cbarmax=cbarmax_r,
                                save=save,
                                save_path=self.fig_export_dir)
        elif self.include_branch == 'only_right':
            helper.plot_heatmap(self.pi_events, self.dFF0, 'green_right', cbarmin=cbarmin_r, cbarmax=cbarmax_r,
                                save=save,
                                save_path=self.fig_export_dir)
        elif self.include_branch == 'only_left':
            helper.plot_heatmap(self.pi_events, self.dFF0, 'green_left', cbarmin=cbarmin_l, cbarmax=cbarmax_l,
                                save=save,
                                save_path=self.fig_export_dir)
        elif self.include_branch == 'neither':
            print("No branch is available.")
        else:
            print("Error: The 'include_branch' argument can only be 'both', 'only_right', 'only_left', or 'neither'.")

    def extract_transient(self, plot_zscore=0, plot_dff0=0):
        if self.include_branch == 'both':
            self.transient_r = helper.extract_transient_info('green_right', self.dFF0, self.pi_events,
                                                             plot_zscore=plot_zscore,
                                                             plot=plot_dff0)
            self.transient_l = helper.extract_transient_info('green_left', self.dFF0, self.pi_events,
                                                             plot_zscore=plot_zscore,
                                                             plot=plot_dff0)
        elif self.include_branch == 'only_right':
            self.transient_r = helper.extract_transient_info('green_right', self.dFF0, self.pi_events,
                                                             plot_zscore=plot_zscore,
                                                             plot=plot_dff0)
        elif self.include_branch == 'only_left':
            self.transient_l = helper.extract_transient_info('green_left', self.dFF0, self.pi_events,
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

    def visualize_correlation_scatter(self, plot=0, save=0):
        # this function depends on the execution of function self.extract_transient()
        if len(self.transient_l.index > 10):
            self.corr_l = helper.visualize_trial_by_trial(self.transient_l, self.dFF0, 'green_left',
                                                          session_label=self.signal_dir[-23:-7],
                                                          plot=1, save=save, save_path=self.fig_export_dir,
                                                          left_or_right='left',
                                                          task=self.task)
        if len(self.transient_r.index > 10):
            self.corr_r = helper.visualize_trial_by_trial(self.transient_r, self.dFF0, 'green_right',
                                                          session_label=self.signal_dir[-23:-7],
                                                          plot=1, save=save, save_path=self.fig_export_dir,
                                                          left_or_right='right',
                                                          task=self.task)

    def extract_reward_features_and_DA(self, plot=0, save_dataframe=0):
        df_iri_exp = self.expreward_df
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
                    df_avg, df_trial_info = helper.construct_matrix_for_average_traces(
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
                # ax1 = helper.add_subplot_axes(axes[2, 1], rect)
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
            df_avg, df_trial_info = helper.construct_matrix_for_average_traces(
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

    def bg_port_in_block_reversal(self, plot_single_traes=0, plot_average=0):
        df_intervals_bg = helper.extract_intervals_bg_inport(self.pi_events)

        col_name_obj = self.dFF0.columns[1:]

        # This dictionary will store all structured traces for the current object, organized by branch
        DA_in_block_transition = {}

        for branch in col_name_obj:
            # Dictionaries to store outputs for current branch (for existing plots and processing)
            transition_dfs, time_series_dfs, trial_info_dfs, reward_dfs = {}, {}, {}, {}

            # This dictionary will store structured traces for the current branch
            current_branch_structured_traces = {}

            # Unique block sequences
            unique_blocks = df_intervals_bg['block_sequence'].unique()

            # Reward columns
            reward_columns = ['reward_1', 'reward_2', 'reward_3', 'reward_4']

            # Process the first four rows of Block 1 (existing code)
            if len(unique_blocks) > 0:
                first_four_block1 = df_intervals_bg[df_intervals_bg['block_sequence'] == unique_blocks[0]].iloc[:4]
                if not first_four_block1.empty:
                    time_series_block1, trial_info_block1 = helper.construct_matrix_for_average_traces(
                        self.zscore, branch,
                        first_four_block1['entry'].to_numpy(),
                        first_four_block1['exit'].to_numpy(),
                        first_four_block1['trial'].to_numpy(),
                        first_four_block1['block'].to_numpy()
                    )
                    reward_block1 = first_four_block1[reward_columns].subtract(first_four_block1['entry'], axis=0)

                    # Optional: Store initial block data if needed for averaging
                    # You can define a specific key and structure for this if required
                    # For example:
                    # initial_block_key = f"initial_block_{str(unique_blocks[0])}_first_{len(time_series_block1)}_trials"
                    # trial_data = {k_idx: time_series_block1.iloc[k_idx].copy() for k_idx in range(len(time_series_block1))}
                    # current_branch_structured_traces[initial_block_key] = trial_data
                else:
                    print(
                        f"Warning for object {self.animal}, branch {branch}: No data for the first block {unique_blocks[0]}. Skipping first_four_block1 processing.")
            else:
                print(
                    f"Warning for object {self.animal}, branch {branch}: No unique blocks found. Skipping processing for this branch.")
                DA_in_block_transition[branch] = current_branch_structured_traces  # Store empty if no blocks
                continue  # Skip to the next branch

            # Iterate over block transitions
            for i in range(1, len(unique_blocks)):
                prev_block, current_block = unique_blocks[i - 1], unique_blocks[i]

                # It's assumed helper.process_block_transition returns time_series_df
                # where rows are the trials: -2, -1, 0 (switch), +1, +2, +3 relative to switch
                _transition_df, _time_series_df, _trial_info_df, _reward_df = helper.process_block_transition(
                    prev_block, current_block, df_intervals_bg, reward_columns, self.zscore, branch
                )

                # Store for existing plotting logic
                key = f"transition_block_{prev_block}_to_{current_block}"
                transition_dfs[key] = _transition_df
                time_series_dfs[key] = _time_series_df
                trial_info_dfs[key] = _trial_info_df
                reward_dfs[key] = _reward_df

                # --- New logic for structuring data for saving ---
                # Define the relative trial numbers corresponding to the rows of _time_series_df
                # Based on your plotting legends: ['trial -2', 'trial -1', 'trial 0', 'trial 1', 'trial 2', 'trial 3']
                relative_trial_labels = [-2, -1, 0, 1, 2, 3]

                structured_transition_key = f"from_{str(prev_block)}_to_{str(current_block)}"
                current_branch_structured_traces[structured_transition_key] = {}

                if len(_time_series_df) == len(relative_trial_labels):
                    for trial_idx, rel_label in enumerate(relative_trial_labels):
                        # _time_series_df.iloc[trial_idx] is the pandas Series for this trial's trace
                        current_branch_structured_traces[structured_transition_key][rel_label] = _time_series_df.iloc[
                            trial_idx].copy()
                else:
                    print(
                        f"Warning for object {self.animal}, branch {branch}, transition {structured_transition_key}: Expected {len(relative_trial_labels)} trials, but found {len(_time_series_df)}. Data for this transition will be incomplete in the structured output.")
                    # Store what's available, labeling by relative trial index if possible,
                    # or you might decide to skip this transition for this branch's structured data.
                    # For now, it will result in a partially filled or empty dict for this transition's trials.
                    # You could fill with NaNs or handle as per your error strategy. For example, to store available ones:
                    # for trial_idx in range(len(_time_series_df)):
                    #    if trial_idx < len(relative_trial_labels): # Ensure we don't go out of bounds for labels
                    #        rel_label = relative_trial_labels[trial_idx] # This might misalign if not careful, assuming first N trials
                    #        current_branch_structured_traces[structured_transition_key][rel_label] = _time_series_df.iloc[trial_idx].copy()

            # Store the structured traces for this branch into the main dictionary for the object
            DA_in_block_transition[branch] = current_branch_structured_traces

            if plot_single_traes:
                # --- Plotting Code ---
                if not time_series_dfs:  # Check if there are any transitions to plot
                    print(f"No transitions to plot for branch {branch} in object {self.animal}")
                else:
                    # Print the first four rows of Group 1 (Block 1)
                    if 'first_four_block1' in locals() and not first_four_block1.empty:  # ensure first_four_block1 exists and is not empty
                        print("First four rows of Group 1:")
                        print(first_four_block1)
                    else:
                        print("No data for 'first_four_block1' or it was empty.")

                    # Print each transition DataFrame
                    for key, df_to_print in transition_dfs.items():
                        print(f"\n{key}:")
                        print(df_to_print)

                    time_series_list = list(time_series_dfs.values())
                    trial_info_list = list(trial_info_dfs.values())
                    reward_list = list(reward_dfs.values())

                    if not time_series_list:  # Check if list is empty before proceeding
                        print(f"No time series data to plot for branch {branch}.")
                    else:
                        # Determine the max number of trials (columns) dynamically
                        # num_cols = max(len(df) for df in time_series_list) # This was original
                        # If time_series_df always has 6 rows (trials) as assumed for structured data:
                        num_cols_plot = 6  # Max trials to plot per transition, aligns with relative_trial_labels
                        if any(len(df) != num_cols_plot for df in time_series_list if df is not None):
                            print(
                                f"Warning: Not all transitions have {num_cols_plot} trials for plotting. Adjusting num_cols_plot dynamically.")
                            # Fallback if some transitions don't have 6 trials
                            valid_dfs = [df for df in time_series_list if df is not None and not df.empty]
                            if valid_dfs:
                                num_cols_plot = max(len(df) for df in valid_dfs) if valid_dfs else 1
                            else:
                                num_cols_plot = 1

                        num_rows_plot = len(transition_dfs)
                        if num_rows_plot == 0 or num_cols_plot == 0:
                            print(f"Cannot create plot for branch {branch} due to zero rows or columns.")
                        else:
                            fig, axes = plt.subplots(num_rows_plot, num_cols_plot, figsize=(50, 15), sharex=True,
                                                     sharey=True)
                            if num_rows_plot == 1 and num_cols_plot == 1:  # Handle single subplot case
                                axes = np.array([[axes]])
                            elif num_rows_plot == 1:
                                axes = axes.reshape(1, -1)
                            elif num_cols_plot == 1:
                                axes = axes.reshape(-1, 1)

                            plt.subplots_adjust(wspace=0.2, hspace=0.3)
                            fig.suptitle(
                                f"{self.animal}:{self.signal_dir[-21:-7]}\n{branch} DA Changes upon Block Switch",
                                fontsize=20, fontweight='bold', multialignment='center')
                            fig.supxlabel("Time since Entering Background Port (sec)", fontsize=18, fontweight='bold')
                            fig.supylabel("DA (in z-score)", fontsize=18, fontweight='bold')

                            xticks = [0, 1.25, 2.5, 3.75, 5, 6]
                            delivered_label_added = False
                            y_min_plot, y_max_plot = float('inf'), float('-inf')

                            for i, (df_ts, df_trial_info, df_reward) in enumerate(
                                    zip(time_series_list, trial_info_list, reward_list)):
                                num_trials_in_transition = len(df_ts)
                                for j in range(num_trials_in_transition):
                                    if j >= num_cols_plot: continue  # Ensure we don't exceed allocated columns
                                    ax = axes[i, j]
                                    trial_num_abs = int(df_trial_info['trial'].iloc[j])  # Absolute trial number
                                    # Title using relative trial number might be more consistent with new structure
                                    # This assumes your trial_info_df aligns with the 6 trials: -2, -1, 0, 1, 2, 3
                                    # If trial_info_df has a 'relative_trial_num' column, use that.
                                    # For now, keeping original title logic using absolute trial number.
                                    ax.set_title(f"trial {trial_num_abs}")

                                    color = self.block_palette[0] if df_trial_info.at[j, 'phase'] == '0.4' else \
                                        self.block_palette[1]
                                    ax.plot(df_ts.iloc[j], c=color, linewidth=2.5)
                                    if not df_ts.iloc[j].empty:
                                        y_min_plot = min(y_min_plot, df_ts.iloc[j].min())
                                        y_max_plot = max(y_max_plot, df_ts.iloc[j].max())

                                    for x_val in df_reward.iloc[j].tolist():
                                        if pd.notna(x_val):  # Ensure x_val is not NaN
                                            ax.axvline(x_val, color='b', linestyle='--', zorder=1,
                                                       label='Reward delivered' if not delivered_label_added else "")
                                            delivered_label_added = True
                                    if (i == 0) & (
                                            j == 0) and delivered_label_added:  # only add legend if label was set
                                        ax.legend()

                            if y_min_plot == float('inf'): y_min_plot = -1  # Default if no data
                            if y_max_plot == float('-inf'): y_max_plot = 1  # Default if no data
                            yticks = sorted(
                                list(set([0] + [int(round(y)) for y in np.linspace(y_min_plot, y_max_plot, 4)])))

                            for ax_row in axes:
                                for ax in ax_row:
                                    ax.set_xlim(0, 6)
                                    ax.set_xticks(xticks)
                                    ax.tick_params(axis='x', labelsize=14)
                                    ax.set_yticks(yticks)
                                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                            fig.show()

                        if plot_average:
                            # Plotting mean traces (intra-object averaging for selected transitions)
                            # This part uses specific transitions [0, 2, 4] from time_series_list
                            # Ensure these indices are valid for the current time_series_list
                            valid_indices_for_mean_plot = [idx for idx in [0, 2, 4] if idx < len(time_series_list)]
                            if len(valid_indices_for_mean_plot) > 0:
                                selected_dfs_for_mean = [time_series_list[k] for k in valid_indices_for_mean_plot if
                                                         time_series_list[k] is not None and len(
                                                             time_series_list[k]) == 6]

                                if len(selected_dfs_for_mean) > 0:  # Ensure we have DFs to average
                                    mean_traces, sem_traces = [], []
                                    for row_idx in range(6):  # Assumes 6 trials (-2 to +3)
                                        series_list_for_mean = [df.iloc[row_idx].dropna() for df in
                                                                selected_dfs_for_mean if
                                                                row_idx < len(df)]
                                        if not series_list_for_mean: continue  # Skip if no data for this row_idx

                                        aligned_df_for_mean = pd.DataFrame(series_list_for_mean)
                                        if not aligned_df_for_mean.empty:
                                            mean_traces.append(aligned_df_for_mean.mean(axis=0))
                                            sem_traces.append(aligned_df_for_mean.sem(axis=0))

                                    if mean_traces:  # Check if mean_traces were actually computed
                                        mean_df = pd.DataFrame(mean_traces)
                                        sem_df = pd.DataFrame(sem_traces)

                                        # Ensure columns with all NaNs are dropped before plotting
                                        if not mean_df.empty:
                                            valid_columns = ~mean_df.isna().all(axis=0)  # Check if entire column is NaN
                                            mean_df = mean_df.loc[:, valid_columns]
                                            sem_df = sem_df.loc[:, valid_columns]

                                        if not mean_df.empty:
                                            fig_mean, ax_mean = plt.subplots()
                                            low_color = self.block_palette[0]
                                            high_color = self.block_palette[1]
                                            pre_map = sns.light_palette(high_color, n_colors=4, reverse=False)
                                            post_map = sns.light_palette(low_color, n_colors=8, reverse=True)
                                            colors_mean = pre_map[2:4] + post_map[0:4]
                                            legends_mean = ['trial -2', 'trial -1', 'trial 0', 'trial 1', 'trial 2',
                                                            'trial 3']

                                            for r in range(mean_df.shape[0]):
                                                if r < len(colors_mean) and r < len(legends_mean):  # Check bounds
                                                    ax_mean.plot(mean_df.columns, mean_df.iloc[r], color=colors_mean[r],
                                                                 linewidth=1.5, label=legends_mean[r])
                                                    # ax_mean.fill_between(mean_df.columns, mean_df.iloc[r] - sem_df.iloc[r], mean_df.iloc[r] + sem_df.iloc[r],
                                                    #                 alpha=0.2, color=colors_mean[r], linewidth=0, zorder=0) # Ensure sem_df has same shape

                                            ax_mean.set_xlim(1, 4)  # Adjust xlim based on your data's time range
                                            ax_mean.legend()
                                            ax_mean.set_title(
                                                f"{self.animal}:{self.signal_dir[-21:-7]}\n{branch} DA Changes (Mean of Selected Transitions)")
                                            ax_mean.set_xlabel("Time since Entering Background Port (sec)")
                                            ax_mean.set_ylabel("DA (in z-score)")
                                            fig_mean.show()
                                        else:
                                            print(f"Mean DataFrame is empty for branch {branch}, skipping mean plot.")
                                    else:
                                        print(f"No mean traces computed for branch {branch}, skipping mean plot.")
                                else:
                                    print(
                                        f"Not enough valid DataFrames (expected 6 trials each) for mean plotting in branch {branch}.")
                            else:
                                print(f"Not enough transitions data for mean plotting in branch {branch}.")

        print(f'Finished processing for object {self.animal}. Structured data is ready.')
        return DA_in_block_transition

    def visualize_DA_vs_NRI_IRI(self, plot_histograms=0, plot_scatters=0, save=0):
        df_IRI_exp = self.expreward_df
        df = df_IRI_exp
        df = df[df['IRI_post'] > 0.6].reset_index(drop=True)

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

    def extract_lastreward_DA(self):
        df = self.expreward_df.copy()
        last_reward_df = df.groupby(['trial', 'block']).tail(1)
        arr_RXI = last_reward_df['exit_time'].to_numpy() - last_reward_df['reward_time'].to_numpy()
        last_reward_df = last_reward_df[arr_RXI > 0.6].reset_index(drop=True)
        arr_RXI = arr_RXI[arr_RXI > 0.6]
        last_reward_df['RXI'] = arr_RXI
        for branch in ['green_right', 'green_left']:
            if branch not in self.dFF0.columns:
                continue  # Skip if branch is missing
            arr_DA_amp = np.zeros(arr_RXI.shape[0])
            arr_DA_amp.fill(np.nan)
            for row_reward in range(last_reward_df.shape[0]):
                start = last_reward_df.loc[row_reward, 'reward_time']
                end = last_reward_df.loc[row_reward, 'exit_time']
                condition_in_range = (self.zscore['time_recording'] >= start) & (self.zscore['time_recording'] < end)
                is_amp_calculation_range = condition_in_range & (self.zscore['time_recording'] < start + 0.5)
                soi_for_amp = self.zscore.loc[is_amp_calculation_range, branch].to_numpy()
                if np.count_nonzero(~np.isnan(soi_for_amp)) > 1:
                    arr_DA_amp[row_reward] = np.max(soi_for_amp) - np.min(soi_for_amp)
            last_reward_df[f'DA_{branch[6:]}'] = arr_DA_amp

        value_vars = ['DA_right', 'DA_left']
        id_vars = [col for col in last_reward_df.columns if col not in value_vars]
        last_reward_df = last_reward_df.melt(id_vars=id_vars,
                               value_vars=value_vars,
                               var_name='hemisphere',
                               value_name='DA')
        last_reward_df['hemisphere'] = last_reward_df['hemisphere'].str.replace('DA_', '')
        self.last_reward_df = last_reward_df

    def extract_nonreward_DA_vs_time(self, exclusion_start_relative, exclusion_end_relative):
        # df_IRI_exp = helper.extract_intervals_expreward(self.pi_events, plot_histograms=0,
        #                                                 ani_str=self.animal,
        #                                                 ses_str=self.signal_dir[-21:-7])
        reward_times = self.expreward_df['reward_time'].to_numpy()
        zscore_times = self.zscore['time_recording'].to_numpy()
        keep_mask = np.full(zscore_times.shape, True)
        exclusion_start = reward_times + exclusion_start_relative
        exclusion_end = reward_times + exclusion_end_relative
        for (start, end) in zip(exclusion_start, exclusion_end):
            is_within_vicinity = (zscore_times >= start) & (zscore_times < end)
            keep_mask[is_within_vicinity] = False
        nonreward_DA_vs_time = self.zscore[keep_mask].reset_index(drop=True)
        nonreward_DA_vs_time.dropna(inplace=True, how='any')
        nonreward_DA_vs_time.reset_index(drop=True)
        port_entry = self.trial_df['exp_entry'].to_numpy()
        port_exit = self.trial_df['exp_exit'].to_numpy()
        trial_numbers = self.trial_df['trial'].to_numpy()  # Get trial IDs for assignment
        block = self.trial_df['phase'].to_numpy()
        nonreward_times = nonreward_DA_vs_time['time_recording'].to_numpy()

        nonreward_DA_vs_time['phase'] = np.nan
        nonreward_DA_vs_time['trial'] = np.nan
        nonreward_DA_vs_time['time_in_port'] = np.nan

        # Map time points to trials and calculating relative time ---
        for i in range(len(port_entry)):
            entry_time = port_entry[i]
            exit_time = port_exit[i]
            trial_id = trial_numbers[i]
            block_i = block[i]
            is_in_current_trial = (nonreward_times >= entry_time) & (nonreward_times < exit_time)
            indices_to_update = nonreward_DA_vs_time.loc[is_in_current_trial].index
            if not indices_to_update.empty:
                nonreward_DA_vs_time.loc[indices_to_update, 'phase'] = block_i
                nonreward_DA_vs_time.loc[indices_to_update, 'trial'] = trial_id
                time_relative_to_entry = (
                        nonreward_DA_vs_time.loc[indices_to_update, 'time_recording'] - entry_time
                )
                nonreward_DA_vs_time.loc[indices_to_update, 'time_in_port'] = time_relative_to_entry

        nonreward_DA_vs_time = nonreward_DA_vs_time.dropna(subset=['phase', 'trial', 'time_in_port']).reset_index(
            drop=True)
        self.nonreward_DA_vs_time = nonreward_DA_vs_time

    def extract_firstmoment_nonreward_DA_vs_time(self):
        first_rewards = self.expreward_df.groupby('trial').first()
        entry_times = first_rewards['entry_time'].to_numpy()
        reward_times = first_rewards['reward_time'].to_numpy()
        zscore_times = self.zscore['time_recording'].to_numpy()
        keep_mask = np.full(zscore_times.shape, False)
        inclusion_start = entry_times
        inclusion_end = reward_times
        for (start, end) in zip(inclusion_start, inclusion_end):
            is_within_1stmoment = (zscore_times >= start) & (zscore_times < end)
            keep_mask[is_within_1stmoment] = True
        nonreward1stmoment_DA_vs_time = self.zscore[keep_mask].reset_index(drop=True)
        nonreward1stmoment_DA_vs_time.dropna(inplace=True, how='any')
        nonreward1stmoment_DA_vs_time.reset_index(drop=True)
        port_entry = self.trial_df['exp_entry'].to_numpy()
        port_exit = self.trial_df['exp_exit'].to_numpy()
        trial_numbers = self.trial_df['trial'].to_numpy()  # Get trial IDs for assignment
        block = self.trial_df['phase'].to_numpy()
        nonreward_times = nonreward1stmoment_DA_vs_time['time_recording'].to_numpy()

        nonreward1stmoment_DA_vs_time['phase'] = np.nan
        nonreward1stmoment_DA_vs_time['trial'] = np.nan
        nonreward1stmoment_DA_vs_time['time_in_port'] = np.nan

        # Map time points to trials and calculating relative time ---
        for i in range(len(port_entry)):
            entry_time = port_entry[i]
            exit_time = port_exit[i]
            trial_id = trial_numbers[i]
            block_i = block[i]
            is_in_current_trial = (nonreward_times >= entry_time) & (nonreward_times < exit_time)
            indices_to_update = nonreward1stmoment_DA_vs_time.loc[is_in_current_trial].index
            if not indices_to_update.empty:
                nonreward1stmoment_DA_vs_time.loc[indices_to_update, 'phase'] = block_i
                nonreward1stmoment_DA_vs_time.loc[indices_to_update, 'trial'] = trial_id
                time_relative_to_entry = (
                        nonreward1stmoment_DA_vs_time.loc[indices_to_update, 'time_recording'] - entry_time
                )
                nonreward1stmoment_DA_vs_time.loc[indices_to_update, 'time_in_port'] = time_relative_to_entry

        nonreward1stmoment_DA_vs_time = nonreward1stmoment_DA_vs_time.dropna(
            subset=['phase', 'trial', 'time_in_port']).reset_index(
            drop=True)
        self.nonreward1stmoment_DA_vs_time = nonreward1stmoment_DA_vs_time

    def visualize_nonreward_DA(self, plot_1st_moment=True, plot_all=True, bin_size=0.5):
        if plot_1st_moment:
            title = f'{self.animal} {self.signal_dir[-23:-7]}: 1st moment non-reward DA'
            helper.visualize_nonreward_DA(self.nonreward1stmoment_DA_vs_time, fig_title=title, bin_size=bin_size/3)
        if plot_all:
            title = f'{self.animal} {self.signal_dir[-23:-7]}: non-reward DA'
            helper.visualize_nonreward_DA(self.nonreward_DA_vs_time, fig_title=title, bin_size=bin_size)

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
        binned_means = helper.construct_reward_history_matrix(binned_means, binsize=binsize)
        if save:
            df_to_save = binned_means[['animal', 'trial', 'bin', 'bin_idx', 'reward_num', 'green_right', 'green_left',
                                       'history_matrix_sparse']]
            if self.include_branch == 'only_left':
                df_to_save['green_right'] = np.nan
            elif self.include_branch == 'only_right':
                df_to_save['green_left'] = np.nan
            df_to_save.to_pickle(os.path.join(self.processed_dir, 'binned_DA_reward_history',
                                              f'{self.animal}_{self.signal_dir[-23:-7]}_binned_DA_vs_history.pkl'))

    # --- Data Saving Methods ---
    def _save_data_object(self, data_object, object_name_for_file, format='parquet', **kwargs):
        """
        Helper method to save a data object to the processed_dir.
        Args:
            data_object: The data to save.
            object_name_for_file (str): String to use in the filename.
            format (str): 'parquet', 'csv', 'pickle', 'npy'.
            **kwargs: Additional keyword arguments to pass to the save function
                      (e.g., index=False for to_csv).
        """
        if data_object is None:
            print(f"Data for '{object_name_for_file}' is None. Nothing to save.")
            return

        if isinstance(data_object, pd.DataFrame) and data_object.empty:
            print(f"DataFrame '{object_name_for_file}' is empty. Nothing to save.")
            return

        os.makedirs(self.processed_dir, exist_ok=True)
        session_identifier = self.signal_dir[-23:-7]
        filename_base = f"{self.animal}_{session_identifier}_{object_name_for_file}"

        # Adjust extension based on format for some types
        actual_format = format
        if isinstance(data_object, np.ndarray) and format != 'npy':
            print(f"NumPy array will be saved in .npy format for {object_name_for_file}.")
            actual_format = 'npy'  # Force .npy for numpy arrays if np.save is used

        file_path = os.path.join(self.processed_dir, f"{filename_base}.{actual_format}")

        try:
            if isinstance(data_object, pd.DataFrame) or isinstance(data_object, pd.Series):
                if format == 'parquet':
                    data_object.to_parquet(file_path, **kwargs)
                elif format == 'pickle':
                    data_object.to_pickle(file_path, **kwargs)
                elif format == 'csv':
                    # Pass index=False by default if not specified in kwargs for CSV
                    kwargs.setdefault('index', False)
                    data_object.to_csv(file_path, **kwargs)
                else:
                    print(f"Unsupported format '{format}' for DataFrame. Not saved.")
                    return
            elif isinstance(data_object, np.ndarray):
                if actual_format == 'npy':  # np.save expects .npy
                    np.save(file_path, data_object)  # np.save doesn't take **kwargs generally
                else:  # Should not happen if we force 'npy'
                    print(f"Cannot save NumPy array {object_name_for_file} in format {format}")
                    return
            elif format == 'pickle':  # Fallback for other types like lists, dictionaries, and custom Python objects
                with open(file_path, 'wb') as f:
                    pickle.dump(data_object, f)
            else:
                print(f"Cannot save type {type(data_object)} with format {format} unless it's 'pickle'.")
                return

            print(f"Saved {object_name_for_file} to {file_path}")

        except Exception as e:
            print(f"Error saving {object_name_for_file} to {file_path}: {e}")

    def save_dFF0_and_zscore(self, format='parquet'):
        if self.dFF0 is not None:
            self._save_data_object(self.dFF0, "dFF0", format)
        if self.zscore is not None:  # The empty check is now inside _save_data_object
            self._save_data_object(self.zscore, "zscore", format)

    def save_pi_events(self, format='csv'):  # pi_events might be best as CSV
        if self.pi_events is not None:
            if format == 'csv':
                self._save_data_object(self.pi_events, "pi_events_processed", format, index=False)
            else:
                self._save_data_object(self.pi_events, "pi_events_processed", format)

    def save_trial_df(self, format='parquet'):  # pi_events might be best as CSV
        if self.trial_df is not None:
            if format == 'csv':
                self._save_data_object(self.trial_df, "trial_df", format, index=False)
            else:
                self._save_data_object(self.trial_df, "trial_df", format)

    def save_bg_behavior_trial_df(self, format='parquet'):
        if self.bg_behav_by_trial is not None:
            if format == 'csv':
                self._save_data_object(self.bg_behav_by_trial, "bg_trial_df", format, index=False)
            else:
                self._save_data_object(self.bg_behav_by_trial, "bg_trial_df", format)

    def save_expreward_df(self, format='parquet'):
        if self.expreward_df is not None:
            if format == 'csv':
                self._save_data_object(self.expreward_df, "expreward_df", format, index=False)
            else:
                self._save_data_object(self.expreward_df, "expreward_df", format)

    def save_DA_vs_features(self, format='parquet'):
        if self.DA_vs_NRI_IRI is not None:
            if format == 'csv':
                self._save_data_object(self.DA_vs_NRI_IRI, "DA_vs_features", format, index=False)
            else:
                self._save_data_object(self.DA_vs_NRI_IRI, "DA_vs_features", format)

    def save_lastreward_df(self, format='parquet'):
        if self.last_reward_df is not None:
            self._save_data_object(self.last_reward_df, "DA_vs_lastR_df", format)

    def save_nonreward_1stmoment_DA(self, format='parquet'):
        if self.nonreward1stmoment_DA_vs_time is not None:
            if format == 'csv':
                self._save_data_object(self.nonreward1stmoment_DA_vs_time, "nonreward_1st", format, index=False)
            else:
                self._save_data_object(self.nonreward1stmoment_DA_vs_time, "nonreward_1st", format)

    def save_nonreward_DA(self, format='parquet'):
        if self.nonreward_DA_vs_time is not None:
            if format == 'csv':
                self._save_data_object(self.nonreward_DA_vs_time, "nonreward_DA", format, index=False)
            else:
                self._save_data_object(self.nonreward_DA_vs_time, "nonreward_DA", format)


if __name__ == '__main__':
    test_session = OneSession('SZ036', 15, include_branch='both', port_swap=0)
    # test_session.examine_raw(save=0)
    test_session.calculate_dFF0(plot=0, plot_middle_step=0, save=0)
    # test_session.save_dFF0_and_zscore(format='parquet')
    # test_session.remove_outliers_dFF0()
    test_session.process_behavior_data(save=0)
    # test_session.save_pi_events(format='parquet')
    test_session.construct_trial_df()
    # test_session.save_trial_df(format='parquet')
    test_session.add_trial_info_to_recording()
    test_session.construct_expreward_interval_df()
    test_session.extract_lastreward_DA()
    test_session.extract_firstmoment_nonreward_DA_vs_time()
    test_session.extract_nonreward_DA_vs_time(exclusion_start_relative=0, exclusion_end_relative=2)
    # test_session.visualize_nonreward_DA(bin_size=1)
    # test_session.save_expreward_df(format='parquet')
    # test_session.extract_bg_behav_by_trial()
    # test_session.plot_reward_aligned_lick_histograms()
    # test_session.calculate_lick_rates_around_bg_reward(reward_idx_to_align=2, plot_comparison=1)
    # test_session.bg_lick_rasterplot()
    # test_session.extract_transient(plot_zscore=0)
    # test_session.visualize_correlation_scatter(save=0)
    # test_session.plot_heatmaps(save=1)
    # test_session.plot_bg_heatmaps(save=0)
    # test_session.actual_leave_vs_adjusted_optimal(save=0)
    test_session.extract_reward_features_and_DA(plot=0, save_dataframe=0)
    df_intervals_exp = test_session.visualize_average_traces(variable='time_in_port', method='even_time',
                                                             block_split=False,
                                                             plot_histograms=0, plot_linecharts=0)
    test_session.visualize_DA_vs_NRI_IRI(plot_scatters=0, plot_histograms=0)
    test_session.save_DA_vs_features(format='csv')
    # DA_in_block_transition = test_session.bg_port_in_block_reversal(plot_single_traes=0, plot_average=0)

    # test_session.scatterplot_nonreward_DA_vs_NRI()
    test_session.for_pub_compare_traces_by_NRI(branch='green_left')
    # test_session.extract_binned_da_vs_reward_history_matrix(binsize=0.1, save=0)

    print("Hello")
