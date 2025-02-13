import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from matplotlib import pyplot as plt
import seaborn as sns

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

        self.animal = animal_str
        self.behav_dir = os.path.join(animal_dir, 'raw_data', behav_file_list[session])
        self.signal_dir = os.path.join(animal_dir, 'raw_data', FP_file_list[session])
        self.arduino_dir = os.path.join(animal_dir, 'raw_data', TTL_file_list[session])
        self.fig_export_dir = os.path.join(animal_dir, 'figures', self.signal_dir[-21:-7])
        print("This class is" + " session " + self.signal_dir[-23:-7])
        self.pi_events, self.neural_events = func.data_read_sync(self.behav_dir, self.signal_dir, self.arduino_dir)
        if self.pi_events['task'].iloc[10] == 'single_reward':
            self.task = 'single_reward'
        elif self.pi_events['task'].iloc[10] == 'cued_no_forgo_forced':
            self.task = 'multi_reward'

        self.block_palette = sns.color_palette('Set2')

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

    def process_behavior_data(self):
        self.pi_events["time_recording"] = (self.pi_events['time'] - self.neural_events.timestamps[0]) / 1000
        self.pi_events = func.data_reduction(self.pi_events, lick_tol=.01, head_tol=0.2)
        self.pi_events = func.add_2ndry_properties_to_pi_events(self.pi_events)
        # self.pi_events.reset_index(drop=True, inplace=True)
        self.idx_taskbegin = self.dFF0.index[
            self.dFF0['time_recording'] >= self.pi_events['time_recording'].min()].min()
        self.idx_taskend = self.dFF0.index[self.dFF0['time_recording'] <= self.pi_events['time_recording'].max()].max()

    # this function would depend on the execution of self.process_behavior_data()
    def actual_leave_vs_adjusted_optimal(self, save=0):
        self.intervals_df = func.make_intervals_df(self.pi_events)
        func.visualize_adjusted_optimal(self.intervals_df, save=save, save_path=self.fig_export_dir)

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

    def visualize_average_traces(self, variable='time_in_port', method='even_time', block_split=False,
                                 plot_linecharts=0, plot_histograms=0, save=0):
        # The two options for variable are: 'time_in_port', 'IRI_prior'
        # The three options of methods are: 'even_time', 'overall_quint', 'separate_quint'.

        df_IRI_exp = func.extract_intervals_expreward(self.pi_events, plot_histograms=plot_histograms,
                                                      ani_str=self.animal,
                                                      ses_str=self.signal_dir[-21:-7])
        df_IRI_exp['quintile'] = pd.qcut(df_IRI_exp[variable], q=5, labels=False)

        col_name_obj = self.dFF0.columns
        for branch in {col_name_obj[i] for i in range(1, col_name_obj.size)}:
            avr_all = np.empty((5, 2))
            avr_low = np.empty((5, 2))
            avr_high = np.empty((5, 2))
            avr_all[:] = np.nan
            avr_low[:] = np.nan
            avr_high[:] = np.nan
            if method == 'even_time':
                if variable == 'time_in_port':
                    avr_all[:, 0] = [1, 3, 5, 7, 9]
                    avr_low[:, 0] = [1, 3, 5, 7, 9]
                    avr_high[:, 0] = [1, 3, 5, 7, 9]
                elif variable == 'IRI_prior':
                    avr_all[:, 0] = [0.5, 1.5, 2.5, 3.5, 4.5]
                    avr_low[:, 0] = [0.5, 1.5, 2.5, 3.5, 4.5]
                    avr_high[:, 0] = [0.5, 1.5, 2.5, 3.5, 4.5]
            if method == 'overall_quint':
                avr_all[:, 0] = df_IRI_exp.groupby('quintile')[variable].median().to_numpy()
                avr_low[:, 0] = df_IRI_exp.groupby('quintile')[variable].median().to_numpy()
                avr_high[:, 0] = df_IRI_exp.groupby('quintile')[variable].median().to_numpy()
            if plot_linecharts:
                fig, axes = plt.subplots(3, 2, figsize=(20, 20), sharex=True, sharey=True)
            for i in range(5):
                # get the relevant trials according to how I divide them
                if method == 'even_time':
                    if variable == 'time_in_port':
                        lower_bound = 2 * i
                        upper_bound = 2 * (i + 1)
                    elif variable == 'IRI_prior':
                        lower_bound = i
                        upper_bound = i + 1
                    df = df_IRI_exp[(df_IRI_exp[variable] > lower_bound) & (df_IRI_exp[variable] < upper_bound)]
                if method == 'overall_quint':
                    df = df_IRI_exp[df_IRI_exp['quintile'] == i]  # This line divides the NRIs into quintiles
                    lower_bound = df[variable].min()
                    upper_bound = df[variable].max()
                label = f'{np.round(lower_bound, decimals=1)} - {np.round(upper_bound, decimals=1)} sec'

                # get the matrices and the summary statistics to store in the self.NRI_amplitude dataframes
                if variable == 'time_in_port':
                    title = 'rewarded at ' + label + ' since entry'
                elif variable == 'IRI_prior':
                    title = 'rewarded at ' + label + ' since last reward'
                df_for_average, df_trial_info = func.construct_matrix_for_average_traces(self.zscore, branch, df.loc[
                    df['next_reward_time'].notna(), 'reward_time'].to_numpy(), df.loc[df[
                    'next_reward_time'].notna(), 'next_reward_time'].to_numpy(), df.loc[df[
                    'next_reward_time'].notna(), 'trial'].to_numpy(), df.loc[df[
                    'next_reward_time'].notna(), 'block'].to_numpy(), interval_name=None)
                avr_all[i, 1] = df_for_average.loc[:,
                                df_for_average.columns < 0.5].mean().max() - df_for_average.loc[:,
                                                                             df_for_average.columns < 0.5].mean().min()
                is_low_block = df_trial_info['phase'] == '0.4'
                is_high_block = df_trial_info['phase'] == '0.8'
                avr_low[i, 1] = df_for_average[is_low_block].loc[:,
                                df_for_average.columns < 0.5].mean().max() - df_for_average.loc[:,
                                                                             df_for_average.columns < 0.5].mean().min()
                avr_high[i, 1] = df_for_average[is_high_block].loc[:,
                                 df_for_average.columns < 0.5].mean().max() - df_for_average.loc[:,
                                                                              df_for_average.columns < 0.5].mean().min()

                # fill in the linecharts with info from this individual group if needed
                if plot_linecharts:
                    if block_split:
                        low_palette = sns.light_palette(self.block_palette[0], df_for_average.shape[
                            0])  # make a graded palette based on the low block color (green)
                        high_palette = sns.light_palette(self.block_palette[1], df_for_average.shape[0])
                        for j in range(df_for_average[is_high_block].shape[0]):
                            axes[i - i // 3 * 3, i // 3].plot(df_for_average[is_high_block].iloc[j], c=high_palette[j],
                                                              linewidth=1)
                        for j in range(df_for_average[is_low_block].shape[0]):
                            axes[i - i // 3 * 3, i // 3].plot(df_for_average[is_low_block].iloc[j], c=low_palette[j],
                                                              linewidth=1)
                        axes[i - i // 3 * 3, i // 3].plot(df_for_average[is_low_block].mean(axis=0),
                                                          c=self.block_palette[0], linewidth=3, label='0.4 block')
                        axes[i - i // 3 * 3, i // 3].plot(df_for_average[is_high_block].mean(axis=0),
                                                          c=self.block_palette[1], linewidth=3, label='0.8 block')
                    else:
                        for j in range(df_for_average.shape[0]):
                            weight = np.round(0.8 - (j + 1) / (df_for_average.shape[0] + 1) * 0.4, decimals=2)
                            axes[i - i // 3 * 3, i // 3].plot(df_for_average.iloc[j], c=str(weight))
                        axes[i - i // 3 * 3, i // 3].plot(df_for_average.mean(axis=0), c='b', linewidth=2.5)

                    # axes[i - i // 3 * 3, i // 3].plot(df_for_average.mean(axis=0), c='b', linewidth=2.5)
                    # axes[i - i // 3 * 3, i // 3].plot(df_for_average[is_low_block].mean(axis=0),
                    #                                   c=self.block_palette[0], linewidth=3, label='0.4 block')
                    # axes[i - i // 3 * 3, i // 3].plot(df_for_average[is_high_block].mean(axis=0),
                    #                                   c=self.block_palette[1], linewidth=3, label='0.8 block')
                    axes[i - i // 3 * 3, i // 3].legend()
                    axes[i - i // 3 * 3, i // 3].tick_params(axis='y', labelsize=15)
                    axes[i - i // 3 * 3, i // 3].tick_params(axis='x', labelsize=15)
                    # axes[i - i // 3 * 3, i // 3].set_xlim(0, 6)
                    # axes[i - i // 3 * 3, i // 3].set_ylim(-1.9, 5.1)
                    axes[i - i // 3 * 3, i // 3].set_title(title, fontsize=15)  # equal time range
                    axes[2, 1].plot(df_for_average.mean(axis=0), linewidth=2.5, label=label)  # equal time range
                    axes[2, 1].set_title('All Average IRI Traces')
                    axes[2, 1].legend(loc='upper right')
                    # avr_all[i, 1] = df_for_average.loc[:,
                    #                          df_for_average.columns < 0.5].mean().max() - df_for_average.loc[:,
                    #                                                                       df_for_average.columns < 0.5].mean().min()

            # Make the miniature dot chart in the summary panel of the linecharts
            if plot_linecharts:
                plt.tight_layout()
                rect = [0.3, 0.59, 0.4,
                        0.4]  # The first two elements give the position; The third and fourth give the size
                ax1 = func.add_subplot_axes(axes[2, 1], rect)
                ax1.scatter(avr_all[:, 0], avr_all[:, 1], c='k', marker='o', s=15)
                if block_split:
                    ax1.scatter(avr_low[:, 0], avr_low[:, 1], c=self.block_palette[0], marker='o', s=20,
                                label='0.4 block')
                    ax1.scatter(avr_high[:, 0], avr_high[:, 1], c=self.block_palette[1], marker='o', s=20,
                                label='0.8 block')
                ax1.legend()
                ax1.set_xlabel('Median Delay (sec)', fontsize=15)
                ax1.set_ylabel('Average Amplitude', fontsize=15)
                fig.text(0.5, 0.008, 'Time since Reward (sec)', ha='center', va='center', fontsize=18)
                fig.text(0.01, 0.5, 'DA (Z-score)', ha='center', va='center', rotation='vertical', fontsize=20)
                fig.suptitle(f"{self.animal}:{self.signal_dir[-21:-7]} {branch}", x=0.5, y=0.99, ha='center',
                             va='center',
                             fontsize=20)
                fig.show()

            if branch == 'green_right':
                if variable == 'time_in_port':
                    self.NRI_amplitude_r = pd.DataFrame(
                        {'median_interval': avr_all[:, 0], 'amp_all': avr_all[:, 1], 'amp_low': avr_low[:, 1],
                         'amp_high': avr_high[:, 1]})
                elif variable == 'IRI_prior':
                    self.IRI_amplitude_r = pd.DataFrame(
                        {'median_interval': avr_all[:, 0], 'amp_all': avr_all[:, 1], 'amp_low': avr_low[:, 1],
                         'amp_high': avr_high[:, 1]})
            elif branch == 'green_left':
                if variable == 'time_in_port':
                    self.NRI_amplitude_l = pd.DataFrame(
                        {'median_interval': avr_all[:, 0], 'amp_all': avr_all[:, 1], 'amp_low': avr_low[:, 1],
                         'amp_high': avr_high[:, 1]})
                elif variable == 'IRI_prior':
                    self.IRI_amplitude_l = pd.DataFrame(
                        {'median_interval': avr_all[:, 0], 'amp_all': avr_all[:, 1], 'amp_low': avr_low[:, 1],
                         'amp_high': avr_high[:, 1]})

        return df_IRI_exp

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
                    ax.plot(df.iloc[j], c=color, linewidth=2.5) # Plot time series
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


if __name__ == '__main__':
    test_session = OneSession('SZ042', 19, include_branch='both')
    # test_session.examine_raw(save=0)
    test_session.calculate_dFF0(plot=0, plot_middle_step=0, save=0)
    # test_session.remove_outliers_dFF0()
    test_session.process_behavior_data()
    # test_session.extract_transient(plot_zscore=0)
    # test_session.visualize_correlation_scatter(save=0)
    # test_session.plot_heatmaps(save=1)
    # test_session.plot_bg_heatmaps(save=0)
    # test_session.actual_leave_vs_adjusted_optimal(save=0)
    df_intervals_exp = test_session.visualize_average_traces(variable='time_in_port', method='even_time',
                                                             block_split=True,
                                                             plot_histograms=0, plot_linecharts=1)
    # test_session.bg_port_in_block_reversal()

    print("Hello")
