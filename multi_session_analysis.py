import os
import helper
import config
from OneSession import OneSession
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
import seaborn as sns
import numpy as np
import statistics
from L import L
import pandas as pd
from scipy import stats
from collections import namedtuple
import pingouin as pg  # packages for anova according to Google


def visualize_transient_consistency(session_obj_list, save=0, save_path=None):
    transient_occur_xsession_l = np.empty(len(session_obj_list))
    transient_occur_xsession_r = np.empty(len(session_obj_list))
    transient_midmag_xsession_l = np.empty(len(session_obj_list))
    transient_midmag_xsession_r = np.empty(len(session_obj_list))
    for i in range(len(session_obj_list)):
        transient_occur_xsession_l[i] = session_obj_list[i].transient_occur_l
        transient_occur_xsession_r[i] = session_obj_list[i].transient_occur_r
        if session_obj_list[i].transient_midmag_l is not None:
            transient_midmag_xsession_l[i] = session_obj_list[i].transient_midmag_l * 100
        if session_obj_list[i].transient_midmag_r is not None:
            transient_midmag_xsession_r[i] = session_obj_list[i].transient_midmag_r * 100
    session_selected_l = np.where(
        (transient_midmag_xsession_l >= np.nanpercentile(transient_midmag_xsession_l, 25) - 0.5) & (
                transient_midmag_xsession_l <= np.nanpercentile(transient_midmag_xsession_l,
                                                                75) + 0.8))
    session_selected_r = np.where(
        (transient_midmag_xsession_r >= np.nanpercentile(transient_midmag_xsession_r, 25) - 0.5) & (
                transient_midmag_xsession_r <= np.nanpercentile(transient_midmag_xsession_r,
                                                                75) + 0.8))
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(hspace=0)
    if sum(transient_occur_xsession_l == None) != len(transient_occur_xsession_l):
        ax[0].plot(transient_occur_xsession_l, label='left hemisphere')
        ax[1].plot(transient_midmag_xsession_l, label='left hemisphere')
    if sum(transient_occur_xsession_r == None) != len(transient_occur_xsession_r):
        ax[0].plot(transient_occur_xsession_r, label='right hemisphere')
        ax[1].plot(transient_midmag_xsession_r, label='right hemisphere')
    ax[1].axhspan(np.nanpercentile(transient_midmag_xsession_l, 25) - 0.5,
                  np.nanpercentile(transient_midmag_xsession_l, 75) + 0.8, color='b', alpha=0.2, zorder=-10)
    ax[1].axhspan(np.nanpercentile(transient_midmag_xsession_r, 25) - 0.5,
                  np.nanpercentile(transient_midmag_xsession_r, 75) + 0.8, color='orange', alpha=0.2, zorder=-10)
    ax[0].scatter(session_selected_l, transient_occur_xsession_l[session_selected_l], color='b', marker='*', zorder=10)
    ax[0].scatter(session_selected_r, transient_occur_xsession_r[session_selected_r], color='orange', marker='*',
                  zorder=10)
    ax[1].scatter(session_selected_l, transient_midmag_xsession_l[session_selected_l], color='b', marker='*', zorder=10)
    ax[1].scatter(session_selected_r, transient_midmag_xsession_r[session_selected_r], color='orange', marker='*',
                  zorder=10)
    ax[0].legend()
    ax[0].set_ylabel('Transient Occurance Rate (/min)', fontsize=15)
    ax[0].set_ylim([0, 25])
    ax[1].set_ylabel('Median Transient Magnitude (%)', fontsize=15)
    ax[1].set_ylim([0, 15])
    ax[1].set_xlabel('Session', fontsize=20)
    fig.suptitle(f'{session_obj_list.animal}: Transient Consistency Analysis', fontsize=30)
    fig.show()
    if save:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path, 'TransientConsistencyXSessions' + '.png')
        fig.savefig(fig_name)
    return session_selected_l, session_selected_r


def visualize_NRI_vs_amplitude_families(ani_summary, variable='time_in_port', block_split=False, normalized=False,
                                        save=False,
                                        save_path=None):
    # variable has two options: 'time_in_port' and 'IRI_prior'
    num_session = len([x for x in ani_summary.session_obj_list if x is not None])
    median_delay_arr = np.zeros((5, num_session))
    amp_arr_low = np.zeros((5, num_session))
    amp_arr_high = np.zeros((5, num_session))
    average_amplitude_arr = np.zeros((5, num_session))

    if block_split:
        # cmap = plt.get_cmap('plasma')
        # new_cmap = cmap.resampled(len(session_obj_list))
        block_palette = sns.color_palette('Set2')
        low_palette = sns.light_palette(block_palette[0], 1.5 * len(ani_summary.session_obj_list))
        high_palette = sns.light_palette(block_palette[1], 1.5 * len(ani_summary.session_obj_list))

    for branch in {'ipsi', 'contra'}:
        fig = plt.figure()
        idx_arr = 0
        for i in range(len(ani_summary.session_obj_list)):
            if ani_summary.session_obj_list[i] == None:
                continue
            else:
                if branch == 'contra':
                    if variable == 'time_in_port':
                        interval_amp_df = ani_summary.session_obj_list[i].NRI_amplitude_l
                    elif variable == 'IRI_prior':
                        interval_amp_df = ani_summary.session_obj_list[i].IRI_amplitude_l
                else:
                    if variable == 'time_in_port':
                        interval_amp_df = ani_summary.session_obj_list[i].NRI_amplitude_r
                    elif variable == 'IRI_prior':
                        interval_amp_df = ani_summary.session_obj_list[i].IRI_amplitude_r
                if interval_amp_df is None:
                    continue
                else:
                    median_delay_arr[:, idx_arr] = interval_amp_df[
                        'median_interval'].to_numpy()  # 'delay' and 'interval' mean the same thing here. Just don't get confused

                    # store the average amplitude of each quintile from each session into a big 2d matrix, so that we can average them across session later
                    amp_arr_low[:, idx_arr] = interval_amp_df['amp_low'].to_numpy()
                    amp_arr_high[:, idx_arr] = interval_amp_df['amp_high'].to_numpy()
                    average_amplitude_arr[:, idx_arr] = interval_amp_df['amp_all'].to_numpy()

                    idx_arr += 1

                    if block_split & normalized:
                        weight = np.round(0.8 - (i + 1) / (len(ani_summary.session_obj_list) + 1) * 0.3, decimals=2)
                        normalized_interval_amplitude_l = interval_amp_df['amp_low'] / \
                                                          interval_amp_df['amp_high']
                        plt.plot(interval_amp_df['median_interval'],
                                 normalized_interval_amplitude_l, c=str(weight), marker='o')
                    elif block_split & (not normalized):
                        plt.plot(interval_amp_df['median_interval'],
                                 interval_amp_df['amp_low'], c=low_palette[i], marker='o')
                        plt.plot(interval_amp_df['median_interval'],
                                 interval_amp_df['amp_high'], c=high_palette[i], marker='o')
                    else:
                        weight = np.round(0.8 - (i + 1) / (len(ani_summary.session_obj_list) + 1) * 0.3, decimals=2)
                        plt.plot(interval_amp_df['median_interval'],
                                 interval_amp_df['amp_all'], c=str(weight), marker='o')

        if block_split & normalized:
            normalized_amp_arr_low = amp_arr_low / amp_arr_high
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(normalized_amp_arr_low, axis=1), c=block_palette[0],
                     marker='o', markersize=10, zorder=10)
            plt.plot(np.median(median_delay_arr, axis=1), np.ones(5), c=block_palette[1], marker='o',
                     markersize=10, zorder=10)
        elif block_split & (not normalized):
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(amp_arr_low, axis=1), c=block_palette[0], marker='o',
                     markersize=10, zorder=10)
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(amp_arr_high, axis=1), c=block_palette[1], marker='o',
                     markersize=10, zorder=10)
        else:
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(average_amplitude_arr, axis=1),
                     marker='o', markersize=10, zorder=10)

        if variable == 'time_in_port':
            plt.title(f'{ani_summary.animal}: {branch} DA Amplitude vs Time since Entry', fontsize=20)
            plt.xlabel('Median Delay off Entry (sec)', fontsize=15)
        elif variable == 'IRI_prior':
            plt.title(f'{ani_summary.animal}: {branch} DA Amplitude vs Time since Last Reward', fontsize=20)
            plt.xlabel('Median IRI (sec)', fontsize=15)
        plt.ylabel('Z-score Amplitude', fontsize=15)
        fig.show()

        # 'ani_sum' stands for animal summary.
        # This dataframe stores the median and quartiles of NRIs and the mean and SEM of DA amplitudes
        # from all sessions of one animal's one hemisphere
        if branch == 'contra':
            if variable == 'time_in_port':
                ani_summary.NRI_amp_contra['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.NRI_amp_contra['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.NRI_amp_contra['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.NRI_amp_contra['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_contra['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_contra['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.NRI_amp_contra['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.NRI_amp_contra['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.NRI_amp_contra['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)
            elif variable == 'IRI_prior':
                ani_summary.IRI_amp_contra['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.IRI_amp_contra['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.IRI_amp_contra['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.IRI_amp_contra['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_contra['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_contra['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.IRI_amp_contra['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.IRI_amp_contra['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.IRI_amp_contra['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)
        elif branch == 'ipsi':
            if variable == 'time_in_port':
                ani_summary.NRI_amp_ipsi['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.NRI_amp_ipsi['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.NRI_amp_ipsi['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.NRI_amp_ipsi['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_ipsi['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_ipsi['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.NRI_amp_ipsi['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.NRI_amp_ipsi['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.NRI_amp_ipsi['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)
            elif variable == 'IRI_prior':
                ani_summary.IRI_amp_ipsi['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.IRI_amp_ipsi['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.IRI_amp_ipsi['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.IRI_amp_ipsi['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_ipsi['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_ipsi['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.IRI_amp_ipsi['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.IRI_amp_ipsi['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.IRI_amp_ipsi['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)


def visualize_NRI_reward_history_vs_DA(ani_summary):
    df_list = []
    for session in ani_summary.session_obj_list:
        if session is not None:
            df_temp = session.DA_NRI_block_priorrewards[
                ['block', 'time_in_port_bin', 'num_rewards_prior_bin', 'right_DA_mean', 'left_DA_mean']].copy()
            df_list.append(df_temp)
    df_combined = pd.concat(df_list, ignore_index=True)
    df_final = df_combined.groupby(['block', 'time_in_port_bin', 'num_rewards_prior_bin'])[
        ['right_DA_mean', 'left_DA_mean']].agg(['mean', 'sem']).reset_index()
    df_final.columns = ['block', 'time_in_port_bin', 'num_rewards_prior_bin', 'right_DA_mean', 'right_DA_sem',
                        'left_DA_mean', 'left_DA_sem']

    # Define unique values for styling
    blocks = df_final['block'].unique()
    prior_rewards = sorted(df_final['num_rewards_prior_bin'].unique())
    prior_reward_labels = {0: '0-2 rewards', 1: '2-4', 2: '4-6', 3: '6-8', 4: '8-10'}
    colors = sns.color_palette('Set2')  # Unique colors for blocks
    sizes = np.linspace(50, 100, len(prior_rewards))  # Marker sizes
    linewidths = np.linspace(1, 4, len(prior_rewards))  # Line thickness
    capsizes = np.linspace(2.5, 5, len(prior_rewards))

    def plot_DA(df, DA_col, DA_sem, title):
        plt.figure(figsize=(7, 5))

        for i, block in enumerate(blocks):
            x_jitter = np.linspace(-0.1, 0.1, len(prior_rewards))
            for j, prior in enumerate(prior_rewards):
                subset = df[(df['block'] == block) & (df['num_rewards_prior_bin'] == prior)]
                x_vals = subset['time_in_port_bin'] + x_jitter[j]  # X-axis: median of time bin
                y_vals = subset[DA_col]  # Y-axis: DA mean
                y_err = subset[DA_sem]  # Error bars: SEM

                # Scatter plot with error bars
                plt.errorbar(x_vals, y_vals, yerr=y_err, fmt='o',
                             color=colors[i], markersize=sizes[j] / 10, capsize=capsizes[j],
                             label=f"Block {block}, {prior_reward_labels[prior]} prior", alpha=0.7)

                # Line connecting points
                plt.plot(x_vals, y_vals, color=colors[i], linewidth=linewidths[j], alpha=0.7)

        custom_xticks = [0, 1, 2, 3]  # Midpoints or bin edges
        custom_xtick_labels = ["0-3", "3-6", "6-9", "9-12"]  # Desired labels
        plt.xticks(custom_xticks, custom_xtick_labels)  # Apply new x-tick labels

        plt.title(title)
        plt.xlabel("Time since Entry (sec)")
        plt.ylabel("DA Amplitude (mean ± SEM)")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True)
        plt.show()

    # Create two separate figures
    plot_DA(df_final, 'right_DA_mean', 'right_DA_sem', f'{ani_summary.animal} Right')
    plot_DA(df_final, 'left_DA_mean', 'left_DA_sem', f'{ani_summary.animal} Left')

    return df_final


def stats_analysis_intervals_vs_DA(ani_summary):
    df_list = []
    for session in ani_summary.session_obj_list:
        if session is not None:
            df_temp = session.DA_vs_NRI_IRI.copy()
            df_list.append(df_temp)
    df_combined = pd.concat(df_list, ignore_index=True)
    return df_combined


# todo: change visualize_NRI_IRI_vs_DA according to the changes made to the DA_vs_NRI_IRI dataframe
# def visualize_NRI_IRI_vs_DA(ani_summary, run_statistics=0, plot_scatters=0, plot_heatmaps=0):
#     df_list = []
#     for session in ani_summary.session_obj_list:
#         if session is not None:
#             df_temp = session.DA_vs_NRI_IRI[
#                 ['NRI', 'IRI', 'DA_right', 'DA_left']].copy()
#             df_list.append(df_temp)
#     df_combined = pd.concat(df_list, ignore_index=True)
#
#     if run_statistics:
#         # helper.run_statistical_analysis(df_combined, 'pearson')
#         # helper.run_statistical_analysis(df_combined, 'spearman')
#         helper.run_statistical_analysis(df_combined, 'linear_regression')
#         helper.run_statistical_analysis(df_combined, 'ANOVA')
#     def plot_DA_NRI_IRI_scatters(df, DA_col, title):
#
#         # Sort dataframe so that lower DA_col values (darker) are plotted first
#         df_sorted = df.sort_values(by=DA_col, ascending=True)
#         df_sorted['NRI'] = df_sorted['NRI'] + np.random.uniform(-0.05, 0.05, size=len(df))
#         df_sorted['IRI'] = df_sorted['IRI'] + np.random.uniform(-0.05, 0.05, size=len(df))
#
#         fig, ax = plt.subplots(figsize=(14, 10))
#         ax.set_facecolor('#888888')  # Medium gray background
#         fig.patch.set_facecolor('#888888')  # Match figure background
#         ax.set_title(title, color='white', fontsize=40)
#         sns.scatterplot(ax=ax, x='NRI', y='IRI', hue=DA_col, data=df_sorted, palette='viridis', s=30, alpha=0.4,
#                         edgecolor=None,
#                         legend=False)
#
#         norm = plt.Normalize(df_sorted[DA_col].quantile(0.1), df_sorted[DA_col].quantile(0.8))
#         sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
#         sm.set_array([])  # Needed for matplotlib colorbar
#         cbar = fig.colorbar(sm, ax=ax)
#         cbar.set_label('DA (in z-score)', color='white')
#         cbar.outline.set_edgecolor('white')
#         cbar.ax.yaxis.set_tick_params(color='white')  # Set tick color
#         plt.setp(cbar.ax.get_yticklabels(), color='white')  # Set tick label color
#
#         ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)
#         ax.xaxis.label.set_color('white')
#         ax.yaxis.label.set_color('white')
#         ax.xaxis.label.set_size(18)
#         ax.yaxis.label.set_size(18)
#         ax.tick_params(axis='both', colors='white', labelsize=14)
#         for spine in ax.spines.values():
#             spine.set_edgecolor('white')
#
#         plt.show()
#
#     if plot_scatters:
#         plot_DA_NRI_IRI_scatters(df_combined, 'DA_right', f"{ani_summary.animal} Right")
#         plot_DA_NRI_IRI_scatters(df_combined, 'DA_left', f"{ani_summary.animal} Left")
#
#     def plot_DA_NRI_IRI_heatmaps(df, DA_col, title):
#         # Define fixed bins (0-2, 2-4, ..., 18-20)
#         x_bins = np.arange(0, 14, 1)  # Bins for NRI
#         y_bins = np.arange(0, 7, 1)  # Bins for IRI
#
#         # Assign each point to a bin (labels are the bin start values)
#         df['x_bin'] = pd.cut(df['NRI'], bins=x_bins, labels=x_bins[:-1])
#         df['y_bin'] = pd.cut(df['IRI'], bins=y_bins, labels=y_bins[:-1])
#
#         # Compute the average value within each bin
#         heatmap_data = df.groupby(['y_bin', 'x_bin'])[DA_col].mean().unstack()
#         for i in range(min(heatmap_data.shape)):
#             heatmap_data.iloc[i, i] = np.nan  # setting all the diagonal elements to nan
#         heatmap_data = heatmap_data[::-1]
#
#         # Convert index and columns to numeric for plotting
#         heatmap_data.index = heatmap_data.index.astype(float)
#         heatmap_data.columns = heatmap_data.columns.astype(float)
#
#         # Normalize the colorbar using min and 70th percentile
#         from matplotlib import colors
#         vmin = df[DA_col].quantile(0.1)
#         vmax = df[DA_col].quantile(0.8)  # 70th percentile
#         norm = colors.Normalize(vmin=vmin, vmax=vmax)
#
#         # Create the figure and heatmap
#         fig, ax = plt.subplots(figsize=(20, 8))
#         sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".1f", linewidths=0.5, norm=norm, cbar=False)
#
#         # Create the colorbar with correct normalization
#         sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
#         sm.set_array([])
#         cbar = fig.colorbar(sm, ax=ax)
#         cbar.set_label('DA (in z-score)', fontsize=14)
#         cbar.ax.tick_params(labelsize=12)
#
#         ax.set_xticks(np.arange(len(x_bins[:-1])))
#         ax.set_yticks(np.arange(len(y_bins[1:-1])) + 1)
#         ax.set_xticklabels(x_bins[:-1])
#         ax.set_yticklabels(y_bins[1:-1][::-1])  # reversed for correct order
#
#         # Label axes
#         plt.xlabel('NRI Bins')
#         plt.ylabel('IRI Bins')
#         plt.title(title)
#
#         plt.show()
#
#     if plot_heatmaps:
#         plot_DA_NRI_IRI_heatmaps(df_combined, 'DA_right', f"{ani_summary.animal} Right")
#         plot_DA_NRI_IRI_heatmaps(df_combined, 'DA_left', f"{ani_summary.animal} Left")

# updates the DA_features dataframe for the animal summary
def visualize_DA_vs_NRI(ani_summary):
    df_list = []
    for session in ani_summary.session_obj_list:
        if session is not None:
            df_temp = session.DA_vs_NRI_IRI[
                ['hemisphere', 'block', 'NRI', 'IRI', 'DA']].copy()
            df_list.append(df_temp)
    df_combined = pd.concat(df_list, ignore_index=True)
    ani_summary.DA_features['animal'] = ani_summary.animal
    ani_summary.DA_features['hemisphere'] = df_combined['hemisphere'].to_numpy()
    ani_summary.DA_features['block'] = df_combined['block'].to_numpy()
    ani_summary.DA_features['NRI'] = df_combined['NRI'].to_numpy()
    ani_summary.DA_features['IRI'] = df_combined['IRI'].to_numpy()
    ani_summary.DA_features['DA'] = df_combined['DA'].to_numpy()


# todo: complete constructing the Animal class
# class Animal:
#     def __init__(self, animal_str, session_list=None, include_branch='both'):
#         self.animal_str = animal_str
#         self.session_obj_list = [None] * len(session_list) if session_list is not None else []
#         if session_list is not None:
#             for i in session_list:
#                 self.session_obj_list[i] = OneSession(animal_str, i, include_branch=include_branch)
#                 self.session_obj_list[i].calculate_dFF0(plot=0, plot_middle_step=0, save=0)
#                 self.session_obj_list[i].process_behavior_data(save=1)


def multi_session_analysis(animal_str, session_list, include_branch='both', port_swap=0):
    animal_dir = os.path.normpath(os.path.join(config.MAIN_DATA_ROOT, animal_str))
    raw_dir = os.path.join(animal_dir, 'raw_data')
    FP_file_list = helper.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = helper.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = helper.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    xsession_figure_export_dir = os.path.join(animal_dir, 'figures')
    # Declaring namedtuple()
    df1 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df2 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df3 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df4 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df5 = pd.DataFrame(
        columns=['animal', 'hemisphere', 'block', 'NRI', 'IRI', 'DA']
    )
    OneAniAllSes = namedtuple('OneAniAllSes',
                              ['animal', 'session_obj_list', 'NRI_amp_ipsi', 'NRI_amp_contra', 'IRI_amp_ipsi',
                               'IRI_amp_contra', 'DA_features'])

    # check if the neural data files, the behavior data files, and the sync data files are of the same numbers
    if (len(FP_file_list) == len(behav_file_list)) & (len(behav_file_list) == len(TTL_file_list)):

        # If so, make a namedtuple,
        # with each including the animal name,
        # the list of objects each being one session,
        # and the summary NRI_amplitude dataframe

        # adding values to a named tuple
        ani_summary = OneAniAllSes(animal=animal_str, session_obj_list=[None] * len(FP_file_list), NRI_amp_ipsi=df1,
                                   NRI_amp_contra=df2, IRI_amp_ipsi=df3, IRI_amp_contra=df4, DA_features=df5)
    else:
        print("Error: the numbers of different data files should be equal!!")
    lick_mod_low = [None] * len(behav_file_list)
    lick_mod_high = [None] * len(behav_file_list)
    DA_block_transition_list = [None] * len(behav_file_list)
    for i in session_list:
        try:
            ani_summary.session_obj_list[i] = OneSession(animal_str, i, include_branch=include_branch,
                                                         port_swap=port_swap)
            # ani_summary.session_obj_list[i].examine_raw(save=0)
            ani_summary.session_obj_list[i].calculate_dFF0(plot=0, plot_middle_step=0, save=0)
            # ani_summary.session_obj_list[i].save_dFF0_and_zscore(format='parquet')
            ani_summary.session_obj_list[i].process_behavior_data(save=0)
            # ani_summary.session_obj_list[i].save_pi_events(format='parquet')
            # ani_summary.session_obj_list[i].construct_trial_df()
            # ani_summary.session_obj_list[i].save_trial_df(format='parquet')
            # ani_summary.session_obj_list[i].construct_expreward_interval_df()
            # ani_summary.session_obj_list[i].save_expreward_df(format='parquet')
            # DA_block_transition_list[i] = ani_summary.session_obj_list[i].bg_port_in_block_reversal(plot_single_traes=0,
            #                                                                                         plot_average=0)
            # ani_summary.session_obj_list[i].extract_bg_behav_by_trial()
            # lick_rates, (lick_mod_low[i], lick_mod_high[i]) = ani_summary.session_obj_list[i].calculate_lick_rates_around_bg_reward()
            # ani_summary.session_obj_list[i].plot_bg_heatmaps(save=1)
            # ani_summary.session_obj_list[i].plot_heatmaps(save=1)
            # ani_summary.session_obj_list[i].actual_leave_vs_adjusted_optimal(save=0)
            # ani_summary.session_obj_list[i].extract_transient(plot_zscore=0)
            # ani_summary.session_obj_list[i].visualize_correlation_scatter(save=0)
            # ani_summary.session_obj_list[i].extract_reward_features_and_DA(save_dataframe=0)
            ani_summary.session_obj_list[i].visualize_DA_vs_NRI_IRI()
            ani_summary.session_obj_list[i].save_DA_vs_features(format='parquet')
            # ani_summary.session_obj_list[i].visualize_average_traces(variable='time_in_port', method='even_time',
            #                                                          block_split=False,
            #                                                          plot_linecharts=0,
            #                                                          plot_histograms=0)
            # ani_summary.session_obj_list[i].extract_binned_da_vs_reward_history_matrix(binsize=0.1, save=1)

        except:
            print(f"skipped session {i} because of error!!!")
            print("----------------------------------")
            continue
    # ses_selected_l, ses_selected_r = visualize_transient_consistency(session_obj_list, save=1, save_path=xsession_figure_export_dir)
    # visualize_NRI_vs_amplitude_families(ani_summary, variable='time_in_port', block_split=True, normalized=False, save=False,
    #                                     save_path=None)
    # df_DA_NRI_rewardhistory = visualize_NRI_reward_history_vs_DA(ani_summary)
    # visualize_NRI_IRI_vs_DA(ani_summary, run_statistics=1,plot_heatmaps=0, plot_scatters=0)
    # visualize_DA_vs_NRI(ani_summary)
    # stats_df = stats_analysis_intervals_vs_DA(ani_summary)

    # # plot lick modulation progression
    # lick_mod_low = [item for item in lick_mod_low if item is not None]
    # lick_mod_high = [item for item in lick_mod_high if item is not None]
    # fig, ax = plt.subplots(figsize=(5, 2))
    # palette = sns.color_palette('Set2')
    # ax.plot(lick_mod_low, marker='o', color=palette[0], label='Low context reward rate')
    # ax.plot(lick_mod_high, marker='o', color=palette[1], label='High context reward rate')
    # ax.set_xlabel('Session number')
    # ax.set_title(f'{animal_str}: Anticipatory Lick Modulation')
    # fig.show()
    return ani_summary


if __name__ == '__main__':
    # multi
    # DAresponse = np.zeros(90)

    # session_list = [0, 2, 4, 6, 8, 10, 13, 15, 17, 20, 21, 22, 23, 24]
    # summary_RK1 = multi_session_analysis('RK001', session_list, include_branch='both', port_swap=0)
    #
    # session_list = [1, 2, 3, 4, 5, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25]
    # summary_RK3 = multi_session_analysis('RK003', session_list, include_branch='both', port_swap=0)
    #
    # session_list = [3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
    # summary_RK5 = multi_session_analysis('RK005', session_list, include_branch='both', port_swap=0)
    #
    # session_list = [0, 1, 2, 4, 11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24]
    # summary_RK6 = multi_session_analysis('RK006', session_list, include_branch='both', port_swap=1)

    # session_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # summary_RK9 = multi_session_analysis('RK009', session_list, include_branch='both', port_swap=1)

    session_list = [1, 2, 3, 5, 7, 9, 11, 12, 14, 15, 19, 22, 23, 24, 25]
    summary_036 = multi_session_analysis('SZ036', session_list, include_branch='both')

    session_list = [0, 1, 2, 4, 5, 6, 8, 9, 11, 15, 16, 17, 18, 19, 20, 22, 24, 25, 27, 28, 29, 31, 32, 33, 35]
    summary_037 = multi_session_analysis('SZ037', session_list, include_branch='both')

    session_list = [1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32,
                    33, 34, 35, 36, 37, 38]
    summary_038 = multi_session_analysis('SZ038', session_list, include_branch='both')

    session_list = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22]
    summary_039 = multi_session_analysis('SZ039', session_list, include_branch='only_left')

    # session_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]
    # summary_041 = multi_session_analysis('SZ041', session_list, include_branch='only_right')
    session_list = [3, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 21, 22, 24, 26, 27, 28, 30]
    summary_042 = multi_session_analysis('SZ042', session_list, include_branch='only_left')

    session_list = [0, 1, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23]
    summary_043 = multi_session_analysis('SZ043', session_list, include_branch='only_right')


    # plot the 4-point line chart DA vs. NRI for each animal
    def get_mean_sem_DA_for_feature(df, var='NRI', sample_per_bin=250):
        df_sorted = df.sort_values(by=var).reset_index(drop=True)
        total_sample = len(df_sorted)
        bin_num = int(total_sample / sample_per_bin)
        arr_bin_median = np.zeros(bin_num)
        arr_mean = np.zeros(bin_num)
        arr_sem = np.zeros(bin_num)
        for i in range(bin_num):
            # if var == 'NRI':
            #     lower_bound = 3 * i
            #     higher_bound = 3 * (i + 1)
            # if var == 'IRI':
            #     lower_bound = 1.5 * i
            #     higher_bound = 1.5 * (i + 1)
            arr_bin_median[i] = (df_sorted.iloc[sample_per_bin * i][var] + df_sorted.iloc[sample_per_bin * (i + 1) - 1][
                var]) / 2
            arr_mean[i] = df_sorted.iloc[sample_per_bin * i:sample_per_bin * (i + 1)]['DA'].mean()
            arr_sem[i] = df_sorted.iloc[sample_per_bin * i:sample_per_bin * (i + 1)]['DA'].sem()
        mean_sem_df = pd.DataFrame({'bin_center': arr_bin_median, 'mean': arr_mean, 'sem': arr_sem})
        return mean_sem_df


    # This is the simple exponential decreasing
    def exp_decreasing(x, cumulative=8., starting=1.):
        a = starting
        b = a / cumulative
        density = a / np.exp(b * x)
        return density


    # Plot DA vs NRI/IRI for each animal
    df1 = summary_036.DA_features
    df2 = summary_037.DA_features
    df3 = summary_038.DA_features
    df4 = summary_039.DA_features[summary_039.DA_features['hemisphere'] == 'left'].reset_index(drop=True)
    df5 = summary_042.DA_features[summary_042.DA_features['hemisphere'] == 'left'].reset_index(drop=True)
    df6 = summary_043.DA_features[summary_043.DA_features['hemisphere'] == 'right'].reset_index(drop=True)
    # xticks = np.array([1, 3, 5, 7, 9, 11])
    # xticks = np.array([1, 2, 3, 4, 5])
    xticks = np.array([1, 3, 5, 7, 9, 11])
    fig, ax = plt.subplots()
    for (df, name) in [(df1, 'SZ036'), (df2, 'SZ037'), (df3, 'SZ038'), (df4, 'SZ039'), (df5, 'SZ042'), (df6, 'SZ043')]:
        # df = df[(df['IRI'] > 1) & (df['IRI'] < df['NRI'])]
        df = df[(df['IRI'] > 1)]
        to_plot_df = get_mean_sem_DA_for_feature(df, var='NRI', sample_per_bin=400)
        x = to_plot_df['bin_center']
        y = to_plot_df['mean']
        y_err = to_plot_df['sem']
        ax.plot(x, y, label=name)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
        # ax.errorbar(to_plot_df['bin_center'], to_plot_df['mean'], yerr=to_plot_df['sem'], fmt='o-',capsize=5, label=name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('DA (z-score)', fontsize=15)
    plt.xlabel('Reward Time since Entry (sec)', fontsize=15)
    plt.xticks(xticks, fontsize=15)
    ax2 = ax.twinx()
    x = np.arange(0, 12, 0.1)
    y = exp_decreasing(x, cumulative=8., starting=1.)
    ax2.plot(x, y / 10, color='gray', linestyle='--', linewidth=1.5, label='Reward Probability')
    ax2.set_zorder(0)
    ax2.patch.set_visible(False)
    ax2.set_ylim(0, 0.12)
    plt.yticks(fontsize=15)
    plt.title('DA vs NRI (sample/bin=400)', fontsize=15)
    plt.legend()
    plt.show()

    # Divided by Block
    df_all = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    xticks = np.array([1, 3, 5, 7, 9, 11])
    df_all = df_all[(df_all['IRI'] > 1) & (df_all['IRI'] < df_all['NRI'])]
    low_df = get_mean_sem_DA_for_feature(df_all[df_all['block'] == '0.4'], var='NRI', sample_per_bin=700)
    high_df = get_mean_sem_DA_for_feature(df_all[df_all['block'] == '0.8'], var='NRI', sample_per_bin=700)
    fig, ax = plt.subplots()
    cpalette = sns.color_palette('Set2')
    x = low_df['bin_center']
    y = low_df['mean']
    y_err = low_df['sem']
    ax.plot(x, y, label='Low Context', color=cpalette[0])
    ax.fill_between(x, y - y_err, y + y_err, color=cpalette[0], alpha=0.2)
    x = high_df['bin_center']
    y = high_df['mean']
    y_err = high_df['sem']
    ax.plot(x, y, label='High Context', color=cpalette[1])
    ax.fill_between(x, y - y_err, y + y_err, color=cpalette[1], alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Reward Time since Port Entry (sec)', fontsize=15)
    plt.ylabel('DA (z-score)', fontsize=15)
    plt.xticks(xticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Block Effect of DA (sample/bin=700)', fontsize=15)
    plt.legend()
    plt.show()

    # Divided by Hemisphere
    df_all = pd.concat([df1, df2, df3])
    contra_df = get_mean_sem_DA_for_feature(df_all[df_all['hemisphere'] == 'left'], var='NRI', sample_per_bin=600)
    ipsi_df = get_mean_sem_DA_for_feature(df_all[df_all['hemisphere'] == 'right'], var='NRI', sample_per_bin=600)
    xticks = [1, 3, 5, 7, 9, 11]
    fig, ax = plt.subplots()
    cpalette = sns.color_palette('Set1')
    x = contra_df['bin_center']
    y = contra_df['mean']
    y_err = contra_df['sem']
    ax.plot(x, y, label='Contralateral', color=cpalette[0])
    ax.fill_between(x, y - y_err, y + y_err, color=cpalette[0], alpha=0.2)
    x = ipsi_df['bin_center']
    y = ipsi_df['mean']
    y_err = ipsi_df['sem']
    ax.plot(x, y, label='Ipsilateral', color=cpalette[1])
    ax.fill_between(x, y - y_err, y + y_err, color=cpalette[1], alpha=0.2)
    # ax.errorbar(x, contra_df['mean'], yerr=contra_df['sem'], fmt='o-', capsize=5, label='contralateral', color=cpalette[0])
    # ax.errorbar(x, ipsi_df['mean'], yerr=ipsi_df['sem'], fmt='o-', capsize=5, label='ipsilateral', color=cpalette[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Reward Time since Port Entry (sec)', fontsize=15)
    plt.ylabel('DA (z-score)', fontsize=15)
    plt.xticks(xticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Hemisphere Effect of DA (sample/bin=600)', fontsize=15)
    plt.legend()
    plt.show()

    # mixed linear model regression and power analysis
    df = pd.concat([stats036, stats037, stats038, stats039[stats039['hemisphere'] == 'left'],
                    stats042[stats042['hemisphere'] == 'left'], stats043[stats043['hemisphere'] == 'right']],
                   ignore_index=True)
    # Step 1: Within-subject Linear Regression
    import statsmodels.api as sm

    # Dictionary to store regression results
    results = {}
    # Grouping by (animal, hemisphere)
    for (animal, hemisphere), group in df.groupby(["animal", "hemisphere"]):
        X = group[["NRI", "IRI"]]  # Predictor variables
        X["NRI_IRI"] = X["NRI"] * X["IRI"]  # Interaction term
        X = sm.add_constant(X)  # Adds intercept
        y = group["DA"]  # Dependent variable
        model = sm.OLS(y, X).fit()
        results[(animal, hemisphere)] = model.params  # Store regression coefficients
    # Convert results into a DataFrame for analysis
    beta_df = pd.DataFrame.from_dict(results, orient="index", columns=["Intercept", "NRI", "IRI", "NRI_IRI"])
    beta_df.index.names = ["Animal", "Hemisphere"]
    print(beta_df)
    # Step 2: Mixed-Effects Model
    import statsmodels.formula.api as smf

    df["subject"] = df[["animal", "hemisphere"]].apply(tuple, axis=1)  # Create subject ID
    df["NRI_IRI"] = df["NRI"] * df["IRI"]  # Interaction term
    model = smf.mixedlm("DA ~ NRI + IRI + NRI_IRI", df, groups=df["subject"]).fit()
    print(model.summary())
    # Step 3: Power Analysis
    from statsmodels.stats.power import TTestPower

    for predictor in ["NRI", "IRI", "NRI_IRI"]:
        effect_size = abs(beta_df[predictor].mean()) / beta_df[predictor].std()
        power_analysis = TTestPower()
        power = power_analysis.power(effect_size=effect_size, nobs=len(beta_df), alpha=0.05)
        print(f"Power for {predictor}: {power}")
    # 1-way ANOVA with repeated measures, the within-subject factor being time invested in exp port before rewarded
    # DAresponse[0:5] = summary_036.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[5:10] = summary_036.NRI_amp_ipsi.mean_amp.to_numpy()
    # DAresponse[10:15] = summary_037.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[15:20] = summary_037.NRI_amp_ipsi.mean_amp.to_numpy()
    # DAresponse[20:25] = summary_038.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[25:30] = summary_038.NRI_amp_ipsi.mean_amp.to_numpy()
    # DAresponse[30:35] = summary_039.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[35:40] = summary_042.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[40:45] = summary_043.NRI_amp_ipsi.mean_amp.to_numpy()
    # subject_np = np.repeat(
    #     ['036_contra', '036_ipsi', '037_contra', '037_ipsi', '038_contra', '038_ipsi', '039_contra', '042_contra',
    #      '043_ipsi'], 5)
    # time_invested = [1, 3, 5, 7, 9] * 9
    # df = pd.DataFrame({
    #     'Subject': subject_np,
    #     'TimeInvested': time_invested,
    #     'DAresponse': DAresponse
    # })
    # aov_1w = pg.rm_anova(dv='DAresponse', within='TimeInvested', subject='Subject', data=df, detailed=True)

    # 2-way ANOVA with repeated measures
    DAresponse[0:5] = summary_036.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[5:10] = summary_036.IRI_amp_ipsi.mean_amp_low.to_numpy()
    DAresponse[10:15] = summary_037.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[15:20] = summary_037.IRI_amp_ipsi.mean_amp_low.to_numpy()
    DAresponse[20:25] = summary_038.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[25:30] = summary_038.IRI_amp_ipsi.mean_amp_low.to_numpy()
    DAresponse[30:35] = summary_039.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[35:40] = summary_042.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[40:45] = summary_043.IRI_amp_ipsi.mean_amp_low.to_numpy()

    DAresponse[45:50] = summary_036.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[50:55] = summary_036.IRI_amp_ipsi.mean_amp_high.to_numpy()
    DAresponse[55:60] = summary_037.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[60:65] = summary_037.IRI_amp_ipsi.mean_amp_high.to_numpy()
    DAresponse[65:70] = summary_038.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[70:75] = summary_038.IRI_amp_ipsi.mean_amp_high.to_numpy()
    DAresponse[75:80] = summary_039.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[80:85] = summary_042.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[85:90] = summary_043.IRI_amp_ipsi.mean_amp_high.to_numpy()

    subject_np = np.tile(np.repeat(
        ['036_contra', '036_ipsi', '037_contra', '037_ipsi', '038_contra', '038_ipsi', '039_contra', '042_contra',
         '043_ipsi'], 5), 2)
    time_invested = [1, 3, 5, 7, 9] * 18
    block = np.repeat(['low', 'high'], 45)
    df = pd.DataFrame({
        'Subject': subject_np,
        'TimeInvested': time_invested,
        'Block': block,
        'DAresponse': DAresponse
    })
    aov_2w = pg.rm_anova(dv='DAresponse', within=['TimeInvested', 'Block'], subject='Subject', data=df, detailed=True)
    fig = plt.figure()
    ani_sum = summary_036
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='036 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='036 contra')
    ani_sum = summary_037
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='037 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='037 contra')
    ani_sum = summary_038
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='038 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='038 contra')
    ani_sum = summary_039
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='039 contra')
    # ani_sum = summary_041
    # plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
    #              yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='041 ipsi')
    # plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
    #              yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='041 contra')
    ani_sum = summary_042
    # plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
    #              yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='042 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='042 contra')
    ani_sum = summary_043
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='043 ipsi')
    # plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
    #              yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='043 contra')
    fig.legend()
    fig.show()
    print('Hello')
    # session_list = list(range(15))
    # session_obj_list = multi_session_analysis('SZ050', session_list, include_branch='both')
    # session_list = list(range(16))
    # session_obj_list = multi_session_analysis('SZ051', session_list, include_branch='both')
    # session_list = list(range(15))
    # session_obj_list = multi_session_analysis('SZ052', session_list, include_branch='both')
    # session_list = list(range(21))
    # session_obj_list = multi_session_analysis('SZ055', session_list, include_branch='both')

    # single
    # session_list = list(range(16))
    # session_obj_list = multi_session_analysis('SZ044', session_list, include_branch='both')
    # session_list = list(range(12))
    # session_obj_list = multi_session_analysis('SZ045', session_list, include_branch='only_left')
    # session_list = list(range(17))
    # session_obj_list = multi_session_analysis('SZ046', session_list, include_branch='both')
    # session_list = list(range(24))
    # session_obj_list = multi_session_analysis('SZ047', session_list, include_branch='both')
    # session_list = list(range(21))
    # session_obj_list = multi_session_analysis('SZ048', session_list, include_branch='both')
    # session_list = list(range(17))
    # session_obj_list = multi_session_analysis('SZ053', session_list, include_branch='both')
    # session_list = list(range(17))
    # session_obj_list = multi_session_analysis('SZ054', session_list, include_branch='both')
    # session_list = list(range(20))
    # session_obj_list = multi_session_analysis('SZ058', session_list, include_branch='both')
    # session_list = list(range(19))
    # session_obj_list = multi_session_analysis('SZ059', session_list, include_branch='both')
