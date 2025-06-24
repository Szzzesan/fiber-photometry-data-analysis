from matplotlib import gridspec
from matplotlib.transforms import ScaledTranslation

import data_loader
import helper

import numpy as np
import pandas as pd
import time

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

mpl.rcParams['figure.dpi'] = 300


# --- First plotting method starts here ---
def figc_example_trial_1d_traces(zscore, trial_df, example_trial_id, ax=None):
    assert example_trial_id in trial_df[
        'trial'].values, f"example_trial_id {example_trial_id} not found in trial_df['trial']"
    row_id_bool = trial_df['trial'] == example_trial_id
    snippet_begin = trial_df.loc[row_id_bool, 'bg_entry'].values[0]
    snippet_end = trial_df.loc[row_id_bool, 'exp_exit'].values[0]
    timestamps_to_plot = zscore.loc[
        (zscore['time_recording'] > snippet_begin) & (zscore['time_recording'] < snippet_end), 'time_recording'].values
    DA_trace_to_plot = zscore.loc[
        (zscore['time_recording'] > snippet_begin) & (zscore['time_recording'] < snippet_end), 'green_left'].values

    bg_entry = trial_df.loc[row_id_bool, 'bg_entry'].values[0]
    bg_exit = trial_df.loc[row_id_bool, 'bg_exit'].values[0] - bg_entry
    exp_entry = trial_df.loc[row_id_bool, 'exp_entry'].values[0] - bg_entry
    exp_exit = trial_df.loc[row_id_bool, 'exp_exit'].values[0] - bg_entry
    entries = [0, exp_entry]
    exits = [bg_exit, exp_exit]
    rewards = trial_df.loc[row_id_bool, 'rewards'].values[0] - bg_entry
    licks = trial_df.loc[row_id_bool, 'licks'].values[0] - bg_entry
    relative_time = timestamps_to_plot - bg_entry
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2))
        return_handle = True
        ax.set_title(f'Example trial: {example_trial_id}')
    else:
        fig = None
        return_handle = False
    ax.axvspan(0, bg_exit, ymin=0.9, ymax=1, color='skyblue', alpha=0.6, label='Context Port')
    ax.axvspan(exp_entry, exp_exit, ymin=0.9, ymax=1, color='lightcoral', alpha=0.6, label='Investment Port')
    draw_vertical_lines(ax, licks, ymin=0.9, ymax=1, color='grey', alpha=0.1)
    draw_vertical_lines(ax, entries, color='b', linestyle='--')
    draw_vertical_lines(ax, exits, color='g', linestyle='--')
    draw_vertical_lines(ax, rewards, color='r')
    ax.plot(relative_time, DA_trace_to_plot, color='black')
    ax.set_xlim([-1, 15])
    ax.set_xticks(np.arange(0, 15.5, 2.5))
    ax.set_xlabel('Time since Trial Starts (sec)')
    ax.set_ylabel('DA (z-score)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    if return_handle:
        fig.show()
        return fig, ax


def figd_example_session_heatmaps(zscore, reward_df, axes=None):
    if axes is None:
        fig = plt.figure(figsize=(2, 2))
        gs = GridSpec(1, 3, width_ratios=[0.5, 20, 1], wspace=0.05)
        ax_bars = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[0, 1])
        ax_cbar = fig.add_subplot(gs[0, 2])
        return_handle = True
    else:
        fig = None
        return_handle = False
        ax_bars = axes[0]
        ax_heatmap = axes[1]
        ax_cbar = axes[2]
    valid_df = reward_df[
        reward_df['time_in_port'].notna() & (reward_df['IRI_prior'] > 1) & (reward_df['IRI_post'] >= 0.6)]
    # Categorize 'time_in_port' for the bar plot
    bins = [0, 3, 6, 9, 12, np.inf]
    # Using labels=False gives integer codes (0, 1, 2, 3, 4) perfect for plotting
    valid_df['time_cat_code'] = pd.cut(valid_df['time_in_port'], bins=bins, labels=False, right=False)

    sorted_df = valid_df.sort_values(by='time_in_port').reset_index(drop=True)
    bin_size_s = 0.05

    original_timestamps = zscore['time_recording'].values
    original_zscore = zscore['green_left'].values
    max_duration = sorted_df['IRI_post'].max()
    cutoff_pre_reward = -0.5
    cutoff_post_reward = 2
    uniform_time_vector = np.arange(cutoff_pre_reward, cutoff_post_reward, bin_size_s)
    heatmap_matrix = np.full((len(sorted_df), len(uniform_time_vector)), np.nan)
    # --- 3. Resample Each Trial and Place onto the Canvas ---
    for idx, row in sorted_df.iterrows():
        start_time = row['reward_time'] + cutoff_pre_reward
        if pd.isna(row['next_reward_time']):
            end_time = row['exit_time']
        else:
            end_time = row['next_reward_time']
        duration = end_time - start_time
        if duration < 0:
            continue
        target_timestamps_absolute = np.arange(start_time, end_time, bin_size_s)
        search_start_idx = helper.find_closest_value(original_timestamps, start_time)
        search_end_idx = helper.find_closest_value(original_timestamps, end_time)
        search_start_idx = max(0, search_start_idx)
        search_end_idx = min(len(original_timestamps), search_end_idx)
        source_timestamps = original_timestamps[search_start_idx:search_end_idx]
        source_zscores = original_zscore[search_start_idx:search_end_idx]
        if source_timestamps.size == 0:
            print(f"Source timestamps empty: {start_time}, {end_time}")
            continue
        resampled_zscores = np.interp(target_timestamps_absolute, source_timestamps, source_zscores)
        if len(resampled_zscores) < len(uniform_time_vector):
            heatmap_matrix[idx, :len(resampled_zscores)] = resampled_zscores
        else:
            heatmap_matrix[idx, :] = resampled_zscores[:len(uniform_time_vector)]
    # --- 4. Plotting ---
    if np.all(np.isnan(heatmap_matrix)):
        print("No valid epochs were found to plot.")
        ax_heatmap.text(0.5, 0.5, 'No Data', ha='center', va='center')
        if return_handle:
            fig.show()
            return fig, ax_heatmap

    nodes = [0.0, 0.5, 0.75, 1.0]
    colors = ["blue", "black", "red", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("my_cmap", list(zip(nodes, colors)))

    # heatmap_matrix = heatmap_matrix[np.sum(~np.isnan(heatmap_matrix), axis=1) > 10]
    category_codes = sorted_df['time_cat_code'].values

    # Plot the category bars on the left axis (ax_bars)
    category_matrix = category_codes[:, np.newaxis]  # Reshape for heatmap
    # Use 'Reds_r' to go from dark red (low NRIs) to light red (high NRIs)
    sns.heatmap(category_matrix, ax=ax_bars, cmap='Reds_r', cbar=False, xticklabels=False, yticklabels=False)
    ax_bars.set_ylabel(None)  # Remove any potential labels

    # Plot main heatmap
    sns.heatmap(
        heatmap_matrix,
        ax=ax_heatmap,
        cmap=custom_cmap,
        center=0,
        cbar=True,
        yticklabels=False,
        cbar_ax=ax_cbar,
    )

    # ax_cbar.tick_params(width=0.3, length=0.8, labelsize=4, direction='in', color='white')
    ax_cbar.tick_params(direction='in', color='white')

    # Customize the x-axis ticks to show time in seconds
    xtick_positions = np.linspace(0, heatmap_matrix.shape[1], 6)
    xtick_labels = np.round(np.linspace(cutoff_pre_reward, cutoff_post_reward, 6), 1)
    ax_heatmap.set_xticks(xtick_positions)
    ax_heatmap.set_xticklabels(xtick_labels, rotation=0)
    ax_heatmap.set_xlabel('Time from Reward (s)')
    # ytick_positions = np.arange(0, heatmap_matrix.shape[0], 30)
    # ytick_labels = np.arange(0, heatmap_matrix.shape[0], 30)
    # ax_heatmap.set_yticks(ytick_positions)
    # ax_heatmap.set_yticklabels(ytick_labels, fontsize=4)
    # ax_heatmap.tick_params(axis='y', width=0.2, length=0.5)
    # ax_heatmap.set_ylabel('Trial', fontsize=4)
    ax_heatmap.set_ylabel(None)
    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, ax_heatmap


def setup_axes():
    fig_size = (12, 12)  # (width, height) in inches
    rows, cols = fig_size[1] * 10, fig_size[0] * 10

    row_1 = [8, 1, 2]
    row_2 = [1.5, 18, 2, 18, 1.5, 18, 2, 18]
    row_3 = [2, 2, 1]
    row_4 = [3, 2]
    col_1 = [1, 2, 2, 3]

    row_1_margins = [4, 6]
    row_2_margins = [1, 1, 4, 6, 1, 1, 4]
    row_3_margins = [6, 6]
    row_4_margins = [10]
    col_1_margins = [8, 8, 8]

    row_1_splits = [int((cols - np.sum(row_1_margins)) * h / sum(row_1)) for h in row_1]
    row_2_splits = [int((cols - np.sum(row_2_margins)) * h / sum(row_2)) for h in row_2]
    row_3_splits = [int((cols - np.sum(row_3_margins)) * h / sum(row_3)) for h in row_3]
    row_4_splits = [int((cols - np.sum(row_4_margins)) * h / sum(row_4)) for h in row_4]
    col_1_splits = [int((rows - np.sum(col_1_margins)) * h / sum(col_1)) for h in col_1]

    row_1_splits = [val for pair in zip(row_1_splits, row_1_margins + [0]) for val in pair][:-1]
    row_2_splits = [val for pair in zip(row_2_splits, row_2_margins + [0]) for val in pair][:-1]
    row_3_splits = [val for pair in zip(row_3_splits, row_3_margins + [0]) for val in pair][:-1]
    row_4_splits = [val for pair in zip(row_4_splits, row_4_margins + [0]) for val in pair][:-1]
    col_1_splits = [val for pair in zip(col_1_splits, col_1_margins + [0]) for val in pair][:-1]

    row_1_splits = np.cumsum(row_1_splits)
    row_2_splits = np.cumsum(row_2_splits)
    row_3_splits = np.cumsum(row_3_splits)
    row_4_splits = np.cumsum(row_4_splits)
    col_1_splits = np.cumsum(col_1_splits)

    row_1_splits[-1] = cols
    row_2_splits[-1] = cols
    row_3_splits[-1] = cols
    row_4_splits[-1] = cols
    col_1_splits += rows - col_1_splits[-1]

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    axes = [
        fig.add_subplot(gs[1:col_1_splits[0],
                        :row_1_splits[0]]),  # example trial plot
        fig.add_subplot(gs[1:col_1_splits[0],
                        row_1_splits[1]:row_1_splits[2]]),  # example trial legend
        fig.add_subplot(gs[:col_1_splits[0],
                        row_1_splits[-2]:]),  # recording locations plot

        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        :row_2_splits[0]]),  # category bar for heatmap 1
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[1]:row_2_splits[2]]),  # heatmap 1
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[3]:row_2_splits[4]]),  # colorbar for heatmap 1
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[5]:row_2_splits[6]]),  # transient averaged from heatmap 1

        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[7]:row_2_splits[8]]),  # category bar for heatmap 2
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[9]:row_2_splits[10]]),  # heatmap 2
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[11]:row_2_splits[12]]),  # colorbar for heatmap 2
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2],
                        row_2_splits[13]:row_2_splits[14]]),  # transient averaged from heatmap 2

        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4],
                        :row_3_splits[0]]),  # DA vs. NRI for all animals
        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4],
                        row_3_splits[1]:row_3_splits[2]]),  # DA vs. NRI but split by blocks from all animals
        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4],
                        row_3_splits[3]:row_3_splits[4]]),  # DA vs. IRI

        fig.add_subplot(gs[col_1_splits[5]:col_1_splits[6],
                        :row_4_splits[0]]),  # heatmap of DA vs. NRI vs. IRI and split by block
        fig.add_subplot(gs[col_1_splits[5]:col_1_splits[6],
                        row_4_splits[1]:row_4_splits[2]]),  # scatters of predicted DA vs. real DA with fitting

    ]

    # # remove right and top spines
    # for ax in axes:
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    axes_to_letter = [0, 2, 3, 7, 11, 12, 13, 14, 15]
    for i, ax in enumerate(axes_to_letter):
        axes[ax].text(
            0.0, 1.0, lettering[i], transform=(
                    axes[ax].transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif', weight='bold')

    axes = np.array(axes)
    return axes


# --- helper functions ---
def draw_vertical_lines(ax, x_npy, ymin=0, ymax=1, color='r', alpha=1, linestyle='-', linewidth=1):
    for x_value in x_npy:
        ax.axvline(x_value, ymin=ymin, ymax=ymax, color=color, linestyle=linestyle, linewidth=linewidth)


# --- end of helper function
def main():
    # --- good example trials that show DA correlated with NRI ---
    # animal SZ036
    # '2024-01-03T16_13': trial 11
    # '2024-01-04T11_40': trial 32
    # '2024-01-08T13_52': trial 32 (this one looks amazing), 42, 47
    # '2024-01-11T16_25': trial 9
    # '2024-01-12T10_58': trial 6
    # '2024-01-13T20_50': trial 38, 39, 43
    # --- example trial log ends ---

    # --- data preparation ---
    animal_str = 'SZ036'
    session_name = '2024-01-08T13_52'
    zscore_example_trial = data_loader.load_session_dataframe(animal_str, 'zscore',
                                                              session_long_name=session_name,
                                                              file_format='parquet')
    trial_df = data_loader.load_session_dataframe(animal_str, 'trial_df', session_long_name=session_name,
                                                  file_format='parquet')

    animal_str = 'SZ036'
    session_name = '2024-01-04T15_49'
    zscore_heatmap = data_loader.load_session_dataframe(animal_str, 'zscore', session_long_name=session_name,
                                                        file_format='parquet')
    reward_df = data_loader.load_session_dataframe(animal_str, 'expreward_df', session_long_name=session_name,
                                                   file_format='parquet')
    # --- end of data preparation ---

    # --- set up axes and add figures to axes ---
    axes = setup_axes()

    tic = time.time()
    figc_example_trial_1d_traces(zscore_example_trial, trial_df, example_trial_id=32, ax=axes[0])
    print(f'figc_example_trial took {time.time() - tic:.2f} seconds')

    tic = time.time()
    figd_example_session_heatmaps(zscore_heatmap, reward_df, axes=axes[3:6])
    print(f'figd_example_session_heatmaps took {time.time() - tic:.2f} seconds')

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
    print('hello')
    # --- end of adding figures to axes



    # --- go through all the trials in one session when looking for good trials ---
    # for trial_id in trial_df['trial'].values:
    #     figc_example_1d_traces(zscore, trial_df, example_trial_id=trial_id, ax=None)
    #     time.sleep(2) # Pause for 2 seconds between each plot to respect the server's rate limit
    # --- end ---


if __name__ == '__main__':
    main()
