import numpy as np
import time
import data_loader
import helper
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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
    ax.set_xlim([-1, 21])
    ax.set_xticks(np.arange(0, 20, 2.5))
    ax.set_xlabel('Time since Trial Starts (sec)')
    ax.set_ylabel('DA (z-score)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(f'Example trial: {example_trial_id}')
    plt.tight_layout()
    if return_handle:
        fig.show()
        return fig, ax


def figd_example_session_heatmaps(zscore, reward_df, axes=None):
    if axes is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        return_handle = True
    else:
        fig = None
        return_handle = False
    valid_df = reward_df[
        reward_df['time_in_port'].notna() & (reward_df['IRI_prior'] > 1) & (reward_df['IRI_post'] >= 0.6)]
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
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        if return_handle:
            fig.show()
            return fig, ax

    nodes = [0.0, 0.5, 0.75, 1.0]
    colors = ["blue", "black", "red", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("my_cmap", list(zip(nodes, colors)))

    heatmap_matrix = heatmap_matrix[np.sum(~np.isnan(heatmap_matrix), axis=1) > 10]

    # Use seaborn.heatmap which handles NaNs gracefully.
    sns.heatmap(
        heatmap_matrix,
        ax=ax,
        cmap=custom_cmap,
        center=0,
        cbar=True,
        yticklabels=False
    )

    # adjust the colorbar tick labels
    cbar_ax = fig.axes[-1]
    # cbar_ax.yaxis.label.set_size(4)
    cbar_ax.tick_params(width=0.2, length=0.5, labelsize=4)

    # Customize the x-axis ticks to show time in seconds
    xtick_positions = np.linspace(0, heatmap_matrix.shape[1], 6)
    xtick_labels = np.round(np.linspace(cutoff_pre_reward, cutoff_post_reward, 6), 1)
    ytick_positions = np.arange(0, heatmap_matrix.shape[0], 30)
    ytick_labels = np.arange(0, heatmap_matrix.shape[0], 30)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=4, rotation=0)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=4)
    ax.tick_params(axis='x', width=0.2, length=0.5)
    ax.tick_params(axis='y', width=0.2, length=0.5)
    ax.set_xlabel('Time from Reward (s)', fontsize=4)
    ax.set_ylabel('Trial', fontsize=4)
    plt.subplots_adjust(left=0, right=0.01, bottom=0, top=0.01)
    plt.tight_layout()
    if return_handle:
        fig.show()
        return fig, ax


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
    session_id = 9
    session_name = '2024-01-08T13_52'
    zscore = data_loader.load_session_dataframe(animal_str, 'zscore', session_long_name=session_name,
                                                file_format='parquet')
    trial_df = data_loader.load_session_dataframe(animal_str, 'trial_df', session_long_name=session_name,
                                                  file_format='parquet')
    reward_df = data_loader.load_session_dataframe(animal_str, 'expreward_df', session_long_name=session_name,
                                                   file_format='parquet')
    # --- end of data preparation ---
    figd_example_session_heatmaps(zscore, reward_df, axes=None)
    figc_example_trial_1d_traces(zscore, trial_df, example_trial_id=32, ax=None)

    # --- go through all the trials in one session when looking for good trials ---
    # for trial_id in trial_df['trial'].values:
    #     figc_example_1d_traces(zscore, trial_df, example_trial_id=trial_id, ax=None)
    #     time.sleep(2) # Pause for 2 seconds between each plot to respect the server's rate limit
    # --- end ---


if __name__ == '__main__':
    main()
