from matplotlib import gridspec
from matplotlib.transforms import ScaledTranslation

import data_loader
import helper

import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.stats import t

import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cmasher

mpl.rcParams['figure.dpi'] = 300


# # Using a dictionary is a clean way to update multiple parameters
# params = {
#     'axes.labelsize': 9,
#     'axes.labelweight': 'semibold',  # Set x and y labels to bold
#     # 'axes.titlesize': 10,
#     # 'axes.titleweight': 'bold',  # Set plot titles to bold
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'font.weight': 'semibold',       # Set other font elements like tick labels to bold
#     'legend.fontsize': 7,
# }
# plt.rcParams.update(params)

# --- First plotting method starts here ---
def figa_example_trial_1d_traces(zscore, trial_df, example_trial_id, ax=None):
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
    ax.axvspan(0, bg_exit, ymin=0.89, ymax=1, facecolor='skyblue', alpha=0.6, edgecolor='none', label='Context Port')
    ax.axvspan(exp_entry, exp_exit, ymin=0.89, ymax=1, facecolor='lightcoral', alpha=0.6, edgecolor='none',
               label='Investment Port')
    draw_vertical_lines(ax, licks, ymin=0.9, ymax=1, color='grey', alpha=0.1)
    draw_vertical_lines(ax, entries, color='b', linestyle='--')
    draw_vertical_lines(ax, exits, color='g', linestyle='--')
    draw_vertical_lines(ax, rewards, color='r')
    ax.plot(relative_time, DA_trace_to_plot, color='black')
    ax.set_xlim([-1, 15])
    ax.set_xticks(np.arange(0, 15.5, 2.5))
    ax.set_xlabel('Time since Trial Starts (s)')
    ax.set_ylabel('DA (z-score)')
    ax.set_title('Example Trial')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    if return_handle:
        fig.show()
        return fig, ax


def figa_example_trial_legend(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1))
        show = True
    else:
        show = False
    from matplotlib import lines

    lines_for_legend = [
        lines.Line2D([0, 0], [0, 10], color='b', linestyle=(0, (3, 1.5)), label='Entry'),
        lines.Line2D([0, 0], [0, 10], color='r', linestyle='solid', label='Reward'),
        lines.Line2D([0, 0], [0, 10], color='g', linestyle=(0, (3, 1.5)), label='Exit'),
        lines.Line2D([0, 0], [0, 10], color='grey', linestyle='solid', label='Lick Times'),
        # lines.Line2D([0, 1], [0, 0], color='k', linestyle='solid', label='DA (z-score)'),
    ]

    context_port_patch = mpatches.Patch(facecolor='skyblue', alpha=0.6, edgecolor='none', label='Context Port')
    investment_port_patch = mpatches.Patch(facecolor='lightcoral', alpha=0.6, edgecolor='none', label='Investment Port')

    all_handles = lines_for_legend[0:4] + [context_port_patch, investment_port_patch]
    legend = ax.legend(handles=all_handles, loc='center', prop={'size': 7.5}, frameon=False)
    # handler_map={plt.Line2D: plotting_utils.HandlerMiniatureLine()})

    # make the lines in the legend vertical
    legend.legendHandles[0].set_data([8, 8], [-1, 8])
    legend.legendHandles[1].set_data([8, 8], [0, 6])
    legend.legendHandles[2].set_data([8, 8], [-1, 8])
    legend.legendHandles[3].set_data([8, 8], [0, 6])
    legend.legendHandles[3].set_linewidth(1)
    # legend.legendHandles[6].set_data([2, 14], [3, 3])

    ax.axis('off')

    if show:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def resample_data_for_heatmap(zscore, reward_df, cutoff_pre_reward=-0.5, cutoff_post_reward=2, bin_size_s=0.05,
                              category_by='time_in_port'):
    """
    This function put the time series data on a canvas that aligns them with each reward in time investment port.
    Returns:
        1. a uniform time vector that spans from cutoff_pre_reward to cutoff_post_reward with the bin size as step size;
        2. category code array that tells you which category a certain row in the heatmap matrix belongs
        3. category labels
        4. heatmap matrix for directly putting into the seaborn heatmap function
    Arguments:
        zscore: time series data
        reward_df: reward_df with each row containing information about a time-investment port reward
        cutoff_pre_reward: left boundary of heatmap data selection
        cutoff_post_reward: right boundary of heatmap data selection
        bin_size_s: bin size in seconds
        category_by: could be 'time_in_port' or 'block'
    """
    # --- 1. Data preparation ---
    valid_df = reward_df[
        reward_df['time_in_port'].notna() & (reward_df['IRI_prior'] > 1) & (reward_df['IRI_post'] >= 0.6)]

    if category_by == 'time_in_port':
        bins = [0, 2, 6, 14, np.inf]
        bins = [0, 1.8, 4, 7.5, 12, np.inf]
        cat_labels = ['0-2 s', '2-6 s', '6-14 s', '>14 s']
        cat_labels = ['0-1.8 s', '1.8-4 s', '4-7.5 s', '7.5-12 s', '>12 s']
        valid_df['cat_code'] = pd.cut(valid_df['time_in_port'], bins=bins, labels=False, right=False)
        sorted_df = valid_df.sort_values(by='time_in_port').reset_index(drop=True)
    elif category_by == 'block':
        cat_labels = ['low', 'high']
        valid_df['cat_code'] = np.nan
        valid_df.loc[valid_df['block'] == '0.4', 'cat_code'] = 0
        valid_df.loc[valid_df['block'] == '0.8', 'cat_code'] = 1
        sorted_df = valid_df.sort_values(by=['block', 'time_in_port']).reset_index(drop=True)
    else:
        raise ValueError("category_by must be 'time_in_port' or 'block'")

    original_timestamps = zscore['time_recording'].values
    original_zscore = zscore['green_left'].values

    uniform_time_vector = np.arange(cutoff_pre_reward, cutoff_post_reward + bin_size_s, bin_size_s)
    heatmap_matrix = np.full((len(sorted_df), len(uniform_time_vector)), np.nan)

    # --- 2. Resample Each Trial and Place onto the Canvas ---
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
    category_codes = sorted_df['cat_code'].values
    return uniform_time_vector, category_codes, cat_labels, heatmap_matrix


def plot_heatmap_and_mean_traces(time_vector, category_codes, cat_labels, heatmap_matrix, palette='Reds_r', split_cat=0,
                                 legend_title='Time in Port', axes=None):
    """
    This function takes in prepared data and plot a heatmap and a plot with mean traces and error bands side by side
    """
    if heatmap_matrix.size == 0:
        print("No data to plot after filtering.")
        return None, None
    # --- 1. Calculate mean and sems ---
    stats = {}
    for i, label in enumerate(cat_labels):
        sub_matrix = heatmap_matrix[category_codes == i]
        if sub_matrix.shape[0] > 0:
            mean_trace = np.nanmean(sub_matrix, axis=0)
            std_trace = np.nanstd(sub_matrix, axis=0)
            n_samples = np.sum(~np.isnan(sub_matrix), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                sem_trace = std_trace / np.sqrt(n_samples)
                sem_trace[np.isnan(sem_trace)] = 0
            stats[label] = {
                'mean': mean_trace, 'sem': sem_trace,
                'upper_bound': mean_trace + sem_trace,
                'lower_bound': mean_trace - sem_trace,
                'n_trials': sub_matrix.shape[0]
            }

    # --- 2. Prepare axes/grids ---
    if axes is None:
        fig = plt.figure(figsize=(6, 3))
        gs_outer = GridSpec(1, 2, width_ratios=[21.5, 20], wspace=0.3)
        gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], width_ratios=[0.5, 20, 1], wspace=0.05)
        ax_bars = fig.add_subplot(gs_inner[0, 0])
        ax_heatmap = fig.add_subplot(gs_inner[0, 1])
        ax_cbar = fig.add_subplot(gs_inner[0, 2])
        ax_mean = fig.add_subplot(gs_outer[0, 1])
        return_handle = True
    else:
        fig = None
        return_handle = False
        ax_bars = axes[0]
        ax_heatmap = axes[1]
        ax_cbar = axes[2]
        ax_mean = axes[3]

    # --- 3. Plotting ---
    if split_cat:  # whether to add a blank space between different categories
        change_indices = np.where(np.diff(category_codes) != 0)[0] + 1
        for i in reversed(change_indices):
            heatmap_matrix = np.insert(heatmap_matrix, i, np.nan, axis=0)
            category_codes = np.insert(category_codes, i, np.nan)

    if np.all(np.isnan(heatmap_matrix)):
        print("No valid epochs were found to plot.")
        ax_heatmap.text(0.5, 0.5, 'No Data', ha='center', va='center')
        if return_handle:
            fig.show()
            return fig, ax_heatmap

    cat_num = len(cat_labels)
    palette_to_use = list(sns.color_palette(palette, n_colors=cat_num))
    custom_cat_cmap = ListedColormap(palette_to_use)

    nodes = [0.0, 0.5, 0.75, 1.0]
    colors = ["blue", "black", "red", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("my_cmap", list(zip(nodes, colors)))

    # Plot the category bars on the left axis (ax_bars)
    category_matrix = category_codes[:, np.newaxis]  # Reshape for heatmap
    # Use 'Reds_r' to go from dark red (low NRIs) to light red (high NRIs)
    # Use 'Set2' to distinguish low block and high block
    sns.heatmap(category_matrix, ax=ax_bars, cmap=custom_cat_cmap, cbar=False, xticklabels=False, yticklabels=False)
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
    # ax_cbar.set_title('DA\n(z-score)', fontsize='small', pad=0) # if label on top
    ax_cbar.set_ylabel('DA (z-score)', fontsize='small', rotation=-90, labelpad=-4, va="bottom")  # if label on the side
    ax_cbar.tick_params(direction='in', color='white', labelsize='small')

    # Customize the x-axis ticks to show time in seconds
    cutoff_pre_reward = round(time_vector[0], 1)
    cutoff_post_reward = round(time_vector[-1], 1)
    bin_size_s = round((time_vector[-1] - time_vector[0]) / (len(time_vector) - 1), 2)
    xtick_labels = [cutoff_pre_reward, 0, 1, int(cutoff_post_reward)]
    xtick_positions = [
        0,  # Start position
        np.abs(time_vector - 0).argmin(),  # Position of zero
        np.abs(time_vector - 0).argmin() + int(1 / bin_size_s),  # Position of one
        len(time_vector) - 1  # End position
    ]

    ax_heatmap.set_xticks(xtick_positions)
    ax_heatmap.set_xticklabels(xtick_labels, rotation=0)
    ax_heatmap.set_ylabel(None)

    # plot mean traces and their error bands from each category
    cat_num = len(cat_labels)
    if cat_num > 2:
        cat_labels_to_use = cat_labels[:-1]
    elif cat_num == 2:
        cat_labels_to_use = cat_labels
    else:
        raise ValueError("cat_labels must contain at least 2 elements.")

    for i, label in enumerate(cat_labels_to_use):
        lower_bound = stats[label]['lower_bound']
        upper_bound = stats[label]['upper_bound']
        ax_mean.plot(time_vector, stats[label]['mean'], label=label, color=palette_to_use[i])
        ax_mean.fill_between(time_vector, lower_bound, upper_bound, color=palette_to_use[i], alpha=0.2)
    ax_mean.spines['top'].set_visible(False)
    ax_mean.spines['right'].set_visible(False)
    # ax_mean.set_xticks(xtick_positions)
    ax_mean.set_xlim([cutoff_pre_reward, cutoff_post_reward])
    ax_mean.set_xticks(xtick_labels)
    ax_mean.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax_mean.set_ylabel('Mean DA (z-score)', labelpad=-6, y=0.55)
    ax_mean.legend(title=legend_title,
                   prop={'weight': 'normal', 'size': 'small'},
                   title_fontproperties={'weight': 'normal', 'size': 'small'},
                   handlelength=1, borderpad=0.4)

    for ax in [ax_heatmap, ax_mean]:
        ax.set_xlabel('Time from Reward (s)')

    if return_handle:
        fig.tight_layout()
        fig.show()


def figc_example_session_heatmap_split_by_NRI(zscore, reward_df, axes=None):
    time_vec, cat_codes, cat_labels, heatmap_mat = resample_data_for_heatmap(
        zscore, reward_df,
        cutoff_pre_reward=-0.5,
        cutoff_post_reward=2,
        bin_size_s=0.05,
        category_by='time_in_port'
    )
    plot_heatmap_and_mean_traces(time_vec, cat_codes, cat_labels, heatmap_mat, palette='Reds_r', split_cat=0,
                                 legend_title='Reward Time\nfrom Entry', axes=axes)


def figd_example_session_heatmap_split_by_block(zscore, reward_df, axes=None):
    time_vec, cat_codes, cat_labels, heatmap_mat = resample_data_for_heatmap(
        zscore, reward_df,
        cutoff_pre_reward=-0.5,
        cutoff_post_reward=2,
        bin_size_s=0.05,
        category_by='block'  # Use 'block' to test the new logic
    )
    plot_heatmap_and_mean_traces(time_vec, cat_codes, cat_labels, heatmap_mat, palette='Set2', split_cat=1,
                                 legend_title='Block', axes=axes)


def get_mean_sem_DA_for_feature(df, var='NRI', sample_per_bin=250):
    df_sorted = df.sort_values(by=var).reset_index(drop=True)
    total_sample = len(df_sorted)
    bin_num = int(total_sample / sample_per_bin)
    arr_bin_center = np.zeros(bin_num)
    arr_mean = np.zeros(bin_num)
    arr_sem = np.zeros(bin_num)
    for i in range(bin_num):
        arr_bin_center[i] = (df_sorted.iloc[sample_per_bin * i][var] + df_sorted.iloc[sample_per_bin * (i + 1) - 1][
            var]) / 2
        arr_mean[i] = df_sorted.iloc[sample_per_bin * i:sample_per_bin * (i + 1)]['DA'].mean()
        arr_sem[i] = df_sorted.iloc[sample_per_bin * i:sample_per_bin * (i + 1)]['DA'].sem()
    mean_sem_df = pd.DataFrame({'bin_center': arr_bin_center, 'mean': arr_mean, 'sem': arr_sem})
    return mean_sem_df


# def make_scatter_plot(df, x_var, y_var):

def figef_DA_vs_NRI(master_df, axes=None):
    data = master_df[(master_df['IRI'] > 1)]
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        return_handle = True
    else:
        fig = None
        return_handle = False
    ax = axes[0]
    for animal in data['animal'].unique():
        subset = data[data['animal'] == animal].copy()
        # # if plot scatter
        xy = np.vstack([subset['NRI'], subset['DA']])
        z = gaussian_kde(xy)(xy)
        subset['density'] = z
        sample_data = subset.sample(n=500, random_state=101).copy()  # if by fraction, use frac=0.05
        sample_data.sort_values('density', inplace=True)
        jitter_amount = 0.05
        sample_data['jittered_NRI'] = sample_data['NRI'] + np.random.normal(0, jitter_amount, size=len(sample_data))
        sns.scatterplot(data=sample_data, x='jittered_NRI', y='DA', hue='density', legend=False, alpha=0.5, s=5,
                        edgecolor='none', ax=ax)
        # df_plot = get_mean_sem_DA_for_feature(subset, var='NRI', sample_per_bin=300)
        # x = df_plot['bin_center']
        # y = df_plot['mean']
        # y_err = df_plot['sem']
        # line = ax.plot(x, y, linewidth=1.5, alpha=0.8, label=animal)
        # ax.fill_between(x, y - y_err, y + y_err, color=line[0].get_color(), alpha=0.5)

        # if plot fitted logarithmic function
        x_fit, y_fit, upper, lower = fit_to_logarithmic(subset, 'NRI', 'DA')
        line = ax.plot(x_fit, y_fit, linewidth=1.5, label=f'{animal}')
        ax.fill_between(x_fit, lower, upper, color=line[0].get_color(), alpha=0.3)
        ax.legend(title='Animal', fontsize='x-small')

    ax = axes[1]
    for block in data['block'].unique():
        if block == '0.4':
            color = sns.color_palette('Set2')[0]
            label = 'low'
        elif block == '0.8':
            color = sns.color_palette('Set2')[1]
            label = 'high'
        subset = data[data['block'] == block].copy()
        xy = np.vstack([subset['NRI'], subset['DA']])
        z = gaussian_kde(xy)(xy)
        subset['density'] = z
        sample_data = subset.sample(n=1000, random_state=1001).copy()  # if by fraction, use frac=0.05
        sample_data.sort_values('density', inplace=True)
        jitter_amount = 0.05
        sample_data['jittered_NRI'] = sample_data['NRI'] + np.random.normal(0, jitter_amount, size=len(sample_data))
        sns.scatterplot(data=sample_data, x='jittered_NRI', y='DA', hue='density',
                        palette=sns.light_palette(color, as_cmap=True),
                        legend=False, alpha=0.7, s=5, edgecolor='none', ax=ax)
        x_fit, y_fit, upper, lower = fit_to_logarithmic(subset, 'NRI', 'DA')
        ax.plot(x_fit, y_fit, c=color, linewidth=1.5, label=label)
        ax.fill_between(x_fit, lower, upper, color=color, alpha=0.3)
        ax.legend(title='Context\nReward Rate', fontsize='small')

    for ax in axes.flat:
        ax.set_xlim([-0.1, 11])
        ax.set_ylim([0, 4.5])
        ax.set_xlabel('Reward Time from Entry (s)')
        ax.set_ylabel('DA amplitude')
        ax.set_title('DA vs. NRI')

    if return_handle:
        fig.show()
        return fig, axes


def fige_DA_vs_NRI_v2(master_df, dodge=True, axes=None):
    data = master_df[(master_df['IRI'] > 1)]
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    # categorize the data into different groups of NRI
    # bins = helper.get_equal_area_bins(num_bins=10, cumulative=8.0, starting=1.0)
    bins = [0, 0.8, 1.8, 2.9, 4.1, 5.5, 7.3, 9.6, np.inf]
    bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    bin_labels[-1] = f'>{bins[-2]}'
    data['cat_code'] = pd.cut(data['NRI'], bins=bins)
    session_summary_data = data.groupby(['animal', 'session', 'cat_code'], observed=True).agg(NRI=('NRI', 'mean'),
                                                                                              DA=('DA',
                                                                                                  'mean')).reset_index()
    animal_summary_data = session_summary_data.groupby(['animal', 'cat_code'], observed=True).agg(
        DA=('DA', 'mean')).reset_index()
    sns.boxplot(data=session_summary_data, x='cat_code', y='DA', color='darkgray', showfliers=False,
                boxprops=dict(alpha=0.8), width=0.9, ax=axes)

    # custom_palette = sns.diverging_palette(10, 240, n=len(animal_order), sep=40, center="light")
    if dodge:
        sns.swarmplot(data=session_summary_data, x='cat_code', y='DA',
                      hue='animal', alpha=0.8, size=3,
                      dodge=dodge, legend=True, ax=axes)
    if not dodge:
        animal_order = data.groupby('animal')['DA'].mean().sort_values(ascending=False).index
        custom_palette = sns.color_palette("RdYlBu", n_colors=len(animal_order))
        sns.swarmplot(data=session_summary_data, x='cat_code', y='DA',
                      hue='animal', alpha=0.8, size=3,
                      dodge=dodge, hue_order=animal_order, palette=custom_palette,
                      legend=True, ax=axes)
    axes.set_title('DA vs. NRI')
    axes.set_xlabel('Reward Time from Entry')
    axes.set_ylabel('Mean DA (z-score)')
    axes.legend(title='Animal', bbox_to_anchor=(1.02, 1), loc='upper left')
    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def figg_DA_vs_IRI(master_df, axes=None):
    data = master_df
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(4, 6))
        return_handle = True
    else:
        fig = None
        return_handle = False
    ax = axes

    for animal in data['animal'].unique():
        subset = data[data['animal'] == animal].copy()
        # # if plot scatter
        xy = np.vstack([subset['IRI'], subset['DA']])
        z = gaussian_kde(xy)(xy)
        subset['density'] = z
        sample_data = subset.sample(n=500, random_state=101).copy()  # if by fraction, use frac=0.05
        sample_data.sort_values('density', inplace=True)
        jitter_amount = 0.05
        sample_data['jittered_IRI'] = sample_data['IRI'] + np.random.normal(0, jitter_amount, size=len(sample_data))
        sns.scatterplot(data=sample_data, x='jittered_IRI', y='DA', hue='density', legend=False, alpha=0.5, s=5,
                        edgecolor='none', ax=ax)

        # # if plot fitted logarithmic function
        # x_fit, y_fit, upper, lower = fit_to_logarithmic(subset, 'NRI', 'DA')
        # line = ax.plot(x_fit, y_fit, linewidth=1.5, label=f'{animal}')
        # ax.fill_between(x_fit, lower, upper, color=line[0].get_color(), alpha=0.3)
        # ax.legend(title='Animal', fontsize='x-small')
        ax.set_xlim([-0.1, 5.55])
        ax.set_ylim([0, 4.5])
        ax.set_xlabel('Reward Time from Prior Reward (s)')
        ax.set_ylabel('DA amplitude')
        ax.set_title('DA vs. IRI')

    if return_handle:
        fig.show()
        return fig, axes


def setup_axes_v1():
    fig_size = (12, 10)  # (width, height) in inches
    rows, cols = fig_size[1] * 10, fig_size[0] * 10

    row_1 = [8, 1, 2]
    row_2 = [1, 15, 2, 15, 1, 15, 2, 15]
    row_3 = [2, 2, 1]
    # row_4 = [3, 2]
    col_1 = [3, 4, 3]

    row_1_margins = [4, 6]
    row_2_margins = [1, 1, 10, 6, 1, 1, 10]
    row_3_margins = [10, 10]
    # row_4_margins = [10]
    col_1_margins = [10, 10]

    row_1_splits = [int((cols - np.sum(row_1_margins)) * h / sum(row_1)) for h in row_1]
    row_2_splits = [int((cols - np.sum(row_2_margins)) * h / sum(row_2)) for h in row_2]
    row_3_splits = [int((cols - np.sum(row_3_margins)) * h / sum(row_3)) for h in row_3]
    # row_4_splits = [int((cols - np.sum(row_4_margins)) * h / sum(row_4)) for h in row_4]
    col_1_splits = [int((rows - np.sum(col_1_margins)) * h / sum(col_1)) for h in col_1]

    row_1_splits = [val for pair in zip(row_1_splits, row_1_margins + [0]) for val in pair][:-1]
    row_2_splits = [val for pair in zip(row_2_splits, row_2_margins + [0]) for val in pair][:-1]
    row_3_splits = [val for pair in zip(row_3_splits, row_3_margins + [0]) for val in pair][:-1]
    # row_4_splits = [val for pair in zip(row_4_splits, row_4_margins + [0]) for val in pair][:-1]
    col_1_splits = [val for pair in zip(col_1_splits, col_1_margins + [0]) for val in pair][:-1]

    row_1_splits = np.cumsum(row_1_splits)
    row_2_splits = np.cumsum(row_2_splits)
    row_3_splits = np.cumsum(row_3_splits)
    # row_4_splits = np.cumsum(row_4_splits)
    col_1_splits = np.cumsum(col_1_splits)

    row_1_splits[-1] = cols
    row_2_splits[-1] = cols
    row_3_splits[-1] = cols
    # row_4_splits[-1] = cols
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

        # fig.add_subplot(gs[col_1_splits[5]:col_1_splits[6],
        #                 :row_4_splits[0]]),  # heatmap of DA vs. NRI vs. IRI and split by block
        # fig.add_subplot(gs[col_1_splits[5]:col_1_splits[6],
        #                 row_4_splits[1]:row_4_splits[2]]),  # scatters of predicted DA vs. real DA with fitting

    ]

    # # remove right and top spines
    # for ax in axes:
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    axes_to_letter = [0, 2, 3, 7, 11, 12, 13]
    for i, ax in enumerate(axes_to_letter):
        axes[ax].text(
            0.0, 1.0, lettering[i], transform=(
                    axes[ax].transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif', weight='bold')

    axes = np.array(axes)
    return axes


def setup_axes_v2():
    fig_size = (12, 10)  # (width, height) in inches
    rows, cols = fig_size[1] * 10, fig_size[0] * 10

    row_1 = [8, 1, 2]
    row_2 = [1, 15, 2, 15, 30]
    col_1 = [1, 2, 2]
    col_2 = [1, 1, 1]

    row_1_margins = [4, 6]
    row_2_margins = [1, 1, 10, 8]
    col_1_margins = [10, 10]
    col_2_margins = [10, 10]

    row_1_splits = [int((cols - np.sum(row_1_margins)) * h / sum(row_1)) for h in row_1]
    row_2_splits = [int((cols - np.sum(row_2_margins)) * h / sum(row_2)) for h in row_2]
    col_1_splits = [int((rows - np.sum(col_1_margins)) * h / sum(col_1)) for h in col_1]
    col_2_splits = [int((rows - np.sum(col_2_margins) - col_1_splits[0] - col_1_margins[0]) * h / sum(col_2)) for h in
                    col_2]
    col_2_splits = [col_1_splits[0]] + col_2_splits
    col_2_margins = [col_1_margins[0]] + col_2_margins

    row_1_splits = [val for pair in zip(row_1_splits, row_1_margins + [0]) for val in pair][:-1]
    row_2_splits = [val for pair in zip(row_2_splits, row_2_margins + [0]) for val in pair][:-1]
    col_1_splits = [val for pair in zip(col_1_splits, col_1_margins + [0]) for val in pair][:-1]
    col_2_splits = [val for pair in zip(col_2_splits, col_2_margins + [0]) for val in pair][:-1]

    row_1_splits = np.cumsum(row_1_splits)
    row_2_splits = np.cumsum(row_2_splits)
    col_1_splits = np.cumsum(col_1_splits)
    col_2_splits = np.cumsum(col_2_splits)

    row_1_splits[-1] = cols
    row_2_splits[-1] = cols
    col_1_splits += rows - col_1_splits[-1]
    col_2_splits[1:] += rows - col_2_splits[-1]

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    axes = [
        fig.add_subplot(gs[1:col_1_splits[0], :row_1_splits[0]]),  # example trial plot
        fig.add_subplot(gs[1:col_1_splits[0], row_1_splits[1]:row_1_splits[2]]),  # example trial legend
        fig.add_subplot(gs[:col_1_splits[0], row_1_splits[-2]:]),  # recording locations plot

        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2], :row_2_splits[0]]),  # category bar for heatmap 1
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2], row_2_splits[1]:row_2_splits[2]]),  # heatmap 1
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2], row_2_splits[3]:row_2_splits[4]]),  # colorbar for heatmap 1
        fig.add_subplot(gs[col_1_splits[1]:col_1_splits[2], row_2_splits[5]:row_2_splits[6]]),
        # transient averaged from heatmap 1

        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4], :row_2_splits[0]]),  # category bar for heatmap 2
        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4], row_2_splits[1]:row_2_splits[2]]),  # heatmap 2
        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4], row_2_splits[3]:row_2_splits[4]]),  # colorbar for heatmap 2
        fig.add_subplot(gs[col_1_splits[3]:col_1_splits[4], row_2_splits[5]:row_2_splits[6]]),
        # transient averaged from heatmap 2

        fig.add_subplot(gs[col_2_splits[1]:col_2_splits[2], row_2_splits[-2]:row_2_splits[-1]]),
        # DA vs. NRI for all animals
        fig.add_subplot(gs[col_2_splits[3]:col_2_splits[4], row_2_splits[-2]:row_2_splits[-1]]),
        # DA vs. NRI but split by blocks from all animals
        fig.add_subplot(gs[col_2_splits[5]:col_2_splits[6],
                        row_2_splits[-2]:int(row_2_splits[-2] + (row_2_splits[-1] - row_2_splits[-2]) / 2)])
        # DA vs. IRI

    ]

    # # remove right and top spines
    # for ax in axes:
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    axes_to_letter = [0, 2, 3, 7, 11, 12, 13]
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


def log_func(x, a, b):
    # y = a * log(x) + b
    # This function requires x > 0.
    return a * np.log(x) + b


def fit_to_logarithmic(df, x_col, y_col):
    # curve fit
    popt, pcov = curve_fit(log_func, df[x_col], df[y_col])
    a_opt, b_opt = popt

    # confidence interval
    n = len(df[x_col])  # Number of data points
    dof = n - len(popt)  # Degrees of freedom
    alpha = 0.05  # Confidence level (1 - 0.95 = 0.05)
    t_crit = np.abs(t.ppf(alpha / 2, dof))  # Critical t-value

    # smooth x space
    x_fit = np.linspace(df[x_col].min(), df[x_col].quantile(0.95), 200)
    y_fit = log_func(x_fit, a_opt, b_opt)

    # standard error of the y-fit
    y_fit_stderr = np.sqrt(pcov[0, 0] * (np.log(x_fit)) ** 2 + pcov[1, 1] + 2 * pcov[0, 1] * np.log(x_fit))

    # upper and lower bounds of the confidence interval
    upper_bound = y_fit + t_crit * y_fit_stderr
    lower_bound = y_fit - t_crit * y_fit_stderr

    return x_fit, y_fit, upper_bound, lower_bound


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

    # --- good example sessions for heatmaps ---
    # animal SZ036
    # '2024-01-03T16_13'
    # '2024-01-04T11_40'
    # animal SZ037
    # '2024-01-04T12_45'
    # '2024-01-12T10_36'
    # animal SZ042
    # '2023-12-28T16_34'
    # '2023-12-30T21_48'
    # --- example trial log ends ---

    # --- data preparation ---
    animal_str = 'SZ036'
    session_name = '2024-01-08T13_52'
    zscore_example_trial = data_loader.load_session_dataframe(animal_str, 'zscore',
                                                              session_long_name=session_name,
                                                              file_format='parquet')
    trial_df = data_loader.load_session_dataframe(animal_str, 'trial_df', session_long_name=session_name,
                                                  file_format='parquet')

    animal_str = 'SZ037'
    session_name = '2024-01-04T12_45'
    zscore_heatmap = data_loader.load_session_dataframe(animal_str, 'zscore', session_long_name=session_name,
                                                        file_format='parquet')
    reward_df = data_loader.load_session_dataframe(animal_str, 'expreward_df', session_long_name=session_name,
                                                   file_format='parquet')

    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    # animal_ids=["SZ036"]
    master_DA_features_df = data_loader.load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', 'parquet')
    # --- end of data preparation ---

    # --- set up axes and add figures to axes ---
    axes = setup_axes_v2()

    tic = time.time()
    figa_example_trial_1d_traces(zscore_example_trial, trial_df, example_trial_id=32, ax=axes[0])
    figa_example_trial_legend(ax=axes[1])
    print(f'figa_example_trial took {time.time() - tic:.2f} seconds')

    tic = time.time()
    figc_example_session_heatmap_split_by_NRI(zscore_heatmap, reward_df, axes=axes[3:7])
    print(f'figc_example_session_heatmap_split_by_NRI took {time.time() - tic:.2f} seconds')

    tic = time.time()
    figd_example_session_heatmap_split_by_block(zscore_heatmap, reward_df, axes=axes[7:11])
    print(f'figd_example_session_heatmap_split_by_block took {time.time() - tic:.2f} seconds')

    tic = time.time()
    figef_DA_vs_NRI(master_DA_features_df, axes=axes[11:13])
    print(f'figef_DA_vs_NRI took {time.time() - tic:.2f} seconds')

    tic = time.time()
    figg_DA_vs_IRI(master_DA_features_df, axes=axes[13])
    print(f'figg_DA_vs_IRI took {time.time() - tic:.2f} seconds')

    plt.subplots_adjust(hspace=0, wspace=0)  # left=0.07, right=0.98, bottom=0.15, top=0.9
    plt.show()
    print('hello')
    # --- end of adding figures to axes

    # --- go through all the trials in one session when looking for good trials ---
    # for trial_id in trial_df['trial'].values:
    #     figc_example_1d_traces(zscore, trial_df, example_trial_id=trial_id, ax=None)
    #     time.sleep(2) # Pause for 2 seconds between each plot to respect the server's rate limit
    # --- end ---


if __name__ == '__main__':
    # main()

    # animal_str = 'SZ037'
    # # session_name = '2024-01-04T15_49'
    # for i in range(0, 25):
    #     session_id = i
    #     zscore_heatmap = data_loader.load_session_dataframe(animal_str, 'zscore', session_id=i,
    #                                                         file_format='parquet')
    #     reward_df = data_loader.load_session_dataframe(animal_str, 'expreward_df', session_id=i,
    #                                                    file_format='parquet')
    #     time_vec, cat_codes, cat_labels, heatmap_mat = resample_data_for_heatmap(
    #         zscore_heatmap, reward_df,
    #         cutoff_pre_reward=-0.5,
    #         cutoff_post_reward=2,
    #         bin_size_s=0.05,
    #         category_by='time_in_port'  # Use 'block' to test the new logic
    #     )
    #     plot_heatmap_and_mean_traces(time_vec, cat_codes, cat_labels, heatmap_mat, palette='Reds_r', split_cat=0,
    #                                  axes=None)
    #
    #     time_vec, cat_codes, cat_labels, heatmap_mat = resample_data_for_heatmap(
    #         zscore_heatmap, reward_df,
    #         cutoff_pre_reward=-0.5,
    #         cutoff_post_reward=2,
    #         bin_size_s=0.05,
    #         category_by='block'  # Use 'block' to test the new logic
    #     )
    #     plot_heatmap_and_mean_traces(time_vec, cat_codes, cat_labels, heatmap_mat, palette='Set2', split_cat=1,
    #                                  axes=None)

    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    # animal_ids = ["SZ036"]
    master_DA_features_df = data_loader.load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features',
                                                                           day_0='2023-11-30', file_format='parquet')
    # figef_DA_vs_NRI(master_DA_features_df, axes=None)
    # figg_DA_vs_IRI(master_DA_features_df, axes=None)
    fige_DA_vs_NRI_v2(master_DA_features_df, dodge=True, axes=None)
    print("hello")
