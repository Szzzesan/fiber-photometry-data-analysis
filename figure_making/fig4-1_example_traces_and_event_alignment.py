import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy import integrate

# project modules
import config
import data_loader
import fig_dopamine  # Reusing example trial and trace logic
import helper
import quality_control as qc

def get_trial_duration(trial_df, trial_id):
    """Calculates duration from bg_entry to exp_exit for a given trial."""
    row = trial_df[trial_df['trial'] == trial_id].iloc[0]
    return row['exp_exit'] - row['bg_entry']

def find_first_lick_after_rewards(reward_df, trial_df):
    """Finds the first lick timestamp occurring after each reward delivery."""
    first_licks = []
    for _, row in reward_df.iterrows():
        trial_id = row['trial']
        rew_time = row['reward_time']

        # Get licks for this specific trial
        trial_licks = trial_df.loc[trial_df['trial'] == trial_id, 'licks'].values
        if len(trial_licks) > 0 and isinstance(trial_licks[0], (list, np.ndarray)):
            licks = np.array(trial_licks[0])
            post_reward_licks = licks[licks > rew_time]
            if len(post_reward_licks) > 0:
                first_licks.append(post_reward_licks[0])
            else:
                first_licks.append(np.nan)
        else:
            first_licks.append(np.nan)
    return np.array(first_licks)


def resample_chronological_heatmap(zscore, zscore_branch, event_times, cutoff_pre=-0.5, cutoff_post=2, bin_size=0.05):
    """
    Resamples data aligned to events, ordered chronologically.
    """
    original_timestamps = zscore['time_recording'].values
    original_zscore = zscore[zscore_branch].values

    uniform_time_vector = np.arange(cutoff_pre, cutoff_post + bin_size, bin_size)
    heatmap_matrix = np.full((len(event_times), len(uniform_time_vector)), np.nan)

    for i, t_event in enumerate(event_times):
        if pd.isna(t_event):
            continue

        start_time = t_event + cutoff_pre
        end_time = t_event + cutoff_post

        target_ts = np.arange(start_time, end_time, bin_size)

        # Find window in source data
        idx_start = helper.find_closest_value(original_timestamps, start_time)
        idx_end = helper.find_closest_value(original_timestamps, end_time)

        source_ts = original_timestamps[idx_start:idx_end]
        source_vals = original_zscore[idx_start:idx_end]

        if source_ts.size > 1:
            resampled = np.interp(target_ts, source_ts, source_vals)
            # Fill the matrix row (handle length mismatch if at end of recording)
            length = min(len(resampled), len(uniform_time_vector))
            heatmap_matrix[i, :length] = resampled[:length]

    return uniform_time_vector, heatmap_matrix


def plot_heatmap_and_mean_traces_no_bars(time_vector, heatmap_matrix, axes=None):
    """
    New plotting function without the categorical colorbars on the left.
    Expects axes: [ax_heatmap, ax_cbar, ax_mean]
    """
    if heatmap_matrix.size == 0 or np.all(np.isnan(heatmap_matrix)):
        print("No valid data to plot.")
        return

    ax_heatmap, ax_cbar, ax_mean = axes

    # Calculate mean and SEM
    mean_trace = np.nanmean(heatmap_matrix, axis=0)
    sem_trace = np.nanstd(heatmap_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(heatmap_matrix), axis=0))

    # Reusing the project's standard dopamine colormap
    nodes = [0.0, 0.5, 0.75, 1.0]
    colors = ["blue", "black", "red", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("dopamine_cmap", list(zip(nodes, colors)))

    # Plot Heatmap
    sns.heatmap(
        heatmap_matrix,
        ax=ax_heatmap,
        cmap=custom_cmap,
        center=0,
        cbar=True,
        yticklabels=False,
        cbar_ax=ax_cbar,
    )
    # Calculate tick positions based on the time_vector
    cutoff_pre = round(time_vector[0], 1)
    cutoff_post = round(time_vector[-1], 1)
    # Determine indices for specific time points
    xtick_labels = [cutoff_pre, 0.0, 1.0, round(cutoff_post, 1)]
    xtick_positions = [
        0,  # Start (-0.5s)
        np.abs(time_vector - 0).argmin(),  # 0s
        np.abs(time_vector - 1).argmin(),  # 1s
        len(time_vector) - 1  # End (2s)
    ]

    ax_heatmap.set_xticks(xtick_positions)
    ax_heatmap.set_xticklabels(xtick_labels, rotation=0)
    ax_heatmap.axvline(x=xtick_positions[1], color='white', linestyle='--', alpha=0.8)
    # ---------------------------

    ax_cbar.set_ylabel('DA (z-score)', fontsize='small', rotation=-90, labelpad=-4, va="bottom")
    ax_heatmap.set_ylabel('Trials (Chronological)', fontsize='small')
    # ax_heatmap.set_xlabel('Time from Event (s)', fontsize='small')

    # Plot Mean Trace with SEM
    ax_mean.plot(time_vector, mean_trace, color='red', linewidth=1.5)
    ax_mean.fill_between(time_vector, mean_trace - sem_trace, mean_trace + sem_trace,
                         color='red', alpha=0.3, edgecolor='none')
    ax_mean.axvline(x=0, color='blue', linestyle='--', linewidth=1)
    ax_mean.set_xlim([time_vector[0], time_vector[-1]])
    ax_mean.set_xticks(xtick_labels)  # Ensure mean trace labels match heatmap
    ax_mean.set_ylabel('Mean DA')
    ax_mean.set_xlabel('Time from Event (s)')
    ax_mean.spines['top'].set_visible(False)
    ax_mean.spines['right'].set_visible(False)

def setup_thesis_grid(dur1, dur2):
    """Creates a grid for 2 top traces and 2 heatmap/trace groups at bottom.
    Grid setup where ax_t1 width is proportional to dur1/dur2."""
    fig = plt.figure(figsize=(12, 14))
    num_cols = 100  # High resolution for precise alignment
    gs = gridspec.GridSpec(20, num_cols, figure=fig)

    max_dur = max(dur1, dur2)
    split_col = int(num_cols * (dur1 / max_dur))

    ax_t1 = fig.add_subplot(gs[0:3, :split_col])
    ax_t2 = fig.add_subplot(gs[4:7, :])  # Full width
    ax_leg = fig.add_subplot(gs[0:3, split_col+10 : split_col+20])  # +1 for padding
    ax_leg.axis('off')

    # Bottom Groups
    def make_group(col_range):
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[9:19, col_range],
                                                    height_ratios=[4, 1], width_ratios=[20, 1], hspace=0.2, wspace=0.05)
        return [fig.add_subplot(inner_gs[0, 0]), fig.add_subplot(inner_gs[0, 1]), fig.add_subplot(inner_gs[1, 0])]

    return fig, [ax_t1, ax_t2, ax_leg], make_group(slice(0, 45)), make_group(slice(55, 100))

# --- Analyze peak-to-event alignment ---
def quantify_da(col_name, dFF0, pi_events, plot=False):
    """
    Detects dopamine peaks and calculates properties.
    As provided in the user's codebase.
    """
    dff0 = dFF0[col_name]
    my_zscore = (dff0.to_numpy() - dff0.mean()) / dff0.std()

    # Restrict threshold calculation to task period
    idx_taskbegin = dFF0.index[dFF0['time_recording'] >= pi_events['time_recording'].min()].min()
    idx_taskend = dFF0.index[dFF0['time_recording'] <= pi_events['time_recording'].max()].max()
    threshold = np.nanpercentile(my_zscore[idx_taskbegin:idx_taskend + 1], 90)

    # Peak detection using scipy
    peaks, _ = find_peaks(my_zscore, rel_height=0.25, width=[4, 40], height=threshold, wlen=60, prominence=1)
    widths = peak_widths(my_zscore, peaks, rel_height=0.25)
    prominences = peak_prominences(my_zscore, peaks, wlen=60)

    # Calculate Area Under Curve (AUC)
    auc = np.empty(len(peaks))
    for i in range(len(peaks)):
        peak_start, peak_end = prominences[1][i], prominences[2][i]
        auc[i] = integrate.trapezoid(y=dff0[peak_start:peak_end],
                                     x=dFF0.time_recording[peak_start:peak_end])

    return my_zscore, peaks, widths, prominences, auc


def calculate_peak_event_intervals(peaks_df, pi_events, event_name, condition, search_window):
    """
    Calculates the time difference (Peak - Event) for the first occurrence
    of a behavioral event within a specific window around the peak.
    """
    peaks_df[event_name] = np.nan
    event_times = pi_events.loc[condition, 'time_recording'].values

    if len(event_times) == 0:
        return peaks_df

    for idx, peak_time in peaks_df['peak_time'].items():
        # Define search window relative to peak
        t_start = peak_time + search_window[0]
        t_end = peak_time + search_window[1]

        # Find events in range
        in_range = event_times[(event_times >= t_start) & (event_times <= t_end)]

        if len(in_range) > 0:
            # Calculate interval: Peak - Event
            # Positive value means peak happened AFTER the event.
            peaks_df.at[idx, event_name] = peak_time - in_range[0]

    return peaks_df


def process_animal_data(animal_ids):
    """
    Iterates through all animals and sessions to aggregate peak-to-event intervals.
    """
    all_results = []

    for animal in animal_ids:
        # Determine number of sessions by scanning the processed directory
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            continue

        session_files = [f for f in os.listdir(processed_dir) if
                         f.startswith(f"{animal}_") and f.endswith("_dFF0.parquet")]
        num_sessions = len(session_files)

        # Get valid hemispheres for this animal from QC
        hemispheres = qc.qc_selections.get(animal, [])

        for sess_idx in range(num_sessions):
            dFF0 = data_loader.load_session_dataframe(animal, 'dFF0', session_id=sess_idx)
            pi_events = data_loader.load_session_dataframe(animal, 'pi_events_processed', session_id=sess_idx)

            if dFF0 is None or pi_events is None:
                continue

            for hemi in hemispheres:
                col = f'green_{hemi}'
                if col not in dFF0.columns:
                    continue

                # Detect Transients
                _, peaks, _, _, _ = quantify_da(col, dFF0, pi_events)

                if len(peaks) == 0:
                    continue

                # Store peak times
                peak_data = pd.DataFrame({
                    'peak_time': dFF0.loc[peaks, 'time_recording'].values,
                    'animal': animal,
                    'session': sess_idx,
                    'hemisphere': hemi
                })

                # Calculate intervals for various behaviors
                # 1. Reward delivery
                peak_data = calculate_peak_event_intervals(peak_data, pi_events, 'Reward',
                                                           (pi_events['key'] == 'reward') & (pi_events['value'] == 1),
                                                           [-1.0, 1.0])
                # 2. First Lick
                peak_data = calculate_peak_event_intervals(peak_data, pi_events, 'First_Lick',
                                                           pi_events['is_1st_lick'], [-1.0, 1.0])
                # 3. Port Entry
                peak_data = calculate_peak_event_intervals(peak_data, pi_events, 'Entry',
                                                           (pi_events['key'] == 'head') & (pi_events['value'] == 1),
                                                           [-1.0, 1.0])
                # 4. Port Exit
                peak_data = calculate_peak_event_intervals(peak_data, pi_events, 'Exit',
                                                           (pi_events['key'] == 'head') & (pi_events['value'] == 0),
                                                           [-1.0, 1.0])
                # 5. First Encounter (Visual/Auditory Cue)
                peak_data = calculate_peak_event_intervals(peak_data, pi_events, 'First_Encounter',
                                                           pi_events['is_1st_encounter'], [-1.0, 1.0])

                all_results.append(peak_data)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def plot_with_std_labels(master_df):
    """
    Plots the violin plot and annotates each category with the Standard Deviation.
    """
    event_order = ['Reward', 'First_Lick', 'Entry', 'Exit', 'First_Encounter']
    plot_df = master_df.melt(id_vars=['animal', 'session', 'hemisphere', 'peak_time'],
                             value_vars=event_order, var_name='Event', value_name='Interval')
    plot_df = plot_df.dropna(subset=['Interval'])

    # Calculate Standard Deviation
    stats = plot_df.groupby('Event')['Interval'].std().reindex(event_order)

    plt.figure(figsize=(12, 7))
    # sns.set_style("whitegrid")
    ax = sns.violinplot(data=plot_df, x='Event', y='Interval',
                        inner='quartile', palette='Accent', order=event_order)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Add Standard Deviation labels
    for i, event in enumerate(event_order):
        if event in stats.index:
            std_val = stats.loc[event]

            # Position the text slightly above the top of the distribution
            y_max = plot_df[plot_df['Event'] == event]['Interval'].max()

            # Use a bold label for the standard deviation
            ax.text(i + 0.1, y_max + 0.05, f"$\sigma = {std_val:.3f}$",
                    fontsize=12, fontweight='bold', color='darkslategray')

    # plt.title('Dopamine Peak Locking: Jitter Analysis ($\sigma$)', fontsize=15)
    plt.ylabel('Peak - Event Interval (s)')
    plt.tight_layout()

    # Save the figure to the summary directory defined in config
    # save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR, 'da_locking_std_only.png')
    # plt.savefig(save_path)
    plt.show()

def setup_thesis_grid_v2(dur1, dur2):
    """
    Modified to have example traces on top and a full-width violin plot
    on the bottom row.
    """
    fig = plt.figure(figsize=(12, 12))
    num_cols = 100
    gs = gridspec.GridSpec(20, num_cols, figure=fig)

    max_dur = max(dur1, dur2)
    split_col = int(num_cols * (dur1 / max_dur))

    # Top Row: Example Traces
    ax_t1 = fig.add_subplot(gs[0:3, :split_col])
    ax_t2 = fig.add_subplot(gs[4:7, :])
    ax_leg = fig.add_subplot(gs[0:3, split_col+10 : split_col+20])
    ax_leg.axis('off')

    # Bottom Row: Single full-width subplot for Violin Plot
    ax_violin = fig.add_subplot(gs[10:19, :])

    return fig, [ax_t1, ax_t2, ax_leg], ax_violin

# --- Modified Violin Plot Function to accept an axis ---
def plot_violin_on_axis(master_df, ax):
    """
    Modified version of your plot_with_std_labels to plot on a provided axis.
    """
    event_order = ['Reward', 'First_Lick', 'Entry', 'Exit', 'First_Encounter']
    plot_df = master_df.melt(id_vars=['animal', 'session', 'hemisphere', 'peak_time'],
                             value_vars=event_order, var_name='Event', value_name='Interval')
    plot_df = plot_df.dropna(subset=['Interval'])

    # Calculate Standard Deviation
    stats = plot_df.groupby('Event')['Interval'].std().reindex(event_order)

    # Plot on the specific axis
    sns.violinplot(data=plot_df, x='Event', y='Interval',
                   inner='quartile', palette='Accent', order=event_order, ax=ax)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Add Standard Deviation labels
    for i, event in enumerate(event_order):
        if event in stats.index:
            std_val = stats.loc[event]
            y_max = plot_df[plot_df['Event'] == event]['Interval'].max()
            ax.text(i + 0.1, y_max + 0.05, f"$\sigma = {std_val:.3f}$",
                    fontsize=12, fontweight='bold', color='darkslategray')

    ax.set_ylabel('Peak - Event Interval (s)')
    ax.set_title('Dopamine Peak-to-Event Alignment', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    # 0. Load Data
    animal_trace = 'SZ036'
    session_trace = '2024-01-08T13_52'
    animal_heat = 'SZ043'
    session_heat = '2023-12-24T15_54'
    hemi = 'left'
    branch = f'green_{hemi}'

    zscore_trace = data_loader.load_session_dataframe(animal_trace, 'zscore', session_long_name=session_trace)
    trial_df_trace = data_loader.load_session_dataframe(animal_trace, 'trial_df', session_long_name=session_trace)

    zscore_heat = data_loader.load_session_dataframe(animal_heat, 'zscore', session_long_name=session_heat)
    reward_df = data_loader.load_session_dataframe(animal_heat, 'expreward_df', session_long_name=session_heat)
    trial_df_heat = data_loader.load_session_dataframe(animal_heat, 'trial_df', session_long_name=session_heat)

    # 1. Get trial durations and setup grid
    t1_id, t2_id = 32, 19
    dur1 = get_trial_duration(trial_df_trace, t1_id)
    dur2 = get_trial_duration(trial_df_trace, t2_id)
    # fig, top_axes, h1_axes, h2_axes = setup_thesis_grid(dur1, dur2)
    fig, top_axes, ax_violin = setup_thesis_grid_v2(dur1, dur2)

    # 2. Plot traces with manual x-limits (No sharex)
    for i, (ax, tid, dur) in enumerate(zip(top_axes[:2], [t1_id, t2_id], [dur1, dur2])):
        fig_dopamine.figa_example_trial_1d_traces(zscore_trace, trial_df_trace, tid, ax=ax)
        ax.set_xlim(0, dur)

        # Customize Titles and Ticks
        ax.set_title(f'Example Trial {i + 1}')
        ticks = np.arange(0, dur + 0.1, 2.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.tick_params(labelbottom=True)
        if i == 1: ax.set_xlabel('Time since Trial Starts (s)')
        if i < 1: ax.set_xlabel('')

    fig_dopamine.figa_example_trial_legend(ax=top_axes[2])


    # # 4. Process Heatmap Group 1: Reward Aligned, Chronological
    # # We pass category_codes as just a sequence to force chronological visual
    # reward_times = reward_df['reward_time'].values
    # t_vec, h_mat = resample_chronological_heatmap(zscore_heat, 'green_right', reward_times)
    #
    #
    # plot_heatmap_and_mean_traces_no_bars(
    #     t_vec, h_mat,
    #     axes=h1_axes
    # )
    # h1_axes[0].set_title('Aligned to Reward')
    #
    # # 5. Process Heatmap Group 2: Lick Aligned, Chronological
    # lick_times = find_first_lick_after_rewards(reward_df, trial_df_heat)
    # t_vec_lick, h_mat_lick = resample_chronological_heatmap(zscore_heat, 'green_right', lick_times)
    #
    # plot_heatmap_and_mean_traces_no_bars(
    #     t_vec_lick, h_mat_lick,
    #     axes=h2_axes
    # )
    # h2_axes[0].set_title('Aligned to First Lick after Reward')

    print("Processing multi-animal peak data...")
    animal_list = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    master_intervals = process_animal_data(animal_list)  # Reusing your function

    if not master_intervals.empty:
        plot_violin_on_axis(master_intervals, ax_violin)

    # Formatting
    lettering = 'abcde'
    for i, ax in enumerate([top_axes[0], ax_violin]):
        ax.text(-0.05, 1.0, lettering[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')

    # Save
    save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR,
                             f'fig_4-1_Example_Traces_Alignment.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
    # animal_list = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    #
    # print("Beginning cross-session peak-locking analysis...")
    # master_df = process_animal_data(animal_list)
    #
    # if not master_df.empty:
    #     plot_with_std_labels(master_df)
    #     # Export data for further statistical testing
    #     # master_df.to_csv("peak_to_event_intervals_summary.csv", index=False)
    # else:
    #     print("Analysis complete: No valid peaks or intervals found in the dataset.")