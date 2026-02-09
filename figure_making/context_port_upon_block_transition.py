import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import config
import quality_control as qc
from data_loader import load_session_dataframe, load_dataframes_for_animal_summary


def get_trial_metadata(pi_events):
    """Extracts trial-level info and identifies block transitions."""
    # Entry to background port (Port 2) is defined by 'trial' key starting (value 1)
    is_bg_entry = (pi_events['key'] == 'trial') & (pi_events['value'] == 1) & pi_events['is_valid']
    df_trials = pi_events[is_bg_entry][['trial', 'phase', 'time_recording']].copy()
    df_trials.columns = ['trial', 'block', 'entry_time']

    # Identify block changes
    df_trials['block_change'] = df_trials['block'] != df_trials['block'].shift()
    # Tag transition type: 1 for L->H, -1 for H->L
    df_trials['transition_type'] = np.where(
        (df_trials['block_change']) & (df_trials['block'] == '0.8'), 'LowToHigh',
        np.where((df_trials['block_change']) & (df_trials['block'] == '0.4'), 'HighToLow', None)
    )
    df_trials.reset_index(drop=True, inplace=True)
    return df_trials


def plot_block_transitions(animal_id, session_long_name):
    # 1. Load Data
    zscore = load_session_dataframe(animal_id, 'zscore', session_long_name=session_long_name)
    pi_events = load_session_dataframe(animal_id, 'pi_events_processed', session_long_name=session_long_name)

    if zscore is None or pi_events is None: return

    # 2. Extract Metadata and Transitions
    df_trials = get_trial_metadata(pi_events)
    transitions = df_trials[df_trials['transition_type'].notna()].index.tolist()

    # 3. Setup Aesthetics
    palette = sns.color_palette('Set2')
    colors = {'0.4': palette[0], '0.8': palette[1]}
    fs = 40  # Recording frequency

    branches = [c for c in zscore.columns if 'green' in c]
    transition_types = ['LowToHigh', 'HighToLow']

    for trans_type in transition_types:
        type_indices = df_trials[df_trials['transition_type'] == trans_type].index.tolist()
        if not type_indices: continue

        for branch in branches:
            num_rows = len(type_indices)
            fig, axes = plt.subplots(num_rows, 6, figsize=(20, 3 * num_rows), sharex=True, sharey=True)
            if num_rows == 1: axes = np.expand_dims(axes, axis=0)

            fig.suptitle(f"{animal_id} | {session_long_name} | {branch}\nTransition: {trans_type}", fontsize=16)

            for i, switch_idx in enumerate(type_indices):
                # Select trials -2, -1, 0, 1, 2, 3 relative to switch
                window = range(switch_idx - 2, switch_idx + 4)

                for j, trial_idx in enumerate(window):
                    if trial_idx < 0 or trial_idx >= len(df_trials): continue

                    trial_info = df_trials.iloc[trial_idx]
                    ax = axes[i, j]

                    # REQUIREMENT (3): Stitch signal by filtering for Port 2
                    # This removes time spent outside the port and joins the signal segments.
                    trial_sig = zscore[(zscore['trial'] == trial_info['trial']) & (zscore['port'] == 'bg')]

                    if not trial_sig.empty:
                        # Create cumulative "Time in Port" x-axis
                        time_in_port = np.arange(len(trial_sig)) / fs
                        ax.plot(time_in_port, trial_sig[branch], color=colors[trial_info['block']], lw=2)

                        # Plot vertical lines at 1.25, 2.5, 3.75, and 5
                        target_lines = [1.25, 2.5, 3.75, 5.0]
                        for tl in target_lines:
                            ax.axvline(tl, color='gray', linestyle='--', alpha=0.2, zorder=1)

                        # Plot Reward Markers (Dashed lines at 0, 1.25, 2.5, 3.75 for High; 0, 2.5, 5.0 for Low)
                        rewards = [1.25, 2.5, 3.75, 5] if trial_info['block'] == '0.8' else [0, 2.5, 5.0]
                        for r in rewards:
                            if r <= 6: ax.axvline(r, color='blue', ls='--', alpha=0.6)

                    # Subplot labeling
                    if i == 0: ax.set_title(f"Trial {j - 2}")
                    if j == 0: ax.set_ylabel(f"Transition {i + 1}\nZ-score")
                    ax.set_xlim(0, 6)  # REQUIREMENT (2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.show()
            # Save the plots
            fig_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.FIGURE_SUBDIR, 'context_transition_single_session')
            if not os.path.exists(fig_dir): os.makedirs(fig_dir)
            save_path = os.path.join(fig_dir, f"{animal_id}_{trans_type}_{branch}.png")
            plt.savefig(save_path)
            print(f"Saved: {save_path}")


def run_batch_transition_analysis(animal_ids):
    """Iterates through all animals and sessions to generate transition plots."""

    # 1. Setup Aesthetics
    palette = sns.color_palette('Set2')
    colors = {'0.4': palette[0], '0.8': palette[1]}
    fs = 40  # Sample rate
    transition_types = ['LowToHigh', 'HighToLow']

    for animal in animal_ids:
        # Construct path to processed directory
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)

        if not os.path.exists(processed_dir):
            print(f"Skipping {animal}: Directory not found at {processed_dir}")
            continue

        # Find all zscore files to identify unique sessions
        # Pattern: {animal}_{timestamp}_zscore.parquet
        search_pattern = os.path.join(processed_dir, f"{animal}_*_zscore.parquet")
        session_files = glob.glob(search_pattern)

        for z_file in session_files:
            # Extract session_long_name (yyyy-mm-ddThh_mm)
            # Filename example: SZ036_2024-01-11T16_25_zscore.parquet
            base_name = os.path.basename(z_file)
            session_long_name = base_name.replace(f"{animal}_", "").replace("_zscore.parquet", "")

            print(f"Processing Animal: {animal} | Session: {session_long_name}")

            # 2. Load Data
            zscore = load_session_dataframe(animal, 'zscore', session_long_name=session_long_name)
            pi_events = load_session_dataframe(animal, 'pi_events_processed', session_long_name=session_long_name)

            if zscore is None or pi_events is None:
                continue

            # 3. Extract Metadata
            df_trials = get_trial_metadata(pi_events)
            branches = [c for c in zscore.columns if 'green' in c]

            for trans_type in transition_types:
                type_indices = df_trials[df_trials['transition_type'] == trans_type].index.tolist()
                if not type_indices:
                    continue

                for branch in branches:
                    num_rows = len(type_indices)
                    fig, axes = plt.subplots(num_rows, 6, figsize=(20, 3 * num_rows), sharex=True, sharey=True,
                                             squeeze=False)

                    fig.suptitle(f"{animal} | {session_long_name} | {branch}\nTransition: {trans_type}", fontsize=16)

                    for i, switch_idx in enumerate(type_indices):
                        window = range(switch_idx - 2, switch_idx + 4)

                        for j, trial_idx in enumerate(window):
                            if trial_idx < 0 or trial_idx >= len(df_trials):
                                continue

                            trial_info = df_trials.iloc[trial_idx]
                            ax = axes[i, j]

                            # Stitching logic: filter for 'bg' port
                            trial_sig = zscore[(zscore['trial'] == trial_info['trial']) & (zscore['port'] == 'bg')]

                            if not trial_sig.empty:
                                time_in_port = np.arange(len(trial_sig)) / fs
                                ax.plot(time_in_port, trial_sig[branch], color=colors[str(trial_info['block'])], lw=2)

                                # Gray reference lines
                                for tl in [1.25, 2.5, 3.75, 5.0]:
                                    ax.axvline(tl, color='gray', linestyle='--', alpha=0.6, zorder=1)

                                # Plot Reward Markers (Dashed lines at 0, 1.25, 2.5, 3.75 for High; 0, 2.5, 5.0 for Low)
                                rewards = [1.25, 2.5, 3.75, 5] if trial_info['block'] == '0.8' else [0, 2.5, 5.0]
                                for r in rewards:
                                    if r <= 6: ax.axvline(r, color='blue', ls='--', alpha=0.6)

                            if i == 0: ax.set_title(f"Trial {j - 2}")
                            if j == 0: ax.set_ylabel(f"Trans {i + 1}\nZ-score")
                            ax.set_xlim(0, 6)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                    # 4. Save with session name in file
                    # Creating a subfolder for these plots to avoid cluttering PROJECT_ROOT
                    output_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.FIGURE_SUBDIR,
                                           'context_transition_single_session')
                    os.makedirs(output_dir, exist_ok=True)

                    save_name = f"{animal}_{session_long_name}_{trans_type}_{branch}.png"
                    save_path = os.path.join(output_dir, save_name)

                    plt.savefig(save_path)
                    plt.close(fig)  # Close to free up memory during batch processing
                    print(f"Successfully saved: {save_name}")


def get_custom_palette(trans_type):
    """
    Creates a list of 6 colors following the user's specific gradient requirements.
    Low Block = Set2[0] (Greenish), High Block = Set2[1] (Salmon/Orange)
    """
    palette = sns.color_palette('Set2')
    green_base = palette[0]
    salmon_base = palette[1]

    if trans_type == 'LowToHigh':
        # Trials -2, -1 (Low): Light Green -> Dark Green
        # Trials 0, 1, 2, 3 (High): Dark Salmon -> Light Salmon
        colors = [
            sns.light_palette(green_base, n_colors=4)[1],  # Light Green (-2)
            green_base,  # Dark Green (-1)
            salmon_base,  # Dark Salmon (0)
            sns.light_palette(salmon_base, n_colors=5, reverse=True)[1],
            sns.light_palette(salmon_base, n_colors=5, reverse=True)[2],
            sns.light_palette(salmon_base, n_colors=5, reverse=True)[3]  # Light Salmon (3)
        ]
    else:  # HighToLow
        # Trials -2, -1 (High): Light Salmon -> Dark Salmon
        # Trials 0, 1, 2, 3 (Low): Dark Green -> Light Green
        colors = [
            sns.light_palette(salmon_base, n_colors=4)[1],  # Light Salmon (-2)
            salmon_base,  # Dark Salmon (-1)
            green_base,  # Dark Green (0)
            sns.light_palette(green_base, n_colors=5, reverse=True)[1],
            sns.light_palette(green_base, n_colors=5, reverse=True)[2],
            sns.light_palette(green_base, n_colors=5, reverse=True)[3]  # Light Green (3)
        ]
    return colors

def process_and_plot_averages(animal_ids):
    fs = 40  # Sampling frequency
    rel_trial_labels = [-2, -1, 0, 1, 2, 3]

    for animal in animal_ids:
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)  #
        if not os.path.exists(processed_dir): continue

        session_files = glob.glob(os.path.join(processed_dir, f"{animal}_*_zscore.parquet"))

        for z_file in session_files:
            session_id = os.path.basename(z_file).replace(f"{animal}_", "").replace("_zscore.parquet", "")

            # 1. Load Session Data
            zscore = load_session_dataframe(animal, 'zscore', session_long_name=session_id)
            pi_events = load_session_dataframe(animal, 'pi_events_processed', session_long_name=session_id)
            if zscore is None or pi_events is None: continue

            df_trials = get_trial_metadata(pi_events)
            branches = [c for c in zscore.columns if 'green' in c]

            for branch in branches:
                fig, (ax_lh, ax_hl) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
                fig.suptitle(f"{animal} | {session_id} | {branch}\nAverage Traces Around Transitions", fontsize=16)

                for trans_type, ax in zip(['LowToHigh', 'HighToLow'], [ax_lh, ax_hl]):
                    type_indices = df_trials[df_trials['transition_type'] == trans_type].index.tolist()

                    if not type_indices:
                        ax.text(0.5, 0.5, "No Transitions Found", ha='center')
                        continue

                    # Color palette for the 6 relative trials to show progression
                    colors = get_custom_palette(trans_type)
                    # Dictionary to hold lists of traces for each relative trial
                    trace_accumulator = {rel: [] for rel in rel_trial_labels}

                    for switch_idx in type_indices:
                        for rel in rel_trial_labels:
                            idx = switch_idx + rel
                            if 0 <= idx < len(df_trials):
                                t_info = df_trials.iloc[idx]
                                # Stitching logic for 'bg' port
                                trial_sig = zscore[(zscore['trial'] == t_info['trial']) & (zscore['port'] == 'bg')]

                                if not trial_sig.empty:
                                    # Ensure all traces are 6 seconds long (240 samples @ 40Hz)
                                    #
                                    sig_vals = trial_sig[branch].values[:240]
                                    if len(sig_vals) == 240:
                                        trace_accumulator[rel].append(sig_vals)

                    # 2. Average and Plot
                    for i, rel in enumerate(rel_trial_labels):
                        if trace_accumulator[rel]:
                            mean_trace = np.mean(trace_accumulator[rel], axis=0)
                            sem_trace = stats.sem(trace_accumulator[rel], axis=0)
                            time_axis = np.linspace(0, 6, 240)

                            ax.plot(time_axis, mean_trace, label=f"Trial {rel}", color=colors[i], lw=2)
                            ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                                            color=colors[i], alpha=0.1)

                    # Subplot Formatting
                    ax.set_title(f"Transition: {trans_type}")
                    ax.set_xlim(0, 6)  #
                    ax.set_xlabel("Time in Port (s)")
                    if ax == ax_lh: ax.set_ylabel("DA Z-score (Mean ± SEM)")

                    # Vertical gray dashed lines for reward timing
                    for tl in [1.25, 2.5, 3.75, 5.0]:
                        ax.axvline(tl, color='gray', linestyle='--', alpha=0.5, zorder=1)

                    ax.legend(fontsize='small', ncol=2)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                # 3. Save Figure
                save_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.FIGURE_SUBDIR,
                                                           'context_transition_single_session')
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"{animal}_{session_id}_{branch}_block_transition_average_traces.png"
                plt.savefig(os.path.join(save_dir, save_name))
                plt.close(fig)
                print(f"Saved Average Plot: {save_name}")



def quantify_trial_integrals(zscore, pi_events, fs=40):
    """
    Calculates integrals for each trial at specified timepoints.
    Keeps 'sec' naming but standardizes hemisphere names for QC compatibility.
    """
    # 1. Identify trials and their block types
    is_entry = (pi_events['key'] == 'trial') & (pi_events['value'] == 1) & pi_events['is_valid']
    df_trials = pi_events[is_entry][['trial', 'phase']].copy()
    df_trials.columns = ['trial', 'block']

    # 2. Setup quantification windows
    timepoints = [1.25, 2.5, 3.75, 5]
    dt = 1.0 / fs

    # Branches to analyze (hemispheres)
    branches = [c for c in zscore.columns if 'green' in c]

    session_results = []

    for _, row in df_trials.iterrows():
        trial_num = row['trial']
        # Stitch signal: filter for background port
        trial_sig_df = zscore[(zscore['trial'] == trial_num) & (zscore['port'] == 'bg')]

        if trial_sig_df.empty:
            continue

        for branch in branches:
            # FIX: Strip 'green_' to match qc_selections (e.g., 'left', 'right')
            hemi_name = branch.replace('green_', '')
            res = {'trial': trial_num, 'block': row['block'], 'hemisphere': hemi_name}
            sig = trial_sig_df[branch].values

            for tp in timepoints:
                # Calculate start and end indices relative to entry
                idx_start = int(tp * fs)
                idx_end = int((tp + 1.25) * fs)

                # Check if we have enough data for the window
                if idx_start < len(sig):
                    window_sig = sig[idx_start:min(idx_end, len(sig))]
                    # Numerical integration (Sum * dt)
                    integral = np.sum(window_sig) * dt
                    # Standardized naming as requested
                    res[f'{tp} sec'] = integral
                else:
                    res[f'{tp} sec'] = np.nan

            session_results.append(res)

    return pd.DataFrame(session_results)

def run_batch_quantification(animal_ids):
    # Requirement: Set2[0] for Low, Set2[1] for High
    palette = sns.color_palette('Set2')
    block_colors = {'0.4': palette[0], '0.8': palette[1]}

    for animal in animal_ids:
        # Build path to processed data
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            continue

        # Discover unique sessions
        session_files = glob.glob(os.path.join(processed_dir, f"{animal}_*_zscore.parquet"))

        for z_file in session_files:
            # Extract session timestamp for naming
            session_id = os.path.basename(z_file).replace(f"{animal}_", "").replace("_zscore.parquet", "")

            zscore = load_session_dataframe(animal, 'zscore', session_long_name=session_id)
            pi_events = load_session_dataframe(animal, 'pi_events_processed', session_long_name=session_id)
            if zscore is None or pi_events is None:
                continue

            # 1. Quantify and Save Parquet
            df_quant = quantify_trial_integrals(zscore, pi_events)
            save_path = os.path.join(processed_dir, f"{animal}_{session_id}_DA_integrals_vs_trial.parquet")
            df_quant.to_parquet(save_path, index=False)

            # 2. Plotting per Hemisphere
            for branch in df_quant['hemisphere'].unique():
                df_hemi = df_quant[df_quant['hemisphere'] == branch].sort_values('trial')

                fig, ax = plt.subplots(figsize=(15, 6))

                # Plot the trace for the 1.25s integral
                ax.plot(df_hemi['trial'], df_hemi['1.25 sec'], color='black', marker='o', markersize=3, lw=1)

                # Formatting: Title and labels with session info
                ax.set_title(f"{animal} | {session_id} | {branch}\nDopamine Integral [1.25s - 2.5s]", fontsize=14)
                ax.set_xlabel("Trial Number")
                ax.set_ylabel("DA Integral (Z*s)")

                # Requirement: Horizontal dashed line at y=0 and No Grids
                ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
                ax.grid(False)

                # Requirement: Background colored patches for blocks
                df_hemi['block_num'] = (df_hemi['block'] != df_hemi['block'].shift()).cumsum()
                for _, block_df in df_hemi.groupby('block_num'):
                    start_trial = block_df['trial'].min()
                    end_trial = block_df['trial'].max()
                    block_type = str(block_df['block'].iloc[0])
                    ax.axvspan(start_trial-0.49, end_trial+0.49, color=block_colors.get(block_type, 'gray'), alpha=0.2, zorder=0)

                plt.tight_layout()

                # Save Plot
                # save_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.FIGURE_SUBDIR,
                #                                            'context_transition_single_session')
                # os.makedirs(save_dir, exist_ok=True)
                # save_name = f"{animal}_{session_id}_{branch}_DA_integral_vs_trial.png"
                # plt.savefig(os.path.join(save_dir, save_name))
                plt.show()


def plot_da_integrals_evolution(df):
    """
        Plots the evolution of DA integrals around block transitions across sessions.
        Layout: 6 rows x 4 columns (2 animal-hemisphere combinations per row).
        Includes horizontal line at y=0 and shaded SEM error bands.
        """
    target_col = '1.25 sec'
    window_range = np.arange(-2, 4)  # trials -2, -1, 0, 1, 2, 3

    # 1. Ensure 'block' is numeric to allow .diff() calculation
    df = df.copy()
    df['block'] = pd.to_numeric(df['block'], errors='coerce')

    # Identify unique groups (Animal + Hemisphere)
    groups = df.groupby(['animal', 'hemisphere'])
    group_keys = list(groups.groups.keys())

    # Layout constants
    n_rows = 4
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 3.5 * n_rows), sharex=True)

    # Helper to plot mean + SEM
    def plot_with_sem(ax, traces_list, color):
        if not traces_list:
            return
        traces_array = np.array(traces_list)
        mean = np.nanmean(traces_array, axis=0)
        sem = np.nanstd(traces_array, axis=0) / np.sqrt(len(traces_array))

        ax.plot(window_range, mean, color=color, lw=2.5, zorder=10)
        ax.fill_between(window_range, mean - sem, mean + sem, color=color, alpha=0.2, zorder=9)

    # 2. Iterate through each group
    for i, key in enumerate(group_keys):
        if i >= 12:  # Capacity of 6x4 grid (2 plots per group)
            break

        row = i // 3
        col_offset = (i % 3) * 2

        ax_lh = axes[row, col_offset]
        ax_hl = axes[row, col_offset + 1]

        animal, hemi = key
        group_df = groups.get_group(key)

        sessions = sorted(group_df['session'].unique())
        colors = sns.color_palette("Greys", n_colors=len(sessions) + 3)[2:]

        lh_traces = []
        hl_traces = []

        for s_idx, session in enumerate(sessions):
            sess_data = group_df[group_df['session'] == session].sort_values('trial').copy()

            # Detect transitions
            sess_data['block_change'] = sess_data['block'].diff().fillna(0)
            transitions = sess_data[sess_data['block_change'] != 0].index

            for trans_idx in transitions:
                loc = sess_data.index.get_loc(trans_idx)
                if loc < 2 or loc + 4 > len(sess_data):
                    continue

                window = sess_data.iloc[loc - 2: loc + 4]
                trace_values = window[target_col].values
                transition_val = sess_data.loc[trans_idx, 'block_change']

                # Column 0: L->H, Column 1: H->L
                if transition_val > 0:
                    ax_lh.plot(window_range, trace_values, color=colors[s_idx], alpha=0.3, lw=0.5)
                    lh_traces.append(trace_values)
                else:
                    ax_hl.plot(window_range, trace_values, color=colors[s_idx], alpha=0.3, lw=0.5)
                    hl_traces.append(trace_values)

        # Plot averages and SEM
        plot_with_sem(ax_lh, lh_traces, 'red')
        plot_with_sem(ax_hl, hl_traces, 'blue')

        # Subplot Formatting
        for ax in [ax_lh, ax_hl]:
            ax.axvline(-0.5, color='gray', linestyle='--', alpha=1, lw=1)  # Transition line
            ax.axhline(0, color='black', linestyle='--', alpha=1, lw=0.8)  # Zero line
            ax.set_xticks(window_range)
            sns.despine(ax=ax)

        # Titles and labels
        ax_lh.set_title(f"{animal}-{hemi}\nLow $\\rightarrow$ High", fontsize=10, fontweight='bold')
        ax_hl.set_title(f"{animal}-{hemi}\nHigh $\\rightarrow$ Low", fontsize=10, fontweight='bold')

        if col_offset == 0:
            ax_lh.set_ylabel("DA Integral (Z*s)")

    # Hide unused subplots
    total_groups_plotted = min(len(group_keys), 12)
    for j in range(total_groups_plotted * 2, n_rows * n_cols):
        axes.flat[j].set_visible(False)

    # Set bottom labels
    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("Trials from transition")

    plt.tight_layout()

    # 6. Save the figure to the thesis directory
    # save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "block_transition_DA_integrals_evolution_ani-hemi.png")
    #
    # plt.savefig(save_path, dpi=300)
    # print(f"Successfully saved figure to: {save_path}")
    plt.show()

# Run for a specific session
if __name__ == "__main__":
    # plot_block_transitions("SZ036", "2024-01-11T16_25")
    animals = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    run_batch_transition_analysis(animals)
    # process_and_plot_averages(animals)
    # run_batch_quantification(animals)


    # Load data using the project's summary loader
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    # animal_ids=["SZ036"]
    master_df1 = load_dataframes_for_animal_summary(animal_ids, 'DA_integrals_vs_trial',
                                                                day_0='2023-11-30', hemisphere_qc=1,
                                                                file_format='parquet')

    animal_ids = ["RK007", "RK008"]
    master_df2 = load_dataframes_for_animal_summary(animal_ids, 'DA_integrals_vs_trial',
                                                                day_0='2025-06-17', hemisphere_qc=1,
                                                                file_format='parquet')
    df = pd.concat([master_df1, master_df2], ignore_index=True)

    if df.empty:
        print("Error: No data found for the specified animals.")
    plot_da_integrals_evolution(df)

