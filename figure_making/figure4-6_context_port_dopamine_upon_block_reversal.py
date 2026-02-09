import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
import matplotlib.image as mpimg

import config
import quality_control as qc
from data_loader import load_session_dataframe, load_dataframes_for_animal_summary


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
            sns.light_palette(green_base, n_colors=4)[-2],  # Light Green (-2)
            green_base,  # Dark Green (-1)
            salmon_base,  # Dark Salmon (0)
            sns.light_palette(salmon_base, n_colors=6, reverse=True)[1],
            sns.light_palette(salmon_base, n_colors=6, reverse=True)[2],
            sns.light_palette(salmon_base, n_colors=6, reverse=True)[3]  # Light Salmon (3)
        ]
    else:  # HighToLow
        # Trials -2, -1 (High): Light Salmon -> Dark Salmon
        # Trials 0, 1, 2, 3 (Low): Dark Green -> Light Green
        colors = [
            sns.light_palette(salmon_base, n_colors=4)[-2],  # Light Salmon (-2)
            salmon_base,  # Dark Salmon (-1)
            green_base,  # Dark Green (0)
            sns.light_palette(green_base, n_colors=6, reverse=True)[1],
            sns.light_palette(green_base, n_colors=6, reverse=True)[2],
            sns.light_palette(green_base, n_colors=6, reverse=True)[3]  # Light Green (3)
        ]
    return colors


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


def average_traces_block_reversal_example_session(zscore, pi_events, trans_type='LowToHigh', fs=40, ax=None):
    """
    Plots average dopamine traces for the left hemisphere (green_left) for a
    specific transition type over a 2.5s window.
    """
    rel_trial_labels = [-2, -1, 0, 1, 2, 3]
    # Window length: 2.5 seconds * 40 Hz = 100 samples
    n_samples = int(2.5 * fs)
    branch = 'green_left'

    # 1. Get trial metadata and transitions
    df_trials = get_trial_metadata(pi_events)

    # 2. Setup Figure
    if ax is None:
        fig, axes = plt.subplots(1, 1, figsize=(5, 6))
        return_handle = True
    else:
        fig = None
        return_handle = False
    colors = get_custom_palette(trans_type)

    # Filter for the specific transition type indices
    type_indices = df_trials[df_trials['transition_type'] == trans_type].index.tolist()

    if not type_indices:
        print(f"No {trans_type} transitions found in this session.")
        return

    # Accumulate traces for each relative trial
    trace_accumulator = {rel: [] for rel in rel_trial_labels}

    for switch_idx in type_indices:
        for rel in rel_trial_labels:
            idx = switch_idx + rel
            if 0 <= idx < len(df_trials):
                t_info = df_trials.iloc[idx]
                # Filter signal for background port
                trial_sig = zscore[(zscore['trial'] == t_info['trial']) & (zscore['port'] == 'bg')]

                if not trial_sig.empty:
                    # Logic Minimal Change: slice to n_samples (100) instead of 240
                    sig_vals = trial_sig[branch].values[:n_samples]
                    if len(sig_vals) == n_samples:
                        trace_accumulator[rel].append(sig_vals)

    # 3. Average and Plot
    time_axis = np.linspace(0, 2.5, n_samples)
    for i, rel in enumerate(rel_trial_labels):
        if trace_accumulator[rel]:
            mean_trace = np.mean(trace_accumulator[rel], axis=0)
            sem_trace = stats.sem(trace_accumulator[rel], axis=0)

            ax.plot(time_axis, mean_trace, label=f"Trial {rel}", color=colors[i], lw=1.5)
            ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=colors[i], alpha=0.2)

    # Formatting
    if trans_type == 'LowToHigh':
        title = 'Low to High'
    elif trans_type == 'HighToLow':
        title = 'High to Low'
    ax.set_title(title)
    ax.set_xlim(0.75, 2.52)
    ax.set_ylim(-2.5, 5.5)
    ax.set_xticks([1, 1.25, 2.5])
    ax.set_xlabel("Time from Entry (s)")
    ax.set_ylabel("Mean DA (z-score)")

    # Vertical gray dashed lines for reward timing within the 2.5s window
    for tl in [1.25, 2.5]:
        ax.axvline(tl, color='gray', linestyle='--', alpha=0.5, zorder=1)

    ax.legend(title='Trial rel. Block Reversal',
              prop={'weight': 'normal', 'size': 'small'},
              title_fontproperties={'weight': 'normal', 'size': 'small'},
              handlelength=2, borderpad=0.4, ncol=2)
    sns.despine(ax=ax)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, ax


def add_schematics(axes, schematic_lh_name, schematic_hl_name):
    """
    Loads images from the figure save directory and places them in axes a and c.
    """
    # Use the same directory defined in your config for saving figures
    save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)

    # Path to your schematic files
    path_lh = os.path.join(save_dir, schematic_lh_name)
    path_hl = os.path.join(save_dir, schematic_hl_name)

    for path, ax_key in zip([path_lh, path_hl], ['schematic_lh', 'schematic_hl']):
        if os.path.exists(path):
            img = mpimg.imread(path)
            # Display the image
            axes[ax_key].imshow(img, aspect='auto')
            # Ensure the axis remains hidden
            axes[ax_key].axis('off')
        else:
            print(f"Warning: Schematic not found at {path}")


def vertical_traces_block_reversal_example(zscore, pi_events, trans_type='LowToHigh', axes_list=None, ylim=(-2.5, 4.0),
                                           fs=40):
    """Plots trial -2 to 3 in vertical subplots."""
    rel_trial_labels = [-1, 0, 1, 2, 3]  # Updated range
    n_samples = int(2.5 * fs)
    branch = 'green_left'
    df_trials = get_trial_metadata(pi_events)

    palette = sns.color_palette('Set2')
    low_color, high_color = palette[0], palette[1]

    type_indices = df_trials[df_trials['transition_type'] == trans_type].index.tolist()
    if not type_indices: return

    trace_accumulator = {rel: [] for rel in rel_trial_labels}
    rel_trial_blocks = {}

    for switch_idx in type_indices:
        for rel in rel_trial_labels:
            idx = switch_idx + rel
            if 0 <= idx < len(df_trials):
                t_info = df_trials.iloc[idx]
                if rel not in rel_trial_blocks:
                    rel_trial_blocks[rel] = t_info['block']

                trial_sig = zscore[(zscore['trial'] == t_info['trial']) & (zscore['port'] == 'bg')]
                if not trial_sig.empty:
                    sig_vals = trial_sig[branch].values[:n_samples]
                    if len(sig_vals) == n_samples:
                        trace_accumulator[rel].append(sig_vals)

    time_axis = np.linspace(0, 2.5, n_samples)
    for i, (rel, ax) in enumerate(zip(rel_trial_labels, axes_list)):
        if trace_accumulator[rel]:
            mean = np.mean(trace_accumulator[rel], axis=0)
            sem = stats.sem(trace_accumulator[rel], axis=0)
            block_val = rel_trial_blocks.get(rel, '0.4')
            color = low_color if block_val == '0.4' else high_color

            ax.plot(time_axis, mean, color=color, lw=1.5)
            ax.fill_between(time_axis, mean - sem, mean + sem, color=color, alpha=0.2)

            # (3) Horizontal dashed line at y=0
            ax.axhline(0, color='black', linestyle='--', alpha=0.3, lw=0.8)

            # Reward Markers (Blue/Grey dashed)
            if block_val == '0.8':
                for t in [1.25, 2.5]: ax.axvline(t, color='blue', linestyle='--', alpha=0.5, lw=1)
            else:
                ax.axvline(1.25, color='grey', linestyle='--', alpha=0.5, lw=1)
                ax.axvline(2.5, color='blue', linestyle='--', alpha=0.5, lw=1)

        # (1) Remove spines, ticks, and labels
        ax.set_ylim(ylim)
        ax.set_xlim(0.75, 2.6)  # Slightly wider for labels
        ax.axis('off')

        # Trial labels
        ax.text(0.75, ylim[1] - 0.2, f"Trial {rel}", fontsize=9, fontweight='bold', va='top')

        # (4) Scale bars (Top-most subplot only)
        if i == 0:
            # 0.25s horizontal bar, 1 z-score vertical bar
            # Coordinates in data units
            bar_x, bar_y = 2.2, ylim[1]+0
            ax.plot([bar_x, bar_x + 0.25], [bar_y-2, bar_y-2], color='black', lw=2)  # 0.25s
            ax.plot([bar_x, bar_x], [bar_y, bar_y-2], color='black', lw=2)  # 1 z-score
            ax.text(bar_x + 0.12, bar_y -3.1, "0.25 s", ha='center', fontsize=7)
            ax.text(bar_x - 0.01, bar_y - 0.9, "2 z-score", va='center', ha='right', fontsize=7, rotation=90)

        # Bottom-most subplot: Add the reward timing labels
        if i == 4:
            ax.text(1.25, ylim[0] - 0.75, '1.25 s', ha='center', fontsize=9)
            ax.text(2.5, ylim[0] - 0.75, '2.5 s', ha='center', fontsize=9)


def block_reversal_population_summary(df, transition_type='LowToHigh', axes=None):
    """
    Plots a population summary of DA integrals across block reversals.
    """
    target_col = '1.25 sec'
    df = df.copy()

    # CRITICAL: Convert string blocks to floats for math operations
    df['block'] = pd.to_numeric(df['block'], errors='coerce')

    # Define transition direction magnitude
    # Low (0.4) -> High (0.8) is +0.4; High (0.8) -> Low (0.4) is -0.4
    target_magnitude = 0.4 if transition_type == 'LowToHigh' else -0.4

    reversal_data = []

    # 1. Group by animal and session
    for (animal, session), sess_df in df.groupby(['animal', 'session']):
        # Average across hemispheres for each trial in this specific session
        sess_summary = sess_df.groupby('trial').agg({
            target_col: 'mean',
            'block': 'first'
        }).sort_index().reset_index()

        # Identify reversal points now that block is numeric
        sess_summary['block_change'] = sess_summary['block'].diff()
        reversal_indices = sess_summary[sess_summary['block_change'] == target_magnitude].index

        for rev_idx in reversal_indices:
            # Look ahead to find the next block change or the end of the session
            future_trials = sess_summary.iloc[rev_idx:]
            next_changes = future_trials[future_trials['block'].diff() != 0].index.tolist()

            # If a next change is found, that marks the boundary
            # (Note: index 0 of next_changes is the current reversal, so we look for index 1)
            end_idx = next_changes[1] if len(next_changes) > 1 else sess_summary.index[-1]

            # Slice from trial -2 up to the determined end (clipping at +7 for plot clarity)
            start_slice = max(0, rev_idx - 2)
            end_slice = min(end_idx + 1, rev_idx + 6)

            block_slice = sess_summary.loc[start_slice:end_slice - 1].copy()
            # Align trials so the first trial of the new block is index 0
            block_slice['rel_trial'] = np.arange(len(block_slice)) - (rev_idx - start_slice)
            block_slice['animal'] = animal
            block_slice['session'] = session
            reversal_data.append(block_slice)

    if not reversal_data:
        print(f"No {transition_type} transitions found.")
        return

    full_rev_df = pd.concat(reversal_data, ignore_index=True)

    # 2. Aggregation for Swarm: Session-level means per relative trial
    session_summary = full_rev_df.groupby(['animal', 'session', 'rel_trial'])[target_col].mean().reset_index()

    # 3. Plotting
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    # Grey Boxplot for Median and IQR
    sns.boxplot(data=session_summary, x='rel_trial', y=target_col,
                showfliers=False, width=0.8, color='lightgrey', notch=True, linewidth=1,
                boxprops=dict(alpha=0.4), medianprops={'linewidth': 2, 'color': 'black'}, ax=axes)

    # Swarmplot colored by Animal
    animal_order = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043', 'RK007', 'RK008']
    sns.swarmplot(data=session_summary, x='rel_trial', y=target_col,
                  hue='animal', hue_order=[a for a in animal_order if a in session_summary['animal'].unique()],
                  size=1.5, dodge=True, linewidth=0.5, edgecolor='face', legend=False, ax=axes)
    fill_alpha = 0.5
    for collection in axes.collections:
        face_colors = collection.get_facecolors()
        face_colors[:, 3] = fill_alpha
        collection.set_facecolors(face_colors)

    # Red Grand Average line across all animals/sessions
    grand_avg = session_summary.groupby('rel_trial')[target_col].mean()
    if transition_type == 'LowToHigh':
        line_color = 'darkred'
    else:
        line_color = 'navy'
    axes.plot(np.arange(len(grand_avg)), grand_avg.values,
              color=line_color, marker='o', lw=1.5, alpha=0.8, label='Population Mean', zorder=100)

    # Visual Guide Formatting
    if transition_type == 'LowToHigh':
        axes.set_ylim(-1.7, 3.3)
    else:
        axes.set_ylim(-2.5, 2.5)
    axes.axvline(1.5, color='black', linestyle='--', alpha=0.5)  # Line between Trial -1 and 0
    axes.axhline(0, color='black', linestyle='-', alpha=0.2)
    if return_handle:
        axes.set_title(f"Dopamine Integrals: {transition_type} Population Summary", fontweight='bold')
    axes.set_xlabel("Trial Relative to Block Reversal")
    axes.set_ylabel("DA Integral (Z*s)")
    # axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Animal")
    sns.despine()

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def add_block_indicators(ax, transition_type):
    """
    Adds horizontal block bars at the top of the population plots.
    - LowToHigh: Low color on left, High color on right.
    - HighToLow: High color on left, Low color on right.
    """
    palette = sns.color_palette('Set2')
    low_color, high_color = palette[0], palette[1]

    # Define colors based on transition
    if transition_type == 'LowToHigh':
        left_color, right_color = low_color, high_color
        left_label, right_label = 'Low', 'High'
    else:
        left_color, right_color = high_color, low_color
        left_label, right_label = 'High', 'Low'

    # Get current y-limits to place the bars at the very top
    y_min, y_max = ax.get_ylim()
    # Position the bar in a thin strip at the top (e.g., top 5% of the plot)
    bar_height = (y_max - y_min) * 0.05
    bar_y_bottom = y_max - bar_height

    # Block Boundary is Trial -0.5 (between rel_trial -1 and 0)
    # Note: In the categorical swarmplot, Trial -1 is at index 1 and Trial 0 is at index 2
    # So the boundary is at x = 1.5
    boundary = 1.5

    # 1. Draw the Bars using axvspan
    # Left Bar (Trial -2 to -0.5)
    ax.axvspan(-0.5, boundary, ymin=0.95, ymax=1.0, color=left_color, alpha=0.8, zorder=0)
    # Right Bar (Trial -0.5 to end)
    ax.axvspan(boundary, 7.5, ymin=0.95, ymax=1.0, color=right_color, alpha=0.8, zorder=0)

    # 2. Add 'Low' / 'High' text overlaid on the bars
    # Using transform=ax.get_xaxis_transform() allows us to use data-x and axes-y coordinates
    text_y = 0.97  # Vertical center of the 0.95-1.0 span
    ax.text(0.5, text_y, left_label, transform=ax.get_xaxis_transform(),
            ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    ax.text(4.5, text_y, right_label, transform=ax.get_xaxis_transform(),
            ha='center', va='center', fontweight='bold', color='white', fontsize=9)


def setup_axes():
    fig = plt.figure(figsize=(12, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 3], figure=fig)
    axes = {}

    # Updated Mapping: HL in row 0, LH in row 1
    layout = [
        ('trace_hl', 0, 0, 'a'),
        ('pop_hl', 0, 1, 'b'),
        ('trace_lh', 1, 0, 'c'),
        ('pop_lh', 1, 1, 'd')
    ]

    for key, row, col, letter in layout:
        ax = fig.add_subplot(gs[row, col])
        axes[key] = ax
        ax.text(0.0, 1.0, letter, transform=(
                ax.transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
                fontsize=16, va='bottom', fontfamily='sans-serif', weight='bold')

    return fig, axes


def setup_axes_v2():
    """
    Sets up a composite figure:
    - Left Column: Nested 6x1 grids for individual trial traces.
    - Right Column: Single axes for population summaries.
    - Ratio: 2:3 width ratio; High-to-Low on top, Low-to-High on bottom.
    """
    fig = plt.figure(figsize=(12, 12), dpi=300)
    # Increase hspace slightly to give the nested grids room to breathe
    gs_main = gridspec.GridSpec(2, 2, width_ratios=[2, 3], figure=fig, hspace=0.4, wspace=0.18)

    axes = {}
    layout_info = [
        ('lh', 0, 'a', 'b'),  # Row 0
        ('hl', 1, 'c', 'd')  # Row 1
    ]

    for prefix, row_idx, let_trace, let_pop in layout_info:
        # Create population axis first to act as the vertical anchor
        ax_pop = fig.add_subplot(gs_main[row_idx, 1])
        axes[f'pop_{prefix}'] = ax_pop

        # Use 6 rows: index 0 is Trial -1, index 1 is the SPACER, 2-5 are Trials 0-3
        # height_ratios: [Trial-1, GAP, T0, T1, T2, T3]
        gs_nested = gs_main[row_idx, 0].subgridspec(6, 1, hspace=0.1,
                                                    height_ratios=[1, 0.2, 1, 1, 1, 1])

        trace_axes_list = []
        grid_indices = [0, 2, 3, 4, 5]  # Skipping index 1 to create the gap

        for i, g_idx in enumerate(grid_indices):
            ax = fig.add_subplot(gs_nested[g_idx, 0])
            trace_axes_list.append(ax)
            ax.axis('off')  # Cleaner for vertical stacks

        axes[f'trace_{prefix}'] = trace_axes_list

    # Re-apply letters manually for clarity in nested structure
    for key, letter in [('trace_lh', 'a'), ('pop_lh', 'b'), ('trace_hl', 'c'), ('pop_hl', 'd')]:
        target = axes[key][0] if isinstance(axes[key], list) else axes[key]
        target.text(0.0, 1.0, letter, transform=(
                target.transAxes + ScaledTranslation(-25 / 72, +10 / 72, fig.dpi_scale_trans)),
                    fontsize=16, fontweight='bold', va='bottom', fontfamily='sans-serif')

    return fig, axes

def setup_axes_v3():
    """
        Sets up a 3-column composite figure (2:2:3 ratio):
        - Col 0 & 1: Schematic (top) and Trace Stack (bottom).
        - Col 2: Population LH (top) and Population HL (bottom).
        Lettering: a-f sequence.
        """
    fig = plt.figure(figsize=(12, 8), dpi=300)

    # 1. Primary Grid: Two main blocks
    # Block 1 (Cols 0 & 1 combined): Width 4
    # Block 2 (Col 2): Width 3
    # wspace here defines the WIDE gap between the traces and the population
    gs_base = gridspec.GridSpec(1, 2, width_ratios=[4, 3], figure=fig, wspace=0.2)

    # 2. Split the first block into two columns for LH and HL
    # wspace here defines the TIGHT gap between the two trace columns
    gs_trace_cols = gs_base[0, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.2)

    # 3. Split the second block into two rows for Population
    gs_pop_col = gs_base[0, 1].subgridspec(2, 1, hspace=0.3)

    axes = {}

    # Setup Population Summaries (Right-most column)
    axes['pop_lh'] = fig.add_subplot(gs_pop_col[0, 0])
    axes['pop_hl'] = fig.add_subplot(gs_pop_col[1, 0])

    # Setup Traces (Left and Middle columns)
    col_configs = [(0, 'lh'), (1, 'hl')]  # LH in Col 0, HL in Col 1

    for col_idx, prefix in col_configs:
        # Each trace column has a 1:5 height ratio for schematic vs. traces
        gs_col_internal = gs_trace_cols[0, col_idx].subgridspec(2, 1, height_ratios=[1, 5], hspace=0.2)

        # Schematic Anchor
        ax_sch = fig.add_subplot(gs_col_internal[0, 0])
        ax_sch.axis('off')
        axes[f'schematic_{prefix}'] = ax_sch

        # Trace Stack (6 slots for Trial -1 vs Trial 0 gap)
        gs_stack = gs_col_internal[1, 0].subgridspec(6, 1, hspace=0.15, height_ratios=[1, 0.4, 1, 1, 1, 1])

        trace_list = []
        for i, g_idx in enumerate([0, 2, 3, 4, 5]):
            ax = fig.add_subplot(gs_stack[g_idx, 0])
            ax.axis('off')
            trace_list.append(ax)
        axes[f'trace_{prefix}'] = trace_list

    # 4. Lettering a-f
    letter_mapping = [
        (axes['schematic_lh'], 'a'), (axes['trace_lh'][0], 'b'),
        (axes['schematic_hl'], 'c'), (axes['trace_hl'][0], 'd'),
        (axes['pop_lh'], 'e'), (axes['pop_hl'], 'f')
    ]

    for ax, let in letter_mapping:
        ax.text(0.0, 1.0, let, transform=(
                ax.transAxes + ScaledTranslation(-25 / 72, +10 / 72, fig.dpi_scale_trans)),
                fontsize=16, fontweight='bold', va='bottom')

    return fig, axes

def main():
    fig, axes = setup_axes_v3()

    # --- Example Session Traces ---
    animal_str = 'SZ036'
    session_name = '2024-01-13T20_50'
    zscore = load_session_dataframe(animal_str, 'zscore', session_long_name=session_name,
                                    file_format='parquet')
    pi_events = load_session_dataframe(animal_str, 'pi_events_processed', session_long_name=session_name,
                                       file_format='parquet')
    # average_traces_block_reversal_example_session(zscore, pi_events, trans_type='LowToHigh', fs=40, ax=axes['trace_lh'])
    # average_traces_block_reversal_example_session(zscore, pi_events, trans_type='HighToLow', fs=40, ax=axes['trace_hl'])
    vertical_traces_block_reversal_example(zscore, pi_events, 'LowToHigh', axes_list=axes['trace_lh'], ylim=(-1.2, 5.5))
    vertical_traces_block_reversal_example(zscore, pi_events, 'HighToLow', axes_list=axes['trace_hl'], ylim=(-2.7, 4.0))

    # --- Summary Plot ---
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
    block_reversal_population_summary(df, transition_type='LowToHigh', axes=axes['pop_lh'])
    block_reversal_population_summary(df, transition_type='HighToLow', axes=axes['pop_hl'])

    # Add the new Block Indicator Bars
    add_block_indicators(axes['pop_lh'], 'LowToHigh')
    add_block_indicators(axes['pop_hl'], 'HighToLow')

    for ax_key in ['pop_lh', 'pop_hl']:
        curr_y = axes[ax_key].get_ylim()
        axes[ax_key].set_ylim(curr_y[0], curr_y[1] * 1.1)

    # --- Refinement: Remove redundant Row 0 X-labels ---
    axes['pop_lh'].set_xlabel('')

    # --- Manual Labelling ---
    # Add time labels to bottom of columns 0 and 1
    for prefix in ['lh', 'hl']:
        last_ax = axes[f'trace_{prefix}'][-1]
        last_ax.text(0.5, -0.35, 'Time since Entering Context Port (s)',
                     transform=last_ax.transAxes, ha='center', va='top', fontsize=8)
    # last_ax_hl = axes['trace_hl'][-1]
    # last_ax_hl.text(0.5, -0.3, 'Time since Entering Context Port (s)',
    #                 transform=last_ax_hl.transAxes,
    #                 ha='center', va='bottom', fontweight='normal')

    # --- Layout Refinement ---
    # Tighten vertical spacing since middle labels are gone
    plt.tight_layout()

    # --- Save Figure ---
    # Ensure the directory exists in your config-defined path
    save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "fig_4-6_DA_Block_Transitions_Composite_incomplete.png")

    # Save at 300 DPI for high-quality printing
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Composite figure successfully saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    main()
