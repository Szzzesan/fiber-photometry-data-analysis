import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import data_loader
from quality_control import qc_selections


def reprocess_nonreward_DA_standalone(animal_id, session_id, excl_start_rel, excl_end_rel):
    """
    Reprocesses non-reward DA traces by loading three distinct dataframes:
    zscore, expreward_df, and trial_df.
    """
    # 1. Load the three required processed dataframes
    zscore_df = data_loader.load_session_dataframe(animal_id, 'zscore', session_id=session_id)
    reward_df = data_loader.load_session_dataframe(animal_id, 'expreward_df', session_id=session_id)
    trial_df  = data_loader.load_session_dataframe(animal_id, 'trial_df', session_id=session_id)

    # Check for missing data
    if any(df is None for df in [zscore_df, reward_df, trial_df]):
        print(f"Skipping {animal_id} session {session_id}: Missing one or more input files.")
        return None

    # --- 2. Reward Exclusion Logic (using reward_df) ---
    reward_times = reward_df['reward_time'].dropna().to_numpy()
    zscore_times = zscore_df['time_recording'].to_numpy()

    keep_mask = np.full(zscore_times.shape, True)

    # Calculate exclusion windows
    exclusion_start = reward_times + excl_start_rel
    exclusion_end = reward_times + excl_end_rel

    for (start, end) in zip(exclusion_start, exclusion_end):
        is_within_vicinity = (zscore_times >= start) & (zscore_times < end)
        keep_mask[is_within_vicinity] = False

    # Extract the points outside reward windows
    nonreward_DA_vs_time = zscore_df[keep_mask].copy().reset_index(drop=True)
    nonreward_DA_vs_time.dropna(inplace=True, how='any')

    # --- 3. Mapping to Trials (using trial_df) ---
    port_entry = trial_df['exp_entry'].to_numpy()
    port_exit = trial_df['exp_exit'].to_numpy()
    trial_numbers = trial_df['trial'].to_numpy()
    block = trial_df['phase'].to_numpy()
    nonreward_times = nonreward_DA_vs_time['time_recording'].to_numpy()

    # Pre-allocate metadata columns
    nonreward_DA_vs_time['phase'] = np.nan
    nonreward_DA_vs_time['trial'] = np.nan
    nonreward_DA_vs_time['time_in_port'] = np.nan

    # Map each time point to its respective behavioral trial
    for i in range(len(port_entry)):
        entry_t = port_entry[i]
        exit_t = port_exit[i]

        # Mask for time points within this specific port entry duration
        is_in_current_trial = (nonreward_times >= entry_t) & (nonreward_times < exit_t)

        # Apply trial-specific info to those indices
        if np.any(is_in_current_trial):
            # Using .loc with boolean mask for explicit assignment
            nonreward_DA_vs_time.loc[is_in_current_trial, 'phase'] = block[i]
            nonreward_DA_vs_time.loc[is_in_current_trial, 'trial'] = trial_numbers[i]
            nonreward_DA_vs_time.loc[is_in_current_trial, 'time_in_port'] = nonreward_times[is_in_current_trial] - entry_t

    # Final cleanup: drop any data that didn't occur during a valid trial/port stay
    nonreward_DA_vs_time = nonreward_DA_vs_time.dropna(subset=['phase', 'trial', 'time_in_port']).reset_index(drop=True)

    return nonreward_DA_vs_time



def bin_nonreward_data(df, bin_size=0.5):
    """
    Bins the dataframe by time_in_port and calculates Mean/SEM per
    animal/hemisphere/phase/bin.
    """
    # 1. Create Bins
    max_time = 10.0  # Limit to 10s for cleaner plotting
    bins = np.arange(0, max_time + bin_size, bin_size)

    # Filter range first
    df_clean = df[df['time_in_port'] <= max_time].copy()

    # Assign bins
    df_clean['time_bin'] = pd.cut(df_clean['time_in_port'], bins=bins, include_lowest=True)

    # Calculate Bin Centers for plotting
    # (We map the CategoricalInterval to its mid point)
    df_clean['bin_center'] = df_clean['time_bin'].apply(lambda x: x.mid).astype(float)

    # 2. Groupby & Aggregate
    # Grouping by Animal + Hemisphere + Phase + TimeBin
    grouped = df_clean.groupby(['animal', 'hemisphere', 'phase', 'bin_center'])['DA']

    # Calculate Mean and SEM
    # Note: SEM here represents variability across trials within the bin
    agg_df = grouped.agg(['mean', 'sem', 'count']).reset_index()

    return agg_df

def plot_nonreward_traces_in_grid(binned_df, title_suffix=""):
    """
    Plots a 4x4 grid where each subplot is one animal-hemisphere pair.
    Traces are binned averages for 0.4 (Low) and 0.8 (High) blocks.
    """
    # 1. Identify all valid pairs in the data
    pairs = binned_df[['animal', 'hemisphere']].drop_duplicates().sort_values(['animal', 'hemisphere'])

    n_plots = len(pairs)
    if n_plots == 0:
        print("No data to plot.")
        return

    # Setup 4x4 Grid
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    # Colors
    color_map = {'0.4': sns.color_palette('Set2')[0], '0.8': sns.color_palette('Set2')[1]}

    # Y limits
    center = np.round(binned_df['mean'].median(), 1)


    # Loop through pairs
    for i, (idx, row) in enumerate(pairs.iterrows()):
        if i >= len(axes_flat): break

        ax = axes_flat[i]
        animal = row['animal']
        hemi = row['hemisphere']

        # Subset data
        subset = binned_df[(binned_df['animal'] == animal) & (binned_df['hemisphere'] == hemi)]

        # Plot each phase
        for phase in ['0.4', '0.8']:
            phase_data = subset[subset['phase'] == phase]
            if phase_data.empty: continue

            ax.errorbar(
                x=phase_data['bin_center'],
                y=phase_data['mean'],
                yerr=phase_data['sem'],
                color=color_map[phase],
                ecolor=color_map[phase],
                label=f'Block {phase}',
                fmt='o-', ms=3, capsize=2, elinewidth=1, alpha=0.8
            )

        ax.set_title(f"{animal} {hemi}", fontsize=9, y=0.98)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(center - 0.6, center + 0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Axis cleanup
        if i >= (rows - 1) * cols: ax.set_xlabel("Time (s)")
        if i % cols == 0: ax.set_ylabel("DA (z)")

        # Only show legend on first plot
        if i == 0: ax.legend(fontsize='x-small')

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(f"Non-Reward Traces (0.5s bins) {title_suffix}", fontsize=14)
    plt.show()

def plot_F0_traces_in_grid(binned_df, title_suffix=""):
    """
    Plots a grid for F0 baseline values.
    """
    pairs = binned_df[['animal', 'hemisphere']].drop_duplicates().sort_values(['animal', 'hemisphere'])
    if len(pairs) == 0: return

    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    color_map = {'0.4': sns.color_palette('Set2')[0], '0.8': sns.color_palette('Set2')[1]}

    # Calculate a dynamic center for the Y-axis based on F0 values
    center = np.round(binned_df['mean'].median(), 1)

    for i, (idx, row) in enumerate(pairs.iterrows()):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        animal, hemi = row['animal'], row['hemisphere']

        subset = binned_df[(binned_df['animal'] == animal) & (binned_df['hemisphere'] == hemi)]

        for phase in ['0.4', '0.8']:
            phase_data = subset[subset['phase'] == phase]
            if phase_data.empty: continue

            ax.errorbar(
                x=phase_data['bin_center'], y=phase_data['mean'], yerr=phase_data['sem'],
                color=color_map[phase], label=f'Block {phase}',
                fmt='o-', ms=3, capsize=2, elinewidth=1, alpha=0.8
            )

        ax.set_title(f"{animal} {hemi}", fontsize=9)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)  # Reference line at median
        ax.set_ylim(center - 0.6, center + 0.6)
        ax.spines[['top', 'right']].set_visible(False)

        if i >= (rows - 1) * cols: ax.set_xlabel("Time (s)")
        # Use LaTeX for the subscript '0'
        if i % cols == 0: ax.set_ylabel("$F_0$ (z)")

        if i == 0: ax.legend(fontsize='x-small')

    for j in range(i + 1, len(axes_flat)): axes_flat[j].axis('off')
    fig.suptitle(f"$F_0$ Baseline Traces {title_suffix}", fontsize=14)
    plt.show()

def main():
    mpl.rcParams['figure.dpi'] = 300
    all_sessions_pooled = []
    # tuple_list = [('SZ036', 15), ('SZ037', 25), ('SZ038', 29), ('SZ039', 20), ('SZ042', 20), ('SZ043', 18),
    #               ('RK007', 19), ('RK008', 11), ('RK009', 14), ('RK010', 13)]
    tuple_list = [('SZ036', 15), ('SZ037', 25), ('SZ038', 29), ('SZ039', 20), ('SZ042', 20), ('SZ043', 18),
                  ('RK007', 19), ('RK008', 11)]

    for (ani, total_sessions) in tqdm(tuple_list, desc="Processing All Animals"):
        for s_id in tqdm(range(total_sessions), desc=f"Sessions for {ani}", leave=False):

            # Extraction with new exclusion parameters
            # e.g., excluding 0s to 2s post-reward
            nr_df = reprocess_nonreward_DA_standalone(ani, s_id, excl_start_rel=0, excl_end_rel=0)

            if nr_df is not None and not nr_df.empty:
                # 1. Identify IDs (columns that stay fixed)
                id_cols = ['time_in_port', 'phase', 'trial', 'time_recording']

                # 2. Use wide_to_long to handle pairs: 'green_left/right' and 'F0_left/right'
                # We rename columns temporarily so the suffix is consistent (e.g., 'green_left', 'F0_left')
                # wide_to_long expects: [prefix][stub][suffix]
                nr_melted = pd.wide_to_long(
                    nr_df,
                    stubnames=['green', 'F0'],
                    i=id_cols,
                    j='hemisphere',
                    sep='_',
                    suffix='\\w+'
                ).reset_index()

                # 3. Rename 'green' to 'DA' to maintain compatibility with your binning function
                nr_melted = nr_melted.rename(columns={'green': 'DA'})
                nr_melted['animal'] = ani

                # 4. Apply Quality Control
                valid_hemis = qc_selections.get(ani, set())
                nr_melted = nr_melted[nr_melted['hemisphere'].isin(valid_hemis)].copy()

                all_sessions_pooled.append(nr_melted)

    # Combine everything and plot
    if all_sessions_pooled:
        master_nr_df = pd.concat(all_sessions_pooled, ignore_index=True)
        # binned_DA = bin_nonreward_data(master_nr_df, bin_size=0.5)
        binned_F0 = bin_nonreward_data(master_nr_df.rename(columns={'DA': 'temp', 'F0': 'DA'}), bin_size=0.5)
        # plot_nonreward_traces_in_grid(binned_DA, title_suffix="")
        plot_F0_traces_in_grid(binned_F0, title_suffix="")

if __name__ == '__main__':
    main()