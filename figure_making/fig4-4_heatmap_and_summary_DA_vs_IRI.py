import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from scipy import stats

# Project modules
import config
import data_loader
import fig_dopamine
import helper


def resample_data_for_IRI_heatmap(zscore, zscore_branch, reward_df, cutoff_pre=-0.5, cutoff_post=2, bin_size_s=0.05):
    """
    Resamples data for a heatmap aligned by reward and categorized by IRI.
    Modified from fig_dopamine.resample_data_for_heatmap.
    """
    # Filter for valid trials with prior rewards
    valid_df = reward_df[
        reward_df['IRI_prior'].notna()
        & (reward_df['IRI_prior'] > 1)
        & (reward_df['IRI_post'] >= 0.6)].copy()

    # Define bins for IRI using quantiles for balanced groups
    tt1 = np.round(valid_df['IRI_prior'].quantile(0.33), 2)
    tt2 = np.round(valid_df['IRI_prior'].quantile(0.67), 2)
    # qt3 = np.round(valid_df['IRI_prior'].quantile(0.75), 1)
    bins = [1, tt1, tt2, np.inf]
    cat_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    cat_labels[-1] = f'>{bins[-2]}'

    valid_df['cat_code'] = pd.cut(valid_df['IRI_prior'], bins=bins, labels=False, right=False)
    sorted_df = valid_df.sort_values(by='IRI_prior').reset_index(drop=True)

    # 2. Setup Resampling
    original_t = zscore['time_recording'].values
    original_v = zscore[zscore_branch].values
    uniform_t = np.arange(cutoff_pre, cutoff_post + bin_size_s, bin_size_s)
    heatmap_matrix = np.full((len(sorted_df), len(uniform_t)), np.nan)

    for idx, row in sorted_df.iterrows():
        t_rew = row['reward_time']
        t_next = row['next_reward_time']

        # Calculate how long the "valid" window is after the reward
        # If there is no next reward (end of trial), use the full cutoff_post
        time_to_next = t_next - t_rew if pd.notna(t_next) else cutoff_post

        # Determine target timestamps for this specific row
        target_t = t_rew + uniform_t

        # Find window in source data
        idx_s = helper.find_closest_value(original_t, t_rew + cutoff_pre)
        idx_e = helper.find_closest_value(original_t, t_rew + cutoff_post)

        if (idx_e - idx_s) > 1:
            resampled = np.interp(target_t, original_t[idx_s:idx_e], original_v[idx_s:idx_e])

            # --- MASKING LOGIC ---
            # Create a mask: only keep values where time from reward is <= time_to_next
            # We also keep the 'pre' period (time < 0)
            mask = (uniform_t <= time_to_next)
            resampled[~mask] = np.nan

            heatmap_matrix[idx, :len(resampled)] = resampled

    return uniform_t, sorted_df['cat_code'].values, cat_labels, heatmap_matrix


def fige_DA_vs_IRI_binned(master_df, axes=None):
    """
    Population box plot of DA peak amplitudes binned by IRI.
    Modified from fig_dopamine.fige_DA_vs_NRI_v2.
    """
    data = master_df.copy()
    # Define bins specifically for IRI
    bins = [1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.7, 3.6, np.inf]
    bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    bin_labels[-1] = f'>{bins[-2]}'
    data['cat_code'] = pd.cut(data['IRI'], bins=bins, labels=bin_labels)

    summary_data = data.groupby(['animal', 'session', 'cat_code'], observed=True).agg(
        DA=('DA', 'mean')).reset_index()

    sns.boxplot(data=summary_data, x='cat_code', y='DA', linewidth=1, showfliers=False,
                notch=True, width=0.9, boxprops=dict(facecolor='lightgrey', alpha=0.4),
                medianprops={'linewidth': 2, 'color': 'black'}, ax=axes)

    animal_order = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043', 'RK007', 'RK008']
    sns.swarmplot(data=summary_data, x='cat_code', y='DA', hue='animal', hue_order=animal_order, size=2,
                  dodge=True, legend=False, ax=axes, linewidth=0.5, edgecolor='face')
    fill_alpha = 0.5
    for collection in axes.collections:
        face_colors = collection.get_facecolors()
        face_colors[:, 3] = fill_alpha
        collection.set_facecolors(face_colors)

    pivot_df = summary_data.pivot_table(index=['animal', 'session'],
                                        columns='cat_code', values='DA')

    y_max = summary_data['DA'].max()
    h = 0.1  # Height of the bracket
    y_bars = [4.2, 4.3, 4.3, 4.6, 4.7, 4.7, 4.8]
    y_bar_start = 4.2
    y_bar_current = y_bar_start
    inset = 0.05

    # Compare adjacent bins
    for i in range(len(bin_labels) - 1):
        bin1, bin2 = bin_labels[i], bin_labels[i + 1]

        # Get paired values
        pair_data = pivot_df[[bin1, bin2]].dropna()
        if len(pair_data) > 3:  # Ensure enough samples for t-test
            t_stat, p_val = stats.ttest_rel(pair_data[bin1], pair_data[bin2])

            # Significance markers
            if p_val < 0.001:
                sig = '***'
                y_stagger = 0.3
            elif p_val < 0.01:
                sig = '**'
                y_stagger = 0.2
            elif p_val < 0.05:
                sig = '*'
                y_stagger = 0.1
            else:
                sig = 'n.s.'
                y_stagger = 0

            # Draw brackets and text
            x1, x2 = i + inset, i + 1 - inset
            y_bar_current = y_bar_current + y_stagger
            y = y_bar_current
            axes.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c='black')
            axes.text((x1 + x2) * 0.5, y + h, sig, ha='center', va='bottom', color='black', fontsize=10)

    axes.set_yticks([1, 2, 3, 4, 5])
    axes.set_xlabel('Inter-Reward Interval (s)')
    axes.set_ylabel('DA Peak Amplitude')
    axes.spines[['top', 'right']].set_visible(False)


def plot_animal_sessions_IRI_heatmaps(animal_id, branch):
    """
    Generates and saves IRI-split heatmaps for every session of a given animal.
    Layout matches the top row of Figure 4-4.
    """
    # 1. Find all sessions for this animal
    processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PROCESSED_DATA_SUBDIR)
    # Search for zscore files to identify sessions
    zscore_files = glob.glob(os.path.join(processed_dir, f"*_zscore.parquet"))
    sessions = [os.path.basename(f).replace(f"{animal_id}_", "").split('_zscore')[0] for f in zscore_files]

    if not sessions:
        print(f"No sessions found for {animal_id} in {processed_dir}")
        return

    for session_name in sessions:
        print(f"Processing {animal_id} session: {session_name}...")

        # Load Data
        zscore = data_loader.load_session_dataframe(animal_id, 'zscore', session_long_name=session_name)
        reward_df = data_loader.load_session_dataframe(animal_id, 'expreward_df', session_long_name=session_name)

        if zscore is None or reward_df is None:
            continue

        # Prepare Figure (Matches Top Row of Fig 4-4)
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

        # Heatmap Group (Bars, Heatmap, Cbar)
        gs_left = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0],
                                                   width_ratios=[0.5, 20, 1], wspace=0.05)
        axes_heatmap = [
            fig.add_subplot(gs_left[0, 0]),
            fig.add_subplot(gs_left[0, 1]),
            fig.add_subplot(gs_left[0, 2])
        ]
        # Mean Trace Axis
        ax_mean = fig.add_subplot(gs[0, 1])
        axes_heatmap.append(ax_mean)

        # Plot IRI Split
        t_vec, codes, labels, mat = resample_data_for_IRI_heatmap(zscore, branch, reward_df)

        if mat is not None:
            fig_dopamine.plot_heatmap_and_mean_traces(
                t_vec, codes, labels, mat,
                palette='Blues_r', axes=axes_heatmap, legend_title='IRI (s)'
            )

            # Formatting
            plt.suptitle(f"Animal: {animal_id} | Session: {session_name} | Branch: {branch}",
                         fontsize=14, fontweight='bold', y=0.95)

            # Save to animal's figure directory
            # save_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.FIGURE_SUBDIR, 'IRI_heatmaps')
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"{session_name}_IRI_heatmap.png")
            # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            # plt.close()
        else:
            plt.close()


def setup_thesis_fig_4_4_layout():
    """Sets up the grid matching the DA vs NRI figure layout."""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.3)

    # Top Row: Heatmap Group
    gs_heatmap = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0],
                                                  width_ratios=[0.5, 20, 1], wspace=0.05)
    axes_heatmap = [
        fig.add_subplot(gs_heatmap[0, 0]),
        fig.add_subplot(gs_heatmap[0, 1]),
        fig.add_subplot(gs_heatmap[0, 2])
    ]
    ax_mean = fig.add_subplot(gs[0, 1])
    axes_heatmap.append(ax_mean)

    # Bottom Row: Population Box Plot
    ax_pop = fig.add_subplot(gs[1, :])
    return fig, axes_heatmap, ax_pop


def main():
    # 1. Load Data
    # session candidates
    # SZ036: '2024-01-02T20_06', '2024-01-03T16_13' (if use 2 decimals for tertiles), '2024-01-05T16_35' (remove rewards with IRI>1)
    # SZ037: '2023-12-13T11_37', '2024-01-14T19_57'
    # SZ038: '2023-12-28T17_36' (2 decimals)
    # SZ043: '2023-12-21T17_19' (2 decimals), '2024-01-02T21_09' (2 decimals)
    animal, session, hemi = 'SZ036', '2024-01-05T16_35', 'left'
    z_heat = data_loader.load_session_dataframe(animal, 'zscore', session_long_name=session)
    r_heat = data_loader.load_session_dataframe(animal, 'expreward_df', session_long_name=session)

    # Load Population Data
    df_sz = data_loader.load_dataframes_for_animal_summary(["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"],
                                                           'DA_vs_features', day_0='2023-11-30', hemisphere_qc=1)
    df_rk = data_loader.load_dataframes_for_animal_summary(["RK007", "RK008"],
                                                           'DA_vs_features', day_0='2025-06-17', hemisphere_qc=1)
    master_df = pd.concat([df_sz, df_rk], ignore_index=True)

    # 2. Plotting
    fig, axes_heat, ax_pop = setup_thesis_fig_4_4_layout()

    # Process and Plot Heatmap Group
    t_vec, codes, labels, mat = resample_data_for_IRI_heatmap(z_heat, f'green_{hemi}', r_heat)
    fig_dopamine.plot_heatmap_and_mean_traces(t_vec, codes, labels, mat, palette='Blues_r',
                                              axes=axes_heat, legend_title='IRI (s)')

    # Plot Population Box Plot
    fige_DA_vs_IRI_binned(master_df, axes=ax_pop)

    # Lettering
    fig.text(0.1, 0.88, 'a', fontsize=16, weight='bold', va='bottom')
    fig.text(0.1, 0.46, 'b', fontsize=16, weight='bold', va='bottom')

    # 3. Save
    save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR, 'fig_4-4_DA_vs_IRI.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
    # animal_id = 'SZ036'
    # branch = 'green_left'
    # plot_animal_sessions_IRI_heatmaps(animal_id, branch)
