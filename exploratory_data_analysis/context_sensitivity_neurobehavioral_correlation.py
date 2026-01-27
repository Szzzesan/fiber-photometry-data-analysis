import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from lifelines.statistics import logrank_test  # Ensure lifelines is installed

import config
from figure_making.data_loader import load_dataframes_for_animal_summary

# Set aesthetic context for publication-quality plots
sns.set_context("talk")
sns.set_style("white")
colors = sns.color_palette('Set2')  # Set2[0] = Low (0.4), Set2[1] = High (0.8)


def run_context_sensitivity_analysis(side='both'):
    if not side in ['ipsi', 'contra']:
        animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
        day_0 = '2023-11-30'

        # 1. Load Neural Data (DA Peaks)
        print("Loading neural data...")
        da_df1 = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)

        # 2. Load Behavioral Data (Trial Summaries)
        print("Loading behavioral data...")
        trial_df1 = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)

        animal_ids = ["RK007", "RK008"]
        day_0 = '2025-06-17'

        # 1. Load Neural Data (DA Peaks)
        print("Loading neural data...")
        da_df2 = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)

        # 2. Load Behavioral Data (Trial Summaries)
        print("Loading behavioral data...")
        trial_df2 = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)

        da_df = pd.concat([da_df1, da_df2], ignore_index=True)
        trial_df = pd.concat([trial_df1, trial_df2], ignore_index=True)

    else:
        animal_ids = ["SZ036", "SZ037", "SZ038"]
        day_0 = '2023-11-30'
        # 1. Load Neural Data (DA Peaks)
        print("Loading neural data...")
        da_df = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)

        # 2. Load Behavioral Data (Trial Summaries)
        print("Loading behavioral data...")
        trial_df = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)

    if da_df.empty or trial_df.empty:
        print("Data loading failed. Please check paths and animal IDs.")
        return

    # --- Step 1: Process Behavioral Data ---
    # Duration in Investment Port = Exit Time - Entry Time
    trial_df['stay_duration'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df = trial_df.dropna(subset=['stay_duration', 'phase'])

    behavior_results = []
    for (animal, session), group in trial_df.groupby(['animal', 'session']):
        low_stay = group[group['phase'] == '0.4']['stay_duration']
        high_stay = group[group['phase'] == '0.8']['stay_duration']

        if len(low_stay) > 5 and len(high_stay) > 5:
            diff_med = high_stay.median() - low_stay.median()

            # Log-rank test for survival curve difference
            try:
                lr_res = logrank_test(low_stay, high_stay)
                lr_stat = lr_res.test_statistic
                lr_p = lr_res.p_value
            except:
                lr_stat, lr_p = np.nan, np.nan

            behavior_results.append({
                'animal': animal, 'session': session,
                'diff_behavior': diff_med, 'logrank_stat': lr_stat
            })

    behavior_sessions = pd.DataFrame(behavior_results)

    # --- Step 2: Neural Difference (DA Peaks) ---
    # Filter by side if requested
    if side in ['ipsi', 'contra']:
        da_filtered = da_df[da_df['side_relative'] == side].copy()
    else:
        da_filtered = da_df.copy()  # 'both' averages across all rows per animal/session/block

    # Group by animal/session/block and calculate mean DA peak
    # If side was 'both', this averages across all available hemispheres
    neural_summary = da_filtered.groupby(['animal', 'session', 'block'])['DA'].mean().unstack('block').reset_index()

    # Neural Metric: High (0.8) - Low (0.4)
    neural_summary['diff_neural'] = neural_summary['0.8'] - neural_summary['0.4']
    df_neural = neural_summary[['animal', 'session', 'diff_neural']].dropna()

    # --- Step 3: Merge and Lagged Prep ---
    df_master = pd.merge(behavior_sessions, df_neural, on=['animal', 'session'], how='inner')
    df_master = df_master.sort_values(['animal', 'session'])

    # Create Lagged Metrics
    df_master['behav_prev'] = df_master.groupby('animal')['diff_behavior'].shift(1)
    df_master['neural_prev'] = df_master.groupby('animal')['diff_neural'].shift(1)

    # --- Step 4: Animal Level Aggregation ---
    df_animal = df_master.groupby('animal')[['diff_behavior', 'diff_neural']].mean().reset_index()

    # --- Step 5: Plotting ---
    save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    os.makedirs(save_path, exist_ok=True)
    suffix = f"_{side}"

    def plot_and_save(data, x, y, title, fname, save=0):
        plt.figure(figsize=(10, 10))
        data = data.dropna(subset=[x, y])
        sns.regplot(data=data, x=x, y=y, scatter=False, color='gray', line_kws={'alpha': 0.5})
        sns.scatterplot(data=data, x=x, y=y, hue='animal', s=100)

        rho, p = spearmanr(data[x], data[y])
        plt.title(f"{title} ({side})\n$\\rho$ = {rho:.3f}, p = {p:.3f}")
        plt.xlabel("$\\Delta$ Stay Duration (H - L)")
        plt.ylabel("$\\Delta$ DA Peak (H - L)")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, f"{fname}{suffix}.png"))
        plt.show()

    # Execute Plots
    plot_and_save(df_animal, 'diff_behavior', 'diff_neural', 'Animal Level', 'corr_animal')
    plot_and_save(df_master, 'diff_behavior', 'diff_neural', 'Session Level', 'corr_session')
    plot_and_save(df_master, 'behav_prev', 'diff_neural', 'Lag: Behav(i-1) vs Neural(i)', 'corr_behav_precede_neural')
    plot_and_save(df_master, 'diff_behavior', 'neural_prev', 'Lag: Behav(i) vs Neural(i-1)', 'corr_neural_precede_behav')

    # print(f"Analysis for '{side}' completed. Figures saved to {save_path}.")


def run_binned_context_analysis(side='both'):
    """
    Analyzes correlations between behavior and DA peaks, binned by reward delivery time (NRI).

    Args:
        side (str): 'both', 'ipsi', or 'contra'.
    """
    # Define animal cohorts based on side
    # if not side in ['ipsi', 'contra']:
    #     animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    #     day_0 = '2023-11-30'
    #
    #     # 1. Load Neural Data (DA Peaks)
    #     print("Loading neural data...")
    #     da_df1 = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)
    #
    #     # 2. Load Behavioral Data (Trial Summaries)
    #     print("Loading behavioral data...")
    #     trial_df1 = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)
    #
    #     animal_ids = ["RK007", "RK008"]
    #     day_0 = '2025-06-17'
    #
    #     # 1. Load Neural Data (DA Peaks)
    #     print("Loading neural data...")
    #     da_df2 = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)
    #
    #     # 2. Load Behavioral Data (Trial Summaries)
    #     print("Loading behavioral data...")
    #     trial_df2 = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)
    #
    #     da_df = pd.concat([da_df1, da_df2], ignore_index=True)
    #     trial_df = pd.concat([trial_df1, trial_df2], ignore_index=True)
    #
    # else:
    #     animal_ids = ["SZ036", "SZ037", "SZ038"]
    #     day_0 = '2023-11-30'
    #     # 1. Load Neural Data (DA Peaks)
    #     print("Loading neural data...")
    #     da_df = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)
    #
    #     # 2. Load Behavioral Data (Trial Summaries)
    #     print("Loading behavioral data...")
    #     trial_df = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)

    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    day_0 = '2023-11-30'

    # 1. Load Neural Data (DA Peaks)
    print("Loading neural data...")
    da_df1 = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)

    # 2. Load Behavioral Data (Trial Summaries)
    print("Loading behavioral data...")
    trial_df1 = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)

    animal_ids = ["RK007", "RK008"]
    day_0 = '2025-06-17'

    # 1. Load Neural Data (DA Peaks)
    print("Loading neural data...")
    da_df2 = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0=day_0, hemisphere_qc=1)

    # 2. Load Behavioral Data (Trial Summaries)
    print("Loading behavioral data...")
    trial_df2 = load_dataframes_for_animal_summary(animal_ids, 'trial_df', day_0=day_0, hemisphere_qc=0)

    da_df = pd.concat([da_df1, da_df2], ignore_index=True)
    trial_df = pd.concat([trial_df1, trial_df2], ignore_index=True)

    if da_df.empty or trial_df.empty:
        print("Data loading failed. Please check paths and animal IDs.")
        return

    # --- Step 1: Behavioral Difference (Stay Duration) ---
    trial_df['stay_duration'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df = trial_df.dropna(subset=['stay_duration', 'phase'])

    behav_list = []
    for (animal, session), group in trial_df.groupby(['animal', 'session']):
        low = group[group['phase'] == '0.4']['stay_duration']
        high = group[group['phase'] == '0.8']['stay_duration']
        if len(low) > 5 and len(high) > 5:
            behav_list.append({
                'animal': animal, 'session': session,
                'diff_behavior': high.median() - low.median()
            })
    df_behav = pd.DataFrame(behav_list)

    # --- Step 2: Neural Binning and Difference ---
    # Define bins for NRI (Time of reward delivery relative to entry)
    bins = [0, 2, 4, 6, 8]
    labels = ['(0, 2]', '(2, 4]', '(4, 6]', '(6, 8]']
    da_df['NRI_bin'] = pd.cut(da_df['NRI'], bins=bins, labels=labels)

    # Filter by side
    if side in ['ipsi', 'contra']:
        da_filtered = da_df[da_df['side_relative'] == side].copy()
    else:
        da_filtered = da_df.copy()

    # Calculate mean DA per animal, session, block, AND NRI bin
    # Use observed=True for categorical grouping
    neural_summary = da_filtered.groupby(['animal', 'session', 'block', 'NRI_bin'], observed=True)['DA'].mean()
    neural_summary = neural_summary.unstack('block').reset_index()

    # Neural Metric: High (0.8) - Low (0.4)
    neural_summary['diff_neural'] = neural_summary['0.8'] - neural_summary['0.4']
    df_neural = neural_summary.dropna(subset=['diff_neural'])

    # --- Step 3: Plotting Function ---
    save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    os.makedirs(save_path, exist_ok=True)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")

    def plot_binned_correlation(master_data, level='session', fname_prefix='corr'):
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
        unique_animals = sorted(master_data['animal'].unique())
        palette = sns.color_palette('Set1', n_colors=len(unique_animals))
        animal_colors = dict(zip(unique_animals, palette))

        for i, bin_label in enumerate(labels):
            ax = axes[i]
            bin_data = master_data[master_data['NRI_bin'] == bin_label].copy()

            if bin_data.empty:
                ax.set_title(f"Bin: {bin_label} s - No Data")
                continue

            x_col = 'diff_behavior'
            y_col = 'diff_neural'
            if 'lag_behav' in fname_prefix: x_col = 'behav_prev'
            if 'lag_neural' in fname_prefix: y_col = 'neural_prev'

            if level == 'animal':
                # ORIGINAL LOGIC: One global regression line for the cohort
                sns.regplot(data=bin_data, x=x_col, y=y_col, scatter=False, color='gray',
                            ax=ax, line_kws={'ls': '--', 'alpha': 0.5})
                sns.scatterplot(data=bin_data, x=x_col, y=y_col, hue='animal', s=120,
                                ax=ax, palette='Set1', legend=(i==0))

                rho, p = spearmanr(bin_data[x_col], bin_data[y_col])
                ax.set_title(f"Animal Level {side} Side: {bin_label} s Window (Global $\\rho$ = {rho:.3f}, p = {p:.3f})")

            else:
                # NEW LOGIC: Individual regression lines for each animal (Session/Lagged)
                all_rhos = []
                stats_text = []

                for animal, color in animal_colors.items():
                    animal_data = bin_data[bin_data['animal'] == animal]
                    # Only plot regression if we have enough sessions (e.g., > 3)
                    if len(animal_data) > 3:
                        sns.regplot(data=animal_data, x=x_col, y=y_col, scatter=True,
                                    ax=ax, color=color, label=animal if i == 0 else None,
                                    scatter_kws={'s': 60, 'alpha': 0.7},
                                    line_kws={'lw': 2, 'alpha': 0.8})

                        rho, p = spearmanr(animal_data[x_col], animal_data[y_col])
                        all_rhos.append(rho)
                        stats_text.append(f"{animal}: $\\rho$={rho:.2f}, p={p:.2f}")
                    else:
                        # Just scatter the points if not enough for regression
                        sns.scatterplot(data=animal_data, x=x_col, y=y_col, color=color, ax=ax, s=60)

                mean_rho = np.mean(all_rhos) if all_rhos else 0
                ax.set_title(f"NRI Window: {bin_label} s (Mean $\\rho$ = {mean_rho:.3f})")
                ax.text(0.02, 0.98, "\n".join(stats_text), transform=ax.transAxes,
                        verticalalignment='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

            ax.set_ylabel("$\\Delta$ DA Peak (H-L)")
            if i == 0:
                ax.legend(bbox_to_anchor=(0.9, 1), loc='upper left', title="Animal")

        plt.xlabel("$\\Delta$ Median Stay Duration (H-L)")
        plt.tight_layout()
        # plt.savefig(os.path.join(save_path, f"{fname_prefix}_{side}_binned.png"))
        plt.show()

    # --- Step 4: Data Merging and Lag Calculation ---
    # Merge behavior and neural (binned)
    df_master = pd.merge(df_behav, df_neural, on=['animal', 'session'], how='inner')
    df_master = df_master.sort_values(['animal', 'session'])

    # Lagged Behavior: Previous behavior vs current binned neural
    df_master['behav_prev'] = df_master.groupby(['animal', 'NRI_bin'], observed=True)['diff_behavior'].shift(1)
    df_master['neural_prev'] = df_master.groupby(['animal', 'NRI_bin'], observed=True)['diff_neural'].shift(1)

    # Animal Level: Average across sessions within each NRI bin
    df_animal = df_master.groupby(['animal', 'NRI_bin'], observed=True)[
        ['diff_behavior', 'diff_neural']].mean().reset_index()

    # --- Step 5: Execute Plots ---
    # 1. Animal Level
    plot_binned_correlation(df_animal, level='animal', fname_prefix='corr_animal')
    # # 2. Session Level
    # plot_binned_correlation(df_master, level='session', fname_prefix='corr_session')
    # # 3. Lagged: Behavior(i-1) vs Neural(i)
    # plot_binned_correlation(df_master, level='session', fname_prefix='corr_lag_behav')
    # # 4. Lagged: Behavior (i) vs Neural (i-1)
    # plot_binned_correlation(df_master, level='session', fname_prefix='corr_lag_neural')

    print(f"Binned analysis for '{side}' complete. Check {save_path}")


def plot_animal_hemis_context_sensitivity_regressions(window_width=2.0):
    # --- Step 1: Data Loading (SZ and RK cohorts) ---
    animal_ids1 = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    day_0_1 = '2023-11-30'
    da_df1 = load_dataframes_for_animal_summary(animal_ids1, 'DA_vs_features', day_0=day_0_1, hemisphere_qc=1)
    trial_df1 = load_dataframes_for_animal_summary(animal_ids1, 'trial_df', day_0=day_0_1, hemisphere_qc=0)

    animal_ids2 = ["RK007", "RK008"]
    day_0_2 = '2025-06-17'
    da_df2 = load_dataframes_for_animal_summary(animal_ids2, 'DA_vs_features', day_0=day_0_2, hemisphere_qc=1)
    trial_df2 = load_dataframes_for_animal_summary(animal_ids2, 'trial_df', day_0=day_0_2, hemisphere_qc=0)

    da_df = pd.concat([da_df1, da_df2], ignore_index=True)
    trial_df = pd.concat([trial_df1, trial_df2], ignore_index=True)

    # --- Step 2: Behavioral Difference (H - L) ---
    trial_df['stay_duration'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df = trial_df.dropna(subset=['stay_duration', 'phase'])

    behav_list = []
    for (animal, session), group in trial_df.groupby(['animal', 'session']):
        low = group[group['phase'] == '0.4']['stay_duration']
        high = group[group['phase'] == '0.8']['stay_duration']
        if len(low) > 5 and len(high) > 5:
            behav_list.append({
                'animal': animal, 'session': session,
                'diff_behavior': high.median() - low.median()
            })
    df_behav = pd.DataFrame(behav_list)

    # --- Step 3: NRI Binning Logic ---
    max_fixed = 8.0
    bins = list(np.arange(0, max_fixed + window_width, window_width))
    if bins[-1] > max_fixed: bins = bins[:-1]
    bins.append(np.inf)

    labels = []
    for i in range(len(bins) - 1):
        label = f"({bins[i]}, {bins[i + 1]}]" if bins[i + 1] != np.inf else f"({bins[i]}, inf]"
        labels.append(label)
    da_df['NRI_bin'] = pd.cut(da_df['NRI'], bins=bins, labels=labels)

    # --- Step 4: Neural Difference Calculation ---
    neural_summary = \
    da_df.groupby(['animal', 'session', 'hemisphere', 'side_relative', 'block', 'NRI_bin'], observed=True)['DA'].mean()
    neural_summary = neural_summary.unstack('block').reset_index()
    neural_summary['diff_neural'] = neural_summary['0.8'] - neural_summary['0.4']
    df_neural = neural_summary.dropna(subset=['diff_neural'])

    df_master = pd.merge(df_behav, df_neural, on=['animal', 'session'], how='inner')
    df_master['animal_hemi'] = df_master['animal'] + " " + df_master['hemisphere']

    # --- Step 5: Plotting with Color Gradient ---
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), sharex=True)

    contra_combos = sorted(df_master[df_master['side_relative'] == 'contra']['animal_hemi'].unique())
    ipsi_combos = sorted(df_master[df_master['side_relative'] == 'ipsi']['animal_hemi'].unique())

    contra_axes = [axes[r, c] for c in [0, 1] for r in range(3)]
    ipsi_axes = [axes[r, c] for c in [2, 3] for r in range(3)]

    def draw_regressions(axes_pool, combos, base_color):
        for i, combo_name in enumerate(combos):
            if i >= len(axes_pool): break
            ax = axes_pool[i]
            subset = df_master[df_master['animal_hemi'] == combo_name]

            # Use light_palette to create a gradient (Dark -> Light)
            color_grad = sns.light_palette(base_color, n_colors=len(labels) + 2, reverse=True)

            plotted_any = False
            for b_idx, bin_label in enumerate(labels):
                bin_data = subset[subset['NRI_bin'] == bin_label]
                if len(bin_data) > 3:
                    # truncate=False forces the line to span the entire x-axis
                    sns.regplot(data=bin_data, x='diff_behavior', y='diff_neural',
                                scatter=False, ax=ax, color=color_grad[b_idx],
                                line_kws={'lw': 3.5, 'label': bin_label, 'alpha': 0.8})
                    plotted_any = True

            ax.set_title(combo_name, fontweight='bold', fontsize=16)
            # if plotted_any:
            #     ax.legend(title='NRI Window', fontsize='x-small', loc='upper right')

            ax.set_xlim(-5.5, 1)
            ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.3)
            ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # for j in range(len(combos), len(axes_pool)):
        #     axes_pool[j].axis('off')

    # CONTRA: Red-based gradient | IPSI: Black-based (Gray) gradient
    draw_regressions(contra_axes, contra_combos, 'darkred')
    draw_regressions(ipsi_axes, ipsi_combos, 'black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, f"animal_hemi_regressions_w{window_width}.png"))
    plt.show()

if __name__ == "__main__":
    # run_context_sensitivity_analysis(side='both')
    run_binned_context_analysis(side='ipsi')
    # plot_animal_hemis_context_sensitivity_regressions(window_width=2)