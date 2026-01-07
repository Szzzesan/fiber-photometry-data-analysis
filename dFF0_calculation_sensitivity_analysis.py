import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
import helper
from OneSession import OneSession

# Import detrenders
from helper.beads_detrend import beads_detrend
from helper.butterworth_detrend import butterworth_detrend
from helper.moving_average_denoise import moving_average_denoise
from helper.lin_reg_fit import lin_reg_fit


# -----------------------------------------------------------------------------
# 1. Configuration & Metadata
# -----------------------------------------------------------------------------

qc_selections = {
    'SZ036': {'left', 'right'},
    'SZ037': {'left', 'right'},
    'SZ038': {'left', 'right'},
    'SZ039': {'left'},
    'SZ042': {'left'},
    'SZ043': {'right'},
    'RK007': {'left'},
    'RK008': {'left'},
    'RK009': {'left'},
    'RK010': {'left', 'right'}
}

# Mapping of Time Investment Port side (from quality_control.py)
EXP_WHICH_SIDE = {
    'SZ036': 'right', 'SZ037': 'right', 'SZ038': 'right', 'SZ039': 'right',
    'SZ042': 'right', 'SZ043': 'right',
    'RK007': 'left', 'RK008': 'right', 'RK009': 'left', 'RK010': 'right'
}


def get_side_relative(animal, hemisphere):
    """Determines if the recording is Ipsilateral or Contralateral to the Exp Port."""
    exp_side = EXP_WHICH_SIDE.get(animal, 'right')  # Default to right if unknown

    # hemisphere in df is usually 'left' or 'right' (lower case)
    # If the recording hemisphere matches the Exp Port side -> Ipsi
    if hemisphere == exp_side:
        return 'ipsi'
    else:
        return 'contra'


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

def plot_lmem_coefficients(model_results, ax, title_suffix=""):
    """
    Plots fixed effect coefficients with 95% CIs on a specific axes object.
    """
    params = model_results.params
    conf = model_results.conf_int()
    conf['estimate'] = params
    conf.columns = ['lower', 'upper', 'estimate']

    # Filter out random effects (Group Var) and Intercept
    rows_to_drop = [x for x in conf.index if 'Group Var' in x or 'Intercept' in x]
    conf_to_plot = conf.drop(index=rows_to_drop)

    # Map patsy names to readable names
    name_map = {
        "C(block, Treatment('0.8'))[T.0.4]": 'context (low vs. high)',
        "logNRI_std": 'time',
        "logIRI_std": 'IRI',
        "C(side_relative, Treatment('ipsi'))[T.contra]": 'side',

        # 2-Way Interactions
        "logNRI_std:C(side_relative, Treatment('ipsi'))[T.contra]": 'time * side',
        "logNRI_std:logIRI_std": 'time * IRI',
        "logNRI_std:C(block, Treatment('0.8'))[T.0.4]": 'time * context',
        "logIRI_std:C(block, Treatment('0.8'))[T.0.4]": 'IRI * context',
        "logIRI_std:C(side_relative, Treatment('ipsi'))[T.contra]": 'IRI * side',
        "C(block, Treatment('0.8'))[T.0.4]:C(side_relative, Treatment('ipsi'))[T.contra]": 'context * side',

        # 3-Way Interactions
        "logNRI_std:logIRI_std:C(block, Treatment('0.8'))[T.0.4]": 'time * IRI * context',
        "logIRI_std:C(block, Treatment('0.8'))[T.0.4]:C(side_relative, Treatment('ipsi'))[T.contra]": 'IRI * context * side',
        "logNRI_std:logIRI_std:C(side_relative, Treatment('ipsi'))[T.contra]": 'time * IRI * side',
        "logNRI_std:C(block, Treatment('0.8'))[T.0.4]:C(side_relative, Treatment('ipsi'))[T.contra]": 'time * context * side',

        # Random Effects
        "Group Var": 'animal (rand)',
        "session Var": 'session (rand)',
        "site Var": 'site (rand)'
    }

    # Rename index
    conf_to_plot_renamed = conf_to_plot.rename(index=name_map)

    # Desired order for the plot
    desired_order = [
        'time',
        'context (low vs. high)',
        'IRI',
        'side',
        'time * side',
        'context * side',
        'IRI * side',
        'time * IRI',
        'time * context',
        'IRI * context',
        'time * IRI * side'
    ]

    # Reorder based on availability
    final_order = [name for name in desired_order if name in conf_to_plot_renamed.index]
    # remaining = [name for name in conf_to_plot_renamed.index if name not in final_order]
    # conf_to_plot = conf_to_plot_renamed.reindex(final_order + remaining)
    conf_to_plot = conf_to_plot_renamed.reindex(final_order)

    # Calculate Errors [estimate - lower, upper - estimate]
    yerr = [conf_to_plot['estimate'] - conf_to_plot['lower'], conf_to_plot['upper'] - conf_to_plot['estimate']]

    # Plot
    ax.errorbar(
        x=range(len(conf_to_plot)),
        y=conf_to_plot['estimate'],
        yerr=yerr,
        fmt='o',
        color='black',
        capsize=2,
        linewidth=0.75,
        linestyle='None',
        markersize=4,
        ecolor='red'
    )

    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_title(f'Coefficients: {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient Estimate')
    ax.set_ylim(-0.5, 0.5)

    # Set x-ticks with rotation
    ax.set_xticks(range(len(conf_to_plot)))
    ax.set_xticklabels(conf_to_plot.index, rotation=35, ha='right', fontsize=9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


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
    rows = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()

    # Colors
    color_map = {'0.4': sns.color_palette('Set2')[0], '0.8': sns.color_palette('Set2')[1]}

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
        ax.set_ylim(-0.6, 0.6)
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

    fig.suptitle(f"Non-Reward Traces (0.5s bins): {title_suffix}", fontsize=14)
    plt.show()

# -----------------------------------------------------------------------------
# 2. Flexible dF/F0 Calculator
# -----------------------------------------------------------------------------
def calculate_dFF0_flexible(raw_separated, session_label,
                            detrend_method='none',
                            denominator_method='fitted_isos',
                            baseline_window=None):
    """
    Calculates dF/F0 using various detrending and baseline strategies.

    Args:
        detrend_method: 'butterworth', 'beads', or 'none'
        denominator_method: 'fitted_isos' or '470_rolling_window' or 'isos_rolling_window'
        baseline_window: window size in seconds (for rolling_window only)
    """
    num_color_site = int(len(raw_separated.columns) / 2 - 1)

    # --- Step A: Detrending ---
    if detrend_method == 'butterworth':
        # Standard high-pass (0.1Hz cutoff in your original code)
        # Note: This removes slow tides >10s
        processed = butterworth_detrend(raw_separated.copy(), session_label=session_label, fps=40, plot=False)
    elif detrend_method == 'beads':
        # BEADS (removes drift, preserves slow tides better than butterworth)
        processed = beads_detrend(raw_separated.copy(), session_label=session_label, plot=False)
    elif detrend_method == 'none':
        # No pre-detrending (Rely on Isosbestic fit to handle drift)
        processed = raw_separated.copy()
    else:
        raise ValueError(f"Unknown detrend_method: {detrend_method}")

    # --- Step B: Denoise & Linear Regression Fit ---
    denoised = moving_average_denoise(processed, win_size=8, plot=False, session_label=session_label)
    fitted = lin_reg_fit(denoised, plot=False, session_label=session_label)

    # --- Step C: Calculate dF/F0 ---
    dFF0 = pd.DataFrame(columns=['time_recording', fitted.columns.values[1][:-4], fitted.columns.values[3][:-4]])
    dFF0.iloc[:, 0] = fitted.iloc[:, 0]

    fps = 40  # Assuming 40Hz

    for i in range(num_color_site):
        # Identify columns
        # fitted structure: [time, signal_site1, isos_site1, signal_site2, isos_site2]
        signal_col_idx = 2 * i + 1
        isos_col_idx = 2 * (i + 1)

        signal = fitted.iloc[:, signal_col_idx]
        fitted_isos = fitted.iloc[:, isos_col_idx]
        col_name = dFF0.columns[i + 1]  # e.g., 'green_right'

        # Numerator: Always Signal - Fitted Isosbestic (Motion correction)
        dF = signal - fitted_isos

        # Denominator: Flexible
        if denominator_method == 'fitted_isos':
            Ft = signal
            F0 = fitted_isos
            dF = Ft - F0
        elif denominator_method == '470_rolling_window':
            if baseline_window is None:
                # Global Mean F0
                F0 = signal.mean()
            else:
                # Rolling Mean F0
                window_frames = int(baseline_window * fps)
                F0 = signal.rolling(window=window_frames, center=True).mean()
                start_mean = signal.iloc[:window_frames].mean()
                end_mean = signal.iloc[-window_frames:].mean()
                half_window = (window_frames - 1) // 2
                F0.iloc[:half_window] = F0.iloc[:half_window].fillna(start_mean)
                F0.iloc[-half_window:] = F0.iloc[-half_window:].fillna(end_mean)
            dF = signal - fitted_isos
        elif denominator_method == 'isos_rolling_window':
            if baseline_window is None:
                F0 = fitted_isos.mean()
            else:
                window_frames = int(baseline_window * fps)
                F0 = fitted_isos.rolling(window=window_frames, center=True).mean()
                start_mean = fitted_isos.iloc[:window_frames].mean()
                end_mean = fitted_isos.iloc[-window_frames:].mean()
                half_window = (window_frames - 1) // 2
                F0.iloc[:half_window] = F0.iloc[:half_window].fillna(start_mean)
                F0.iloc[-half_window:] = F0.iloc[-half_window:].fillna(end_mean)
            dF = signal - fitted_isos
        elif denominator_method == 'subtracted (470 nm - isos) rolling window':
            Ft = signal - fitted_isos
            if baseline_window is None:
                F0 = signal.mean()
            else:
                window_frames = int(baseline_window * fps)
                F0 = Ft.rolling(window=window_frames, center=True).mean()
                start_mean = Ft.iloc[:window_frames].mean()
                end_mean = Ft.iloc[-window_frames:].mean()
                half_window = (window_frames - 1) // 2
                F0.iloc[:half_window] = F0.iloc[:half_window].fillna(start_mean)
                F0.iloc[-half_window:] = F0.iloc[-half_window:].fillna(end_mean)
            dF = Ft - F0

        dFF0.iloc[:, i + 1] = dF / F0

    dFF0.iloc[:, 0] = dFF0.iloc[:, 0].div(1000)  # Convert ms to seconds
    return dFF0


# -----------------------------------------------------------------------------
# 3. Session Definitions
# -----------------------------------------------------------------------------
def get_all_sessions():
    """Returns a list of all available sessions as tuples:
       (animal_str, session_id, include_branch, port_swap)"""

    sessions = []

    # SZ Animals
    if 'SZ036' in qc_selections: sessions.extend(
        [('SZ036', s, 'both', 0) for s in [1, 2, 3, 5, 7, 9, 11, 12, 14, 15, 19, 22, 23, 24, 25]])
    if 'SZ037' in qc_selections: sessions.extend([('SZ037', s, 'both', 0) for s in
                                                  [0, 1, 2, 4, 5, 6, 8, 9, 11, 15, 16, 17, 18, 19, 20, 22, 24, 25, 27,
                                                   28, 29, 31, 32, 33, 35]])
    if 'SZ038' in qc_selections: sessions.extend([('SZ038', s, 'both', 0) for s in
                                                  [1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                   24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]])
    if 'SZ039' in qc_selections: sessions.extend(
        [('SZ039', s, 'only_left', 0) for s in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22]])
    if 'SZ042' in qc_selections: sessions.extend([('SZ042', s, 'only_left', 0) for s in
                                                  [3, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 21, 22, 24, 26, 27, 28,
                                                   30]])
    if 'SZ043' in qc_selections: sessions.extend(
        [('SZ043', s, 'only_right', 0) for s in [0, 1, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23]])

    # RK Animals
    if 'RK007' in qc_selections: sessions.extend([('RK007', s, 'left', 1) for s in range(20)])
    if 'RK008' in qc_selections: sessions.extend([('RK008', s, 'left', 0) for s in range(14, 25)])
    if 'RK009' in qc_selections: sessions.extend([('RK009', s, 'left', 1) for s in range(15)])
    if 'RK010' in qc_selections: sessions.extend([('RK010', s, 'both', 0) for s in range(13)])

    return sessions


# -----------------------------------------------------------------------------
# 4. Main Analysis Script
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # A. Pick 20 Random Sessions
    all_sessions = get_all_sessions()
    random.seed(999)  # Fixed seed for reproducibility
    selected_sessions = random.sample(all_sessions, 150)

    print(f"Selected {len(selected_sessions)} sessions for sensitivity analysis.")

    # B. Define Scenarios
    scenarios = [
        # 0. The Original Method (10 sec rolling window as Baseline)
        {'name': 'Original (Butterworth + 10sec 470nm Rolling)', 'detrend': 'butterworth',
         'denom': '470_rolling_window', 'win': 10},

        # # 1. The Original Method (fitted isos as Baseline)
        # {'name': 'Original (Butterworth + 10sec FitIso Rolling)', 'detrend': 'butterworth',
        #  'denom': 'isos_rolling_window', 'win': 10},
        #
        # # 2. No High-Pass Filter (Best for Tides?)
        # {'name': 'No Detrend (Raw + FitIso)', 'detrend': 'none',
        #  'denom': 'fitted_isos', 'win': None},
        #
        # # 3. BEADS Detrending (Advanced Baseline Removal)
        # {'name': 'BEADS Detrend + FitIso', 'detrend': 'beads',
        #  'denom': 'fitted_isos', 'win': None},

        # 4. Long Rolling Baseline (6 Minutes)
        {'name': 'BEADS + 6min Rolling 470nm', 'detrend': 'beads',
         'denom': '470_rolling_window', 'win': 360},

        # 5. Long Rolling Baseline (6 Minutes) with fitted isos
        {'name': 'BEADS + 6min Rolling FittedIso', 'detrend': 'beads',
         'denom': 'isos_rolling_window', 'win': 360}
    ]

    results_summary = []

    # C. Run Loop
    for scen in scenarios:
        print(f"\n--- Running Scenario: {scen['name']} ---")
        pooled_lmem_data = []
        pooled_trace_data = []

        for (animal, session_id, branch_cfg, port_swap) in selected_sessions:
            try:
                # 1. Load Session (Minimal Init)
                sess = OneSession(animal, session_id, include_branch=branch_cfg, port_swap=port_swap)

                # 2. Get Raw Data
                # Note: We rely on helper.de_interleave to get the raw dataframe structure
                helper.check_framedrop(sess.neural_events)
                raw_separated = helper.de_interleave(sess.neural_events,
                                                     session_label=f"{animal}_{session_id}",
                                                     save_path=None, plot=0, save=0)

                # 3. Calculate dF/F0 (Custom Method)
                sess.dFF0 = calculate_dFF0_flexible(raw_separated,
                                                    session_label=f"{animal}_{session_id}",
                                                    detrend_method=scen['detrend'],
                                                    denominator_method=scen['denom'],
                                                    baseline_window=scen['win'])

                # 4. Standard Post-Processing (Z-Score & Alignment)
                # Replicate z-scoring from OneSession logic
                sess.zscore = pd.DataFrame({'time_recording': sess.dFF0.time_recording})
                for col in sess.dFF0.columns:
                    if 'green' in col:
                        sess.zscore[col] = stats.zscore(sess.dFF0[col].tolist(), nan_policy='omit')

                sess.process_behavior_data(save=0)
                sess.construct_trial_df()
                sess.add_trial_info_to_recording()
                sess.construct_expreward_interval_df()

                # 5. Extract DA Features (DA vs NRI/IRI)
                # sess.extract_reward_features_and_DA(plot=0, save_dataframe=0)
                sess.visualize_DA_vs_NRI_IRI(plot_scatters=0, plot_histograms=0)
                if sess.DA_vs_NRI_IRI is not None and not sess.DA_vs_NRI_IRI.empty:
                    df = sess.DA_vs_NRI_IRI.copy()
                    valid_hemis = qc_selections.get(animal, set())
                    df = df[df['hemisphere'].isin(valid_hemis)].copy()
                    # Add identifiers for LMEM grouping
                    df['animal'] = animal
                    df['session'] = sess.signal_dir[-21:-7]
                    df['side_relative'] = df['hemisphere'].apply(lambda h: get_side_relative(animal, h))
                    pooled_lmem_data.append(df)

                # 6. Non-reward Dopamine Trace Data
                sess.extract_nonreward_DA_vs_time(exclusion_start_relative=0, exclusion_end_relative=2)
                if hasattr(sess, 'nonreward_DA_vs_time') and sess.nonreward_DA_vs_time is not None:
                    nr_df = sess.nonreward_DA_vs_time.copy()
                    # Ensure phase is consistent string
                    nr_df['phase'] = nr_df['phase'].astype(str)

                    data_cols = [c for c in nr_df.columns if 'green' in c]
                    if data_cols:
                        id_vars = ['time_in_port', 'phase']
                        nr_melted = nr_df.melt(id_vars=id_vars, value_vars=data_cols,
                                               var_name='hemisphere_col', value_name='DA')
                        nr_melted['animal'] = animal
                        nr_melted['hemisphere'] = nr_melted['hemisphere_col'].str.replace('green_', '')
                        valid_hemis = qc_selections.get(animal, set())
                        nr_melted = nr_melted[nr_melted['hemisphere'].isin(valid_hemis)].copy()
                        pooled_trace_data.append(nr_melted)

            except Exception as e:
                print(f"Skipping {animal} Session {session_id}: {e}")

        # D. Concatenate & Run Statistics
        if not pooled_lmem_data:
            print(f"No data collected for scenario {scen['name']}")
            continue

        # --- LMEM Fitting & Plotting ---
        master_df = pd.concat(pooled_lmem_data, ignore_index=True)

        df_model = master_df[(master_df['IRI'] > 1)].copy()
        df_model['animal_hemisphere'] = df_model['animal'].astype(str) + '_' + df_model['hemisphere'].astype(str)
        df_model['logNRI'] = np.log(df_model['NRI'])
        df_model['logIRI'] = np.log(df_model['IRI'])
        for col in ['logNRI', 'logIRI']:
            mean_val = df_model[col].mean()
            std_val = df_model[col].std()
            df_model[f'{col}_std'] = (df_model[col] - mean_val) / std_val

        print(f"  Data Points for Model: {len(df_model)}")

        model_formula = (
            "DA ~ (logNRI_std + logIRI_std "
            "+ C(block, Treatment('0.8')) "
            "+ C(side_relative, Treatment('ipsi')))**3 "
            "+ C(hemisphere, Treatment('right'))"
        )

        random_slopes = (
            "1 + "
            "logNRI_std + "
            "logIRI_std + "
            "C(block, Treatment(reference='0.8'))"
        )

        model = smf.mixedlm(
            model_formula,
            data=df_model,
            groups=df_model["animal"],
            re_formula=random_slopes,
            vc_formula={
                "session": "0 + C(session)",
                "site": "0 + C(animal_hemisphere)"
            }
        )

        # Fit Model
        print("  Fitting LMEM (method='powell')...")
        model_results = model.fit(method="powell", reml=False)

        fig1, ax1 = plt.subplots(figsize=(8, 6))  # Create NEW figure for this scenario
        plot_lmem_coefficients(model_results, ax1, title_suffix=scen['name'])
        plt.tight_layout()
        plt.show()

        # --- Nonreward DA Traces by Block ---
        if pooled_trace_data:
            trace_df = pd.concat(pooled_trace_data, ignore_index=True)
            binned_df = bin_nonreward_data(trace_df, bin_size=0.5)
            plot_nonreward_traces_in_grid(binned_df, title_suffix=scen['name'])





    # E. Final Summary Table
    print("\n\n=== Sensitivity Analysis Summary (20 Random Sessions) ===")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df)

