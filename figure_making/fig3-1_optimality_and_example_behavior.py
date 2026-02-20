import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import wilcoxon

# Project imports
import config
import data_loader

# --- Visual Style & Configuration ---
sns.set_context("paper", font_scale=1.1)
sns.set_style("white")
sns.set_style("ticks")

FIG_WIDTH = 12
ROW_HEIGHT_RATIOS = [2, 3, 4]
ROW1_WIDTH_RATIOS = [2, 10, 0.5]

COLOR_LOW = sns.color_palette('Set2')[0]
COLOR_HIGH = sns.color_palette('Set2')[1]


# ==============================================================================
# SECTION 0: User-Provided Helper Functions
# ==============================================================================

def min_dif(a, b, tolerance=0, return_index=False, rev=False):
    """
    Calculates the minimum difference between elements of two arrays (b - a).
    Finds the smallest positive difference (next event in b after event in a).
    """
    if isinstance(a, pd.Series):
        a = a.values
    if isinstance(b, pd.Series):
        b = b.values

    # Standardize shape
    a = np.array(a)
    b = np.array(b)

    if rev:
        outer = -1 * np.subtract.outer(a, b)
        outer[outer <= tolerance] = np.nan
    else:
        # outer[i, j] = b[j] - a[i]
        outer = np.subtract.outer(b, a)
        outer[outer <= tolerance] = np.nan

    # Suppress "All-NaN slice" warnings for empty columns
    with np.errstate(all='ignore'):
        mins = np.nanmin(outer, axis=0)

    if return_index:
        with np.errstate(all='ignore'):
            index = np.nanargmin(outer, axis=0)
        return index, mins
    return mins


def get_entry_exit(df, trial):
    """
    Extracts entry and exit times for Background (Context) and Exponential (Investment) ports
    for a specific trial, handling various edge cases (middle of trial, early entries, etc.).
    """
    is_trial = df.trial == trial
    start = df.value == 1
    end = df.value == 0
    port1 = df.port == 1  # Investment
    port2 = df.port == 2  # Context

    # Trial boundaries
    try:
        trial_start = df[is_trial & start & (df.key == 'trial')].session_time.values[0]
        trial_end = df[is_trial & end & (df.key == 'trial')].session_time.values[0]
        # Trial middle is defined by the LED turning off in Port 2 (Context exit cue)
        trial_middle = df[is_trial & end & (df.key == 'LED') & port2].session_time.values[0]
    except IndexError:
        # If trial structure is incomplete, return empty
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # --- Background (Context) Port 2 ---
    bg_entries = df[is_trial & port2 & start & (df.key == 'head')].session_time.to_numpy()
    bg_exits = df[is_trial & port2 & end & (df.key == 'head')].session_time.to_numpy()

    # Handle BG boundary conditions
    if len(bg_entries) == 0 or (len(bg_exits) > 0 and bg_entries[0] > bg_exits[0]):
        bg_entries = np.concatenate([[trial_start], bg_entries])

    # If last entry has no exit and is close to end, trim or cap?
    # Logic from snippet: if trial_end - bg_entries[-1] < .1, remove entry.
    if len(bg_entries) > 0 and (trial_end - bg_entries[-1] < .1):
        bg_entries = bg_entries[:-1]

    # If missing last exit, assume trial_middle (end of context phase)
    if len(bg_exits) == 0 or (len(bg_entries) > 0 and bg_entries[-1] > bg_exits[-1]):
        bg_exits = np.concatenate([bg_exits, [trial_middle]])

    # --- Exponential (Investment) Port 1 ---
    # Standard entries (after trial middle)
    exp_entries = df[
        is_trial & port1 & start & (df.key == 'head') & (df.session_time > trial_middle)].session_time.to_numpy()
    exp_exits = df[
        is_trial & port1 & end & (df.key == 'head') & (df.session_time > trial_middle)].session_time.to_numpy()

    if not (len(exp_entries) == 0 and len(exp_exits) == 0):
        if len(exp_entries) == 0:
            exp_entries = np.concatenate([[trial_middle], exp_entries])
        if len(exp_exits) == 0:
            exp_exits = np.concatenate([exp_exits, [trial_end]])

        if len(exp_entries) > 0 and len(exp_exits) > 0:
            if exp_entries[0] > exp_exits[0]:
                exp_entries = np.concatenate([[trial_middle], exp_entries])
            if exp_entries[-1] > exp_exits[-1]:
                exp_exits = np.concatenate([exp_exits, [trial_end]])

    # Early entries (before trial middle - rare/error)
    early_exp_entries = df[
        is_trial & port1 & start & (df.key == 'head') & (df.session_time < trial_middle)].session_time.to_numpy()
    early_exp_exits = df[
        is_trial & port1 & end & (df.key == 'head') & (df.session_time < trial_middle)].session_time.to_numpy()

    # (Skipping detailed correction for early entries for brevity, using main logic)

    return bg_entries, bg_exits, exp_entries, exp_exits, early_exp_entries, early_exp_exits


# ==============================================================================
# SECTION 1: Optimality Modeling (Row 1 Left)
# ==============================================================================

def global_reward_gain(x, cumulative=8., starting=1.):
    b = starting / cumulative
    return cumulative * (1 - np.exp(-b * x))


def global_reward_rate(x, context_time=5, consumption_time=0.5, travel_time=0.5, cumulative=8., starting=1.):
    # +4 represents the 4 fixed context rewards
    rou_g = (global_reward_gain(x, cumulative, starting) + 4) / (x + context_time + consumption_time + travel_time * 2)
    return rou_g


def exp_decreasing(x, cumulative=8., starting=1.):
    b = starting / cumulative
    # Instantaneous rate (derivative of gain)
    return starting * np.exp(-b * x)


def plot_optimality(ax):
    x = np.arange(0, 15, 0.1)
    # Note: context_time=10 for Low (0.4 rate -> longer time), 5 for High
    y_low = global_reward_rate(x, context_time=10)
    y_high = global_reward_rate(x, context_time=5)

    x_low_max = x[np.argmax(y_low)]
    x_high_max = x[np.argmax(y_high)]

    x_adjusted = np.arange(0.1, 15, 0.1)
    y_inst = exp_decreasing(x_adjusted)

    ax.plot(x_adjusted, y_inst, color='grey', label='Instantaneous', alpha=0.6)
    ax.plot(x, y_low, color=COLOR_LOW, label='Global (Low)')
    ax.plot(x, y_high, color=COLOR_HIGH, label='Global (High)')

    ax.scatter(x_low_max, np.max(y_low), color=COLOR_LOW, zorder=5)
    ax.scatter(x_high_max, np.max(y_high), color=COLOR_HIGH, zorder=5)

    ax.vlines(x=x_low_max, ymin=0, ymax=np.max(y_low), color=COLOR_LOW, linestyle='--', lw=1.5)
    ax.vlines(x=x_high_max, ymin=0, ymax=np.max(y_high), color=COLOR_HIGH, linestyle='--', lw=1.5)

    ax.set_ylim([0, 1.05])
    ax.set_title("MVT Optimality")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward Rate")
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),  # (x, y) coordinates in axis fractions
        fontsize='x-small',  # Makes the text smaller
        labelspacing=0.2,  # Reduces vertical space between labels
        handlelength=1.0,  # Shortens the colored line icons
    )
    sns.despine(ax=ax)


# ==============================================================================
# SECTION 2: Example Session Visualization (Row 1 Right)
# ==============================================================================

def extract_raw_trial(pi_events):
    # Simplified parser to get patches
    entries = pi_events[(pi_events.key == 'head') & (pi_events.value == 1)].copy()
    entries['type'] = 'entry'
    exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0)].copy()
    exits['type'] = 'exit'

    # Simple sort and pair (assuming alternating structure for visualization purposes)
    events = pd.concat([entries, exits]).sort_values('session_time')

    patches_list = []
    # Identify valid intervals
    # Port 2 = Context (-), Port 1 = Invest (+)

    # Vectorized pairing is risky with noise, using iterative approach for robustness
    # Only process first 10 mins for the figure
    max_time = 60 * 19
    events = events[events.session_time < max_time]

    # We iterate by port to find pairs
    for port in [1, 2]:
        p_events = events[events.port == port]
        p_entries = p_events[p_events.type == 'entry'].session_time.values
        p_exits = p_events[p_events.type == 'exit'].session_time.values

        # Ensure alignment
        if len(p_entries) == 0: continue

        # Trim
        n = min(len(p_entries), len(p_exits))
        p_entries = p_entries[:n]
        p_exits = p_exits[:n]

        # Filter where exit < entry (bad data)
        valid = p_exits > p_entries
        p_entries = p_entries[valid]
        p_exits = p_exits[valid]

        port_sign = -1 if port == 2 else 1

        for en, ex in zip(p_entries, p_exits):
            patches_list.append({
                'entry': en,
                'exit': ex,
                'stay': ex - en,
                'port_sign': port_sign,
                'rewards': []  # Filled later if needed
            })

    # Add rewards
    rewards = pi_events[(pi_events.key == 'reward') & (pi_events.value == 1) & (pi_events.session_time < max_time)]

    return pd.DataFrame(patches_list), rewards


def plot_behav_in_patches(ax, animal_id, session_long_name):
    pi_events = data_loader.load_session_dataframe(animal_id, 'pi_events_processed',
                                                   session_long_name=session_long_name)
    if pi_events is None: return

    # --- NORMALIZE TIME ---
    bg_entries = pi_events[(pi_events.port == 2) & (pi_events.key == 'head') & (pi_events.value == 1)]
    if not bg_entries.empty:
        t_zero = bg_entries.session_time.min()
        pi_events['session_time'] = pi_events['session_time'] - t_zero

    limit = 18 * 60  # 18 mins

    # --- PLOT BLOCKS (Horizontal Bars) ---
    # Identify block transitions (phase changes)
    pi_events = pi_events.sort_values('session_time')
    pi_events['phase'] = pi_events['phase'].astype(float)

    # We create a dataframe of phase changes
    # Use 'trial' start times to define block boundaries more cleanly than raw events
    trials = pi_events[pi_events.key == 'trial']
    if not trials.empty:
        # Group by trial to find phase
        trial_phases = trials.groupby('trial')['phase'].first().reset_index()
        # Get start/end time for each trial
        trial_starts = trials[trials.value == 1].groupby('trial')['session_time'].min()
        trial_ends = trials[trials.value == 0].groupby('trial')['session_time'].max()

        trial_phases['start'] = trial_phases['trial'].map(trial_starts)
        trial_phases['end'] = trial_phases['trial'].map(trial_ends)

        # Iterate to fuse contiguous blocks
        current_phase = None
        block_start = 0

        # Y-position for the bar (top of the plot)
        bar_y = 18
        bar_height = 2

        for i, row in trial_phases.iterrows():
            if pd.isna(row['start']): continue
            if row['start'] > limit: break

            p = row['phase']
            if p != current_phase:
                # Close previous block
                if current_phase is not None:
                    color = COLOR_HIGH if current_phase == 0.8 else COLOR_LOW
                    width = row['start'] - block_start
                    rect = patches.Rectangle((block_start / 60, bar_y), width / 60, bar_height,
                                             fc=color, alpha=0.6, ec=None)
                    ax.add_patch(rect)
                    # Text label for the block (optional, or just color code)
                    # ax.text((block_start + width/2)/60, bar_y + 0.5, "High" if current_phase==0.8 else "Low",
                    #         ha='center', va='center', fontsize=6, color=color)

                # Start new block
                current_phase = p
                block_start = row['start']

        # Close final block
        if current_phase is not None:
            color = COLOR_HIGH if current_phase == 0.8 else COLOR_LOW
            width = limit - block_start
            rect = patches.Rectangle((block_start / 60, bar_y), width / 60, bar_height,
                                     fc=color, alpha=0.6, ec=None)
            ax.add_patch(rect)

    # --- PLOT PATCHES ---
    entries = pi_events[(pi_events.key == 'head') & (pi_events.value == 1)].copy()
    exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0)].copy()
    rewards = pi_events[(pi_events.key == 'reward') & (pi_events.value == 1)].copy()

    entries = entries[(entries.session_time >= 0) & (entries.session_time < limit)]

    for _, row in entries.iterrows():
        exit_match = exits[(exits.port == row.port) & (exits.session_time > row.session_time)]
        if exit_match.empty: continue
        exit_time = exit_match.iloc[0].session_time

        stay = exit_time - row.session_time
        port_sign = -1 if row.port == 2 else 1
        c = 'lightcoral' if port_sign > 0 else 'skyblue'

        rect = patches.Rectangle((row.session_time / 60, 0), stay / 60, stay * port_sign, fc=c, alpha=0.5, zorder=1)
        ax.add_patch(rect)

        patch_rewards = rewards[(rewards.session_time >= row.session_time) & (rewards.session_time <= exit_time)]
        if not patch_rewards.empty:
            y_vals = (patch_rewards.session_time - row.session_time) * port_sign
            ax.scatter(patch_rewards.session_time / 60, y_vals, c='black', marker='o', s=0.5, zorder=2)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Duration (s)", labelpad=-10)
    ax.set_xlim([-0.5, 18.5])
    ax.set_xticks(np.arange(0, 18.5, 3))
    # Increase Y-limit to accommodate the block bars at the top
    ax.set_ylim([-15, 20])
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f"Example Session")
    sns.despine(ax=ax)


def plot_behavior_legend(ax):
    """Creates a custom legend for the behavior visualization."""
    # Create handles
    h_rew = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='black',
                          markeredgecolor='none', markersize=6, label='Reward')

    p_inv = patches.Patch(facecolor='lightcoral', alpha=0.5, label='Investment\nPort')
    p_ctx = patches.Patch(facecolor='skyblue', alpha=0.5, label='Context\nPort')

    p_high = patches.Patch(facecolor=COLOR_HIGH, alpha=0.6, label='High\nBlock')
    p_low = patches.Patch(facecolor=COLOR_LOW, alpha=0.6, label='Low\nBlock')

    handles = [h_rew, p_inv, p_ctx, p_high, p_low]

    # Place legend in the center of this dedicated axis
    ax.legend(handles=handles, loc='center', frameon=False, fontsize='x-small', borderpad=0)
    ax.axis('off')

# ==============================================================================
# SECTION 3: Cross-Session Metrics (Row 2)
# ==============================================================================

def calculate_session_metrics(df):
    """Calculates the 4 key metrics for a single session dataframe."""

    # --- 1. Consumption ---
    # Time from Context End (LED On, Port 2) to Nearest Valid Exit (Port 2)
    bg_end_times = df[(df.key == 'LED') & (df.port == 2) & (df.value == 1)].session_time

    # Check for 'is_valid' column safely
    if 'is_valid' in df.columns:
        valid_exits = df[(df.key == 'head') & (df.port == 2) & (df.value == 0) & (df['is_valid'] == 1)].session_time
    else:
        valid_exits = df[(df.key == 'head') & (df.port == 2) & (df.value == 0)].session_time

    # Calculate difference
    if len(bg_end_times) > 0 and len(valid_exits) > 0:
        dif = min_dif(bg_end_times, valid_exits)
        # Filter for positive values < 10s
        valid_dif = dif[(~np.isnan(dif)) & (dif > 0) & (dif < 10.0)]
        cons = np.nanmean(valid_dif) if len(valid_dif) > 0 else np.nan
    else:
        cons = np.nan

    # --- 2. Percent Engaged & 3. Realized Reward Rate ---
    # ROBUST METHOD: Map trials to phases first, then aggregate.

    # 1. Create a map of Trial ID -> Phase (0.4 or 0.8)
    # We group by trial and take the first non-null phase value found in that trial
    if 'phase' in df.columns:
        trial_phase_map = df.groupby('trial')['phase'].first()
    else:
        print("Warning: 'phase' column missing in dataframe.")
        return cons, {'0.4': {}, '0.8': {}}, {'0.4': np.nan, '0.8': np.nan}

    metrics_by_block = {'0.4': {}, '0.8': {}}
    travel_time = 0.3

    # Iterate explicitly over the two known phases
    for target_phase in ['0.4', '0.8']:
        # Find all trial IDs that belong to this phase
        # We use string conversion to handle float/string mismatches (e.g., 0.4 vs "0.4")
        trials_in_phase = []
        for t_id, p_val in trial_phase_map.items():
            if pd.isna(t_id) or pd.isna(p_val): continue

            # Check for match (fuzzy float match or string match)
            is_match = False
            try:
                if str(p_val) == target_phase:
                    is_match = True
                elif abs(float(p_val) - float(target_phase)) < 0.001:
                    is_match = True
            except:
                pass

            if is_match:
                trials_in_phase.append(t_id)

        # Calculate stats for these trials
        engaged_durations = []
        total_durations = []
        rewards_count = 0

        for trial in trials_in_phase:
            try:
                # Get Entry/Exit times using your helper function
                bg_in, bg_out, exp_in, exp_out, _, _ = get_entry_exit(df, trial)

                # Engaged time = Time in ports
                t_bg = np.sum(bg_out - bg_in)
                t_exp = np.sum(exp_out - exp_in)

                # Total trial time (Trial Start to Trial End)
                t_start = df[(df.trial == trial) & (df.key == 'trial') & (df.value == 1)].session_time.values[0]
                t_end = df[(df.trial == trial) & (df.key == 'trial') & (df.value == 0)].session_time.values[0]

                total_durations.append(t_end - t_start)
                engaged_durations.append(t_bg + t_exp)

                # Count Rewards
                r_count = len(df[(df.trial == trial) & (df.key == 'reward') & (df.value == 1)])
                rewards_count += r_count
            except IndexError:
                # Skip incomplete trials
                continue

        # Aggregate
        # Add travel time penalty (0.5s * 2 * number of trials)
        sum_engaged = sum(engaged_durations) + (travel_time * 2 * len(total_durations))
        sum_total = sum(total_durations)

        if sum_total > 0:
            metrics_by_block[target_phase]['pct_engaged'] = sum_engaged / sum_total
        else:
            metrics_by_block[target_phase]['pct_engaged'] = np.nan

        if sum_engaged > 0:
            metrics_by_block[target_phase]['realized_rr'] = rewards_count / sum_engaged
        else:
            metrics_by_block[target_phase]['realized_rr'] = np.nan

    # --- 4. Re-entry Index ---
    reentry_by_block = {'0.4': np.nan, '0.8': np.nan}

    # Identify all BG Exits
    is_bg_exit = (df.port == 2) & (df.key == 'head') & (df.value == 0)

    for target_phase in ['0.4', '0.8']:
        # Recalculate trials_in_phase or reuse logic above
        trials_in_phase = []
        for t_id, p_val in trial_phase_map.items():
            try:
                if str(p_val) == target_phase or abs(float(p_val) - float(target_phase)) < 0.001:
                    trials_in_phase.append(t_id)
            except:
                pass

        num_ideal = len(trials_in_phase)

        # Count BG Exits that occurred during these specific trial IDs
        # (This is robust even if the 'phase' column is NaN on the head exit row)
        if num_ideal > 0:
            num_actual = len(df[is_bg_exit & df.trial.isin(trials_in_phase)])
            reentry_by_block[target_phase] = num_actual / num_ideal

    return cons, metrics_by_block, reentry_by_block


def get_metrics_history(animal_id):
    # Fixed typo in path: 'proccessed' -> 'processed'
    processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PROCESSED_DATA_SUBDIR)
    files = sorted(glob.glob(os.path.join(processed_dir, "*_pi_events_processed.parquet")))

    data = []
    print(f"Processing {len(files)} sessions for {animal_id}...")

    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            cons, blk_metrics, reentry = calculate_session_metrics(df)

            row = {'session': i, 'consumption': cons}

            for phase in ['0.4', '0.8']:
                m = blk_metrics.get(phase, {})
                row[f'engaged_{phase}'] = m.get('pct_engaged', np.nan)
                row[f'rr_{phase}'] = m.get('realized_rr', np.nan)
                row[f'reentry_{phase}'] = reentry.get(phase, np.nan)

            data.append(row)
        except Exception as e:
            print(f"Skipping session {i} in history calc: {e}")
            continue

    return pd.DataFrame(data)


def plot_metrics_panel(axes, animal_id):
    files = sorted(glob.glob(
        os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PRETRAINING_PROCESSED_DATA_SUBDIR, "*_pi_events_proccessed.parquet")))

    data = []
    print(f"Processing metrics for {animal_id}...")

    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            cons, mets, reent = calculate_session_metrics(df)

            # Pack data
            data.append({
                'session': i,
                'consumption': cons,
                'rr_L': mets['0.4'].get('realized_rr', np.nan),
                'rr_H': mets['0.8'].get('realized_rr', np.nan),
                'eng_L': mets['0.4'].get('pct_engaged', np.nan),
                'eng_H': mets['0.8'].get('pct_engaged', np.nan),
                'ree_L': reent.get('0.4', np.nan),
                'ree_H': reent.get('0.8', np.nan)
            })
        except Exception as e:
            # IMPORTANT: Print error to see why sessions are dropped
            print(f"Skipping session {i}: {e}")
            continue

    df = pd.DataFrame(data)
    if df.empty:
        print("No valid metric data found.")
        return

    # --- BINNING LOGIC (Every 5 Sessions) ---
    df['group'] = df.index // 5
    grouped = df.groupby('group')
    df_mean = grouped.mean()
    df_sem = grouped.sem()  # Standard Error

    # x-axis for plotting (Groups 0, 1, 2...)
    x = df_mean.index

    # Common Errorbar Kwargs
    err_kws = dict(marker='o', capsize=3, markersize=5, linestyle='-', alpha=0.8)

    # 1. Consumption
    axes[0].errorbar(x, df_mean.consumption, yerr=df_sem.consumption, color='gray', **err_kws)
    axes[0].set_ylabel("Consumption (s)")

    # 2. Realized RR
    axes[1].errorbar(x, df_mean.rr_L, yerr=df_sem.rr_L, color=COLOR_LOW, label='Low', **err_kws)
    axes[1].errorbar(x, df_mean.rr_H, yerr=df_sem.rr_H, color=COLOR_HIGH, label='High', **err_kws)
    axes[1].set_ylabel("Reward Rate (rew/s)")

    # 3. Percent Engaged
    axes[2].errorbar(x, df_mean.eng_L, yerr=df_sem.eng_L, color=COLOR_LOW, **err_kws)
    axes[2].errorbar(x, df_mean.eng_H, yerr=df_sem.eng_H, color=COLOR_HIGH, **err_kws)
    axes[2].set_ylabel("% Engaged")

    # 4. Re-entry
    axes[3].errorbar(x, df_mean.ree_L, yerr=df_sem.ree_L, color=COLOR_LOW, **err_kws)
    axes[3].errorbar(x, df_mean.ree_H, yerr=df_sem.ree_H, color=COLOR_HIGH, **err_kws)
    axes[3].set_ylabel("Re-entry Index")

    for ax in axes:
        sns.despine(ax=ax)
        ax.set_xlabel("Five-session Group")

    axes[1].legend(fontsize='x-small')


# ==============================================================================
# SECTION 4: Survival Curves (Row 3)
# ==============================================================================

def actual_global_reward_rate(x, bg_stay, travel_bg2exp, travel_exp2bg, cumulative=8., starting=1.):
    a = starting
    b = a / cumulative
    local_gain = cumulative * (1 - np.exp(-b * x))
    global_gain = local_gain + 4
    rou_g = global_gain / (x + bg_stay + travel_bg2exp + travel_exp2bg)
    return rou_g

def calculate_actual_travel_time(trial_df):
    trial_df['bg_stay'] = trial_df['bg_exit'] - trial_df['bg_entry']
    bg_stay_low = trial_df.loc[trial_df['phase'] == '0.4', 'bg_stay'].mean()
    bg_stay_high = trial_df.loc[trial_df['phase'] == '0.8', 'bg_stay'].mean()
    # travel times from bg to exp
    travel_times_bg_to_exp = []
    for index, row in trial_df.iterrows():
        if row['bg_exit'] and row['exp_entry']:
            travel_time = row['exp_entry'] - row['bg_exit']
            travel_times_bg_to_exp.append(travel_time)
        else:
            travel_times_bg_to_exp.append(np.nan)
    trial_df['travel_time_bg_to_exp'] = travel_times_bg_to_exp
    # travel times from exp to bg in the next trial
    bg_entry_map = trial_df.set_index(['session', 'trial'])['bg_entry']
    next_trial_indices = [(row['session'], row['trial'] + 1) for index, row in trial_df.iterrows()]
    trial_df['next_trial_bg_entry'] = [bg_entry_map.get(idx) for idx in next_trial_indices]
    travel_times_exp_to_bg = []
    for index, row in trial_df.iterrows():
        if row['next_trial_bg_entry'] and row['exp_exit']:
            travel_time = row['next_trial_bg_entry'] - row['exp_exit']
            travel_times_exp_to_bg.append(travel_time)
        else:
            travel_times_exp_to_bg.append(np.nan)
    trial_df['travel_time_exp_to_bg'] = travel_times_exp_to_bg
    travel_bg2exp = trial_df['travel_time_bg_to_exp'].mean()
    travel_exp2bg = trial_df['travel_time_exp_to_bg'].mean()
    return bg_stay_low, bg_stay_high, travel_bg2exp, travel_exp2bg

def calculate_adjusted_optimal(trial_df):
    x = np.arange(0, 25, 0.1)
    bg_stay_low, bg_stay_high, travel_bg2exp, travel_exp2bg = calculate_actual_travel_time(trial_df)
    global_reward_rate_low = actual_global_reward_rate(x, bg_stay_low, travel_bg2exp, travel_exp2bg)
    global_reward_rate_high = actual_global_reward_rate(x, bg_stay_high, travel_bg2exp, travel_exp2bg)
    optimal_low = x[np.argmax(global_reward_rate_low)]
    optimal_high = x[np.argmax(global_reward_rate_high)]
    return np.round(optimal_low, 1), np.round(optimal_high, 1)


def perform_log_rank_test(trial_df):
    trial_df['leave_time'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df['event_observed'] = 1
    # log-rank test
    high_mask = trial_df['phase'] == '0.8'
    low_mask = trial_df['phase'] == '0.4'
    results = logrank_test(
        durations_A=trial_df.loc[high_mask, 'leave_time'],
        event_observed_A=trial_df.loc[high_mask, 'event_observed'],
        durations_B=trial_df.loc[low_mask, 'leave_time'],
        event_observed_B=trial_df.loc[low_mask, 'event_observed']
    )
    print("\nLog-rank test results:")
    results.print_summary()
    return results.summary


def plot_kaplan_meier_grid(trial_df, ax, title_str, opt_low=None, opt_high=None, p_str=None):
    kmf = KaplanMeierFitter()
    trial_df['leave_time'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df['event_observed'] = 1

    # Plot Curves
    mask_h = trial_df.phase == '0.8'
    if mask_h.sum() > 0:
        kmf.fit(trial_df.loc[mask_h, 'leave_time'], event_observed=trial_df.loc[mask_h, 'event_observed'])
        kmf.plot_survival_function(ax=ax, c=COLOR_HIGH, ci_show=True)

    mask_l = trial_df.phase == '0.4'
    if mask_l.sum() > 0:
        kmf.fit(trial_df.loc[mask_l, 'leave_time'], event_observed=trial_df.loc[mask_l, 'event_observed'])
        kmf.plot_survival_function(ax=ax, c=COLOR_LOW, ci_show=True)

    # Plot Optimal Lines
    if opt_low is not None:
        ax.axvline(x=opt_low, color=COLOR_LOW, linestyle='--', alpha=0.8, lw=1.5)
    if opt_high is not None:
        ax.axvline(x=opt_high, color=COLOR_HIGH, linestyle='--', alpha=0.8, lw=1.5)

    # Plot P-value
    if p_str:
        ax.text(0.95, 0.95, p_str, transform=ax.transAxes, ha='right', va='top', fontsize=9)

    ax.set_title(title_str, fontsize=10)
    ax.set_xlim(0, 25)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.get_legend().remove()
    sns.despine(ax=ax)

def plot_boxplot(trial_df, ax):
    """Specific user request: paired dots with offset."""
    df = trial_df.copy()
    df['exp_leave_time'] = df['exp_exit'] - df['exp_entry']
    median_df = df.groupby(['phase', 'animal'])['exp_leave_time'].median().reset_index()

    palette = {'0.4': COLOR_LOW, '0.8': COLOR_HIGH}

    # Boxplot
    sns.boxplot(x='phase', y='exp_leave_time', data=median_df, order=['0.4', '0.8'],
                palette=palette, ax=ax, width=0.4, showfliers=False, boxprops=dict(alpha=0.7))

    # Dot Offset Logic
    offset = 0.25
    median_df['x_offset'] = median_df['phase'].map({'0.4': 0 + offset, '0.8': 1 - offset})

    sns.lineplot(x='x_offset', y='exp_leave_time', data=median_df, units='animal', estimator=None,
                 ax=ax, color='gray', alpha=0.6, marker='o', markersize=6)

    # Stats
    g1 = median_df.loc[median_df['phase'] == '0.4', 'exp_leave_time'].values
    g2 = median_df.loc[median_df['phase'] == '0.8', 'exp_leave_time'].values
    if len(g1) > 0 and len(g2) > 0:
        _, p = wilcoxon(g1, g2)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        y_max = median_df['exp_leave_time'].max()
        bar_h = y_max * 1.05
        ax.plot([0, 0, 1, 1], [bar_h, bar_h + 0.5, bar_h + 0.5, bar_h], c='dimgray', lw=1)
        ax.text(0.5, bar_h + 0.6, sig, ha='center', va='bottom', color='black')

    ax.set_xticklabels(['Low', 'High'])
    ax.set_ylabel("Median Leave Time (s)")
    ax.set_xlabel("Context Reward\nRate (Block)")
    sns.despine(ax=ax)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # --- Data Setup ---
    animal_info = {
        "SZ036": {'day_0': '2023-11-30', 'sex': 'F'},
        "SZ037": {'day_0': '2023-11-30', 'sex': 'F'},
        "SZ038": {'day_0': '2023-11-30', 'sex': 'F'},
        "SZ039": {'day_0': '2023-11-30', 'sex': 'F'},
        "SZ042": {'day_0': '2023-11-30', 'sex': 'M'},
        "SZ043": {'day_0': '2023-11-30', 'sex': 'M'},
        "RK007": {'day_0': '2025-06-17', 'sex': 'M'},
        "RK008": {'day_0': '2025-06-17', 'sex': 'M'},
    }
    animal_ids = list(animal_info.keys())

    example_animal = "SZ039"
    example_session = "2023-12-30T20_44"

    # --- Grid Setup ---
    fig = plt.figure(figsize=(FIG_WIDTH, 12))
    gs_main = gridspec.GridSpec(3, 1, height_ratios=ROW_HEIGHT_RATIOS, hspace=0.35)

    # --- ROW 1 ---
    gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], width_ratios=ROW1_WIDTH_RATIOS, wspace=0.15)
    plot_optimality(fig.add_subplot(gs_r1[0]))
    ax_behav_example = fig.add_subplot(gs_r1[1])
    plot_behav_in_patches(ax_behav_example, example_animal, example_session)
    plot_behavior_legend(fig.add_subplot(gs_r1[2]))

    # --- ROW 2 ---
    gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1], wspace=0.4)
    axes_met = [fig.add_subplot(gs_r2[i]) for i in range(4)]
    plot_metrics_panel(axes_met, example_animal)

    # --- ROW 3: 4x2 Grid (Left) + Boxplot (Right) ---
    # Split Row 3 into Left (Grid) and Right (Boxplot)
    # Ratio 4:1 roughly allows the grid to be wide enough
    gs_r3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2], width_ratios=[5, 1], wspace=0.15)

    # Left: 4x2 Grid
    gs_grid = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_r3[0], hspace=0.4, wspace=0.1)

    all_trials_list = []
    print("Generating survival plots with optimality & stats...")
    for i, aid in enumerate(animal_ids):
        if i >= 8: break
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs_grid[r, c])

        df = data_loader.load_dataframes_for_animal_summary([aid], 'trial_df',
                                                            day_0=animal_info[aid]['day_0'],
                                                            hemisphere_qc=0)

        title_str = f"Animal {i + 1}, {animal_info[aid]['sex']}"

        if not df.empty:
            all_trials_list.append(df)

            # 1. OPTIMALITY
            try:
                opt_low, opt_high = calculate_adjusted_optimal(df)
            except Exception as e:
                print(f"Optimality calc failed for {aid}: {e}")
                opt_low, opt_high = None, None

            # 2. STATISTICS (Log-Rank)
            p_str = None
            try:
                res_df = perform_log_rank_test(df)
                if res_df is not None:
                    p_val = res_df.p.iloc[0]
                    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            except Exception as e:
                print(f"Log-rank failed for {aid}: {e}")

            plot_kaplan_meier_grid(df, ax, title_str, opt_low=opt_low, opt_high=opt_high, p_str=p_str)

            if c == 0:
                ax.set_ylabel("Proportion of Trials \nin Investment Port")
            else:
                ax.set_yticks([])
            if r == 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticks([])

    # Right: Boxplot
    if all_trials_list:
        master_df = pd.concat(all_trials_list, ignore_index=True)
        ax_box = fig.add_subplot(gs_r3[1])
        plot_boxplot(master_df, ax_box)
        ax_box.set_title("All Animals")

    # Lettering
    trans = ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    fig.axes[0].text(0.0, 1.0, 'a', transform=fig.axes[0].transAxes + trans, fontsize=16, weight='bold', va='bottom')
    ax_behav_example.text(0.0, 1.0, 'b', transform=ax_behav_example.transAxes + trans, fontsize=16, weight='bold', va='bottom')
    axes_met[0].text(0.0, 1.0, 'c', transform=axes_met[0].transAxes + trans, fontsize=16, weight='bold', va='bottom')
    axes_met[1].text(0.0, 1.0, 'd', transform=axes_met[1].transAxes + trans, fontsize=16, weight='bold', va='bottom')
    axes_met[2].text(0.0, 1.0, 'e', transform=axes_met[2].transAxes + trans, fontsize=16, weight='bold', va='bottom')
    axes_met[3].text(0.0, 1.0, 'f', transform=axes_met[3].transAxes + trans, fontsize=16, weight='bold', va='bottom')
    # Label 'c' on the first plot of the grid
    fig.axes[7].text(0.0, 1.0, 'g', transform=fig.axes[7].transAxes + trans, fontsize=16, weight='bold', va='bottom')
    ax_box.text(0.0, 1.0, 'h', transform=ax_box.transAxes + trans, fontsize=16, weight='bold', va='bottom')
    #
    save_path = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR, 'fig_3-1_behavior_composite.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()