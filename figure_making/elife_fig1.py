"""
eLife Figure 1 -- Task structure and behavioral strategy.

Panel layout mirrors Figure 1 of Sutlief et al. (eLife RP111930), which uses the same task:
    A  task schematic (image file; placeholder box if the file is missing)
    B  survival of patch residence vs. time from entry, split by context block
    C  discrete hazard of leaving vs. time from entry, split by context block
    D  survival vs. time from previous reward (censored at next reward), split by context block
    E  discrete hazard vs. time from previous reward, split by context block
    F  discrete hazard vs. time from previous reward, split by time of that reward from entry
    G  whitened linear-SVM coefficients (time from reward, time from entry, context)
    H  MVT optimality (global reward rate per block; optimal exit times)

Data extraction is reused from the analysis scripts so there is one source of truth:
    exploratory_data_analysis/hazard_analysis.py -> extract_survival_data
    exploratory_data_analysis/svm_analysis.py    -> build_continuous_session_data
Plotting follows the conventions of fig_dopamine.py: an explicit setup_axes_v2() grid, panel functions
that accept `axes=` (and make their own figure when axes is None), Set2[0]/Set2[1] for low/high block,
and a truncated 'Reds_r' palette for time-range categories.
"""
import os
import sys

# make the project root importable (for config / quality_control / exploratory_data_analysis)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matplotlib import gridspec
from matplotlib.transforms import ScaledTranslation
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
from itertools import combinations

from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.svm import SVC

import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import config
from exploratory_data_analysis.hazard_analysis import extract_survival_data
from exploratory_data_analysis.svm_analysis import build_continuous_session_data

mpl.rcParams['figure.dpi'] = 300

# --- shared palette (same as fig_dopamine.py) ---
BLOCK_PALETTE = sns.color_palette('Set2', 2)
COLOR_LOW, COLOR_HIGH = BLOCK_PALETTE[0], BLOCK_PALETTE[1]
BLOCK_VALUES = [0.4, 0.8]
BLOCK_LABELS = ['Low', 'High']
BLOCK_COLORS = [COLOR_LOW, COLOR_HIGH]


def time_range_palette(cat_num, palette='Reds_r'):
    # same trick as fig_dopamine: drop the lightest shade so the last category stays visible
    return list(sns.color_palette(palette, n_colors=cat_num + 1))[:cat_num]


def _p_to_stars(p_value):
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return 'ns'


def _draw_bracket(ax, x1, x2, y, h, annot, inset=0.05, col='k'):
    # same bracket style as figf_summary_block_split in fig_dopamine.py
    bx1, bx2 = x1 + inset, x2 - inset
    ax.plot([bx1, bx1, bx2, bx2], [y, y + h, y + h, y], lw=1, c=col)
    ax.text((bx1 + bx2) * 0.5, y + h / 2, annot, ha='center', va='bottom', color=col, fontsize=10)


def legend_right_of_curves(ax, vertical='upper', pad=0.03, x_right=None, **legend_kws):
    """
    Put the legend in the empty space to the right of where the curves end (as in the reference
    Figure 1), extending the x-axis when needed so the legend never overlaps the data.
    x_right: if given, the legend's right border is aligned with this x value (in data units) and
    the x-axis is extended to at least x_right; otherwise the legend's left border sits just after
    the curves end.
    Returns the (possibly extended) right x-limit.
    """
    if x_right is not None and ax.get_xlim()[1] < x_right:
        ax.set_xlim(ax.get_xlim()[0], x_right)
    x_min, x_max = ax.get_xlim()
    curve_end = max(np.nanmax(line.get_xdata()) for line in ax.get_lines() if len(line.get_xdata()))

    # measure the legend width (in axes fraction) by drawing it once
    legend = ax.legend(loc='upper left', **legend_kws)
    renderer = ax.figure.canvas.get_renderer()
    legend_frac = legend.get_window_extent(renderer).width / ax.get_window_extent(renderer).width
    legend.remove()

    # widen the x-axis if the curves leave too little room for the legend
    end_frac = (curve_end - x_min) / (x_max - x_min)
    if x_right is None and end_frac + pad + legend_frac > 1:
        x_max = x_min + (curve_end - x_min) / (1 - pad - legend_frac)
        x_max = np.ceil(x_max / 2) * 2  # round up to an even number of seconds
        ax.set_xlim(x_min, x_max)
        end_frac = (curve_end - x_min) / (x_max - x_min)

    y_anchor = {'upper': 1.0, 'center': 0.5, 'lower': 0.0}[vertical]
    if x_right is None:
        legend = ax.legend(loc=f'{vertical} left' if vertical != 'center' else 'center left',
                           bbox_to_anchor=(end_frac + pad, y_anchor), borderaxespad=0, **legend_kws)
    else:
        right_frac = (x_right - x_min) / (x_max - x_min)
        # flag (don't print yet): the panel may still be resized by the v2 layout; main() reports at the end
        ax._legend_overlaps = right_frac - legend_frac < end_frac + pad
        legend = ax.legend(loc=f'{vertical} right' if vertical != 'center' else 'center right',
                           bbox_to_anchor=(right_frac, y_anchor), borderaxespad=0, **legend_kws)
    legend.get_title().set_multialignment('center')  # centre each line of a multi-line title
    # remember the settings so the legend can be re-placed if the axes are resized later
    ax._legend_right_kws = dict(vertical=vertical, pad=pad, x_right=x_right, **legend_kws)
    return x_max


# --- data preparation (cached, because the SVM fits are slow) ---
def _cache_path(name):
    return os.path.join(config.MAIN_DATA_ROOT, config.PAPER_OUTPUT_SUBDIR, name)


def load_or_extract_survival_data(animal_ids, recompute=False):
    trial_path, reward_path = _cache_path('fig1_survival_trial_df.csv'), _cache_path('fig1_survival_reward_df.csv')
    if not recompute and os.path.exists(trial_path) and os.path.exists(reward_path):
        return pd.read_csv(trial_path), pd.read_csv(reward_path)
    trial_df, reward_df = extract_survival_data(animal_ids)
    trial_df.to_csv(trial_path, index=False)
    reward_df.to_csv(reward_path, index=False)
    return trial_df, reward_df


def compute_svm_coefficients(animal_ids):
    """
    Per-session linear SVM (stay=0 / leave=1) on whitened time_from_entry, time_from_reward, block_value.
    Same model as svm_analysis.get_svm_coefficients_and_predictions, without the (slow) second
    unwhitened fit that is only needed for exit-time projections.
    """
    records = []
    for animal in animal_ids:
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            continue
        n_sessions = len([f for f in os.listdir(processed_dir) if f.endswith('trial_df.parquet')])
        for session_id in range(n_sessions):
            cont_df = build_continuous_session_data(animal, session_id=session_id)
            if cont_df is None or len(cont_df) < 50:
                continue
            features = cont_df[['time_from_entry', 'time_from_reward', 'block_value']]
            if features['block_value'].nunique() < 2:
                continue  # block coefficient is undefined if the session only had one block
            features_whitened = (features - features.mean()) / features.std()
            model = SVC(kernel='linear', C=1)
            model.fit(features_whitened, cont_df['stay_leave'])
            coefs = model.coef_[0]
            records.append({'animal': animal, 'session_id': session_id,
                            'entry_coef': coefs[0], 'reward_coef': coefs[1], 'block_coef': coefs[2]})
    return pd.DataFrame(records)


def load_or_compute_svm_coefficients(animal_ids, recompute=False):
    path = _cache_path('fig1_svm_coefficients.csv')
    if not recompute and os.path.exists(path):
        return pd.read_csv(path)
    svm_df = compute_svm_coefficients(animal_ids)
    svm_df.to_csv(path, index=False)
    return svm_df


# --- MVT helpers (copied from fig3-1_optimality_and_example_behavior.py; that file can't be imported by name) ---
def global_reward_gain(x, cumulative=8., starting=1.):
    b = starting / cumulative
    return cumulative * (1 - np.exp(-b * x))


def global_reward_rate(x, context_time=5, consumption_time=0.5, travel_time=0.5, cumulative=8., starting=1.):
    # +4 represents the 4 fixed context rewards
    return (global_reward_gain(x, cumulative, starting) + 4) / (x + context_time + consumption_time + travel_time * 2)


def exp_decreasing(x, cumulative=8., starting=1.):
    b = starting / cumulative
    return starting * np.exp(-b * x)


def mvt_optimal_exit_times(x=np.arange(0, 15, 0.1)):
    # context_time = 10 s for the low block (4 rewards / 10 s) and 5 s for the high block (4 rewards / 5 s)
    return {0.4: x[np.argmax(global_reward_rate(x, context_time=10))],
            0.8: x[np.argmax(global_reward_rate(x, context_time=5))]}


# --- panel functions ---
def figa_task_schematic(image_path, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        return_handle = True
    else:
        fig = None
        return_handle = False
    if image_path is not None and os.path.exists(image_path):
        ax.imshow(mpimg.imread(image_path))
    else:
        ax.text(0.5, 0.5, 'Task schematic\n(add image file)', ha='center', va='center',
                fontsize='small', color='grey', transform=ax.transAxes)
        ax.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                           ec='lightgrey', ls='--'))
    ax.axis('off')
    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, ax


def plot_survival(df, duration_col, group_col, groups, colors, labels, max_x, axes=None):
    """Kaplan-Meier survival (same estimator as hazard_analysis.plot_sr), drawn as steps with 95% CI."""
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(3, 3))
        return_handle = True
    else:
        fig = None
        return_handle = False

    kmf = KaplanMeierFitter()
    for group, color, label in zip(groups, colors, labels):
        mask = np.isclose(df[group_col], group) if np.issubdtype(type(group), np.floating) else df[group_col] == group
        if mask.sum() == 0:
            continue
        kmf.fit(df.loc[mask, duration_col], event_observed=df.loc[mask, 'event_observed'])
        t_vec = kmf.survival_function_.index.values
        surv = kmf.survival_function_.iloc[:, 0].values
        ci = kmf.confidence_interval_survival_function_
        axes.step(t_vec, surv, where='post', color=color, linewidth=1, label=label)
        axes.fill_between(t_vec, ci.iloc[:, 0].values, ci.iloc[:, 1].values, step='post',
                          color=color, alpha=0.4, edgecolor='none')

    axes.set_xlim(0, max_x)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel('Proportion Remaining\nin Patch')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def plot_discrete_hazard(df, duration_col, group_col, groups, colors, labels, max_x, bin_width=0.5,
                         min_at_risk=5, ci_z=1.645, axes=None):
    """
    Discrete hazard P(exit in bin | still in port at bin start), same estimator as
    hazard_analysis.plot_discrete_hr. Shading is ci_z * binomial SE (1.645 -> 90% CI).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        return_handle = True
    else:
        fig = None
        return_handle = False

    bins = np.arange(0, max_x + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    for group, color, label in zip(groups, colors, labels):
        mask = np.isclose(df[group_col], group) if np.issubdtype(type(group), np.floating) else df[group_col] == group
        group_df = df[mask]
        if len(group_df) == 0:
            continue
        exits, _ = np.histogram(group_df.loc[group_df['event_observed'] == 1, duration_col].values, bins=bins)
        total_counts, _ = np.histogram(group_df[duration_col].values, bins=bins)
        at_risk = len(group_df) - np.insert(np.cumsum(total_counts)[:-1], 0, 0)

        hazard = np.full(len(exits), np.nan)
        se = np.full(len(exits), np.nan)
        valid = at_risk > 0
        hazard[valid] = exits[valid] / at_risk[valid]
        se[valid] = np.sqrt(hazard[valid] * (1 - hazard[valid]) / at_risk[valid])

        plot_mask = at_risk >= min_at_risk
        axes.plot(bin_centers[plot_mask], hazard[plot_mask], color=color, linewidth=1, label=label)
        axes.fill_between(bin_centers[plot_mask],
                          np.maximum(0, hazard[plot_mask] - ci_z * se[plot_mask]),
                          np.minimum(1, hazard[plot_mask] + ci_z * se[plot_mask]),
                          color=color, alpha=0.4, edgecolor='none')

    axes.set_xlim(0, max_x)
    axes.set_ylim(bottom=0)
    axes.set_ylabel('Exit Probability')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def _logrank_by_block(df, duration_col):
    low, high = np.isclose(df['context'], 0.4), np.isclose(df['context'], 0.8)
    res = logrank_test(df.loc[low, duration_col], df.loc[high, duration_col],
                       event_observed_A=df.loc[low, 'event_observed'],
                       event_observed_B=df.loc[high, 'event_observed'])
    print(f'Log-rank ({duration_col}, low vs high): chi2 = {res.test_statistic:.2f}, p = {res.p_value:.3e}')
    return res


def figb_survival_from_entry(trial_df, axes=None):
    _logrank_by_block(trial_df, 'time_from_entry')
    out = plot_survival(trial_df, 'time_from_entry', 'context', BLOCK_VALUES, BLOCK_COLORS, BLOCK_LABELS,
                        max_x=20, axes=axes)
    ax = axes if axes is not None else out[1]
    # dashed lines at the MVT-optimal exit times, drawn the same way as in panel H
    for block, t_opt in mvt_optimal_exit_times().items():
        color = COLOR_LOW if block == 0.4 else COLOR_HIGH
        ax.axvline(x=t_opt, color=color, linestyle='--', lw=1)
    ax.set_xlabel('Time from Entry (s)')
    # block legend is shown once, here on the first panel where the blocks appear
    legend = ax.legend(title='Context\nReward Rate', title_fontsize='small', fontsize='medium', frameon=True,
                       edgecolor='lightgrey', fancybox=False, loc='upper right')
    legend.get_title().set_multialignment('center')
    return out


def figc_hazard_from_entry(trial_df, min_at_risk=5, axes=None):
    out = plot_discrete_hazard(trial_df, 'time_from_entry', 'context', BLOCK_VALUES, BLOCK_COLORS, BLOCK_LABELS,
                               max_x=20, min_at_risk=min_at_risk, axes=axes)
    ax = axes if axes is not None else out[1]
    ax.set_xlabel('Time from Entry (s)')
    # block legend is shown once, on the survival-from-entry panel
    return out


def figd_survival_from_reward(reward_df, axes=None):
    _logrank_by_block(reward_df, 'time_from_prev_reward')
    out = plot_survival(reward_df, 'time_from_prev_reward', 'context', BLOCK_VALUES, BLOCK_COLORS, BLOCK_LABELS,
                        max_x=10, axes=axes)
    ax = axes if axes is not None else out[1]
    ax.set_xlabel('Time from Reward (s)')
    return out


def fige_hazard_from_reward_by_block(reward_df, min_at_risk=5, axes=None):
    out = plot_discrete_hazard(reward_df, 'time_from_prev_reward', 'context', BLOCK_VALUES, BLOCK_COLORS,
                               BLOCK_LABELS, max_x=10, min_at_risk=min_at_risk, axes=axes)
    ax = axes if axes is not None else out[1]
    ax.set_xlabel('Time from Reward (s)')
    return out


def figf_hazard_from_reward_by_entry_time(reward_df, bins=(0, 3, 6, 9, 12), min_at_risk=5, legend_x_right=20,
                                          axes=None):
    df = reward_df.copy()
    bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    df['entry_bin'] = pd.cut(df['time_from_entry_at_reward'], bins=list(bins), labels=bin_labels)
    out = plot_discrete_hazard(df, 'time_from_prev_reward', 'entry_bin', bin_labels,
                               time_range_palette(len(bin_labels)), bin_labels, max_x=10,
                               min_at_risk=min_at_risk, axes=axes)
    ax = axes if axes is not None else out[1]
    ax.set_xlabel('Time from Reward (s)')
    ax.set_ylabel('')
    # legend sits in the empty space to the right of the curves; latest reward times on top, as in the reference
    handles, labels = ax.get_legend_handles_labels()
    legend_right_of_curves(ax, vertical='upper', x_right=legend_x_right, handles=handles[::-1], labels=labels[::-1],
                           title='Reward Time\nfrom Entry (s)', title_fontsize='small', fontsize='medium',
                           frameon=True, edgecolor='lightgrey', fancybox=False)
    return out


def figg_svm_coefficients(svm_df, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        return_handle = True
    else:
        fig = None
        return_handle = False

    coef_cols = ['reward_coef', 'entry_coef', 'block_coef']
    tick_labels = ['Time from\nReward', 'Time from\nEntry', 'Context']
    long_df = svm_df.melt(id_vars=['animal', 'session_id'], value_vars=coef_cols,
                          var_name='predictor', value_name='coef')

    print('--- SVM coefficients: one-sample t-tests vs 0 ---')
    for col in coef_cols:
        t_stat, p_val = stats.ttest_1samp(svm_df[col], 0)
        print(f'{col}: mean = {svm_df[col].mean():.3f}, t = {t_stat:.2f}, p = {p_val:.3e}')

    sns.boxplot(data=long_df, x='predictor', y='coef', order=coef_cols, hue='predictor', hue_order=coef_cols,
                palette='Greys', width=0.5, showfliers=False, legend=False,
                medianprops={'linewidth': 1, 'color': 'black'}, ax=axes)
    sns.stripplot(data=long_df, x='predictor', y='coef', order=coef_cols, color='black', alpha=0.3, size=2,
                  jitter=0.15, ax=axes)
    axes.axhline(0, linestyle='--', color='k', alpha=0.5, linewidth=1)

    # pairwise paired t-tests, brackets stacked above the data
    print('--- SVM coefficients: paired t-tests ---')
    y_max = long_df['coef'].max()
    y_range = long_df['coef'].max() - long_df['coef'].min()
    h, step = 0.02 * y_range, 0.09 * y_range
    pairs = [(0, 1), (1, 2), (0, 2)]
    for level, (i, j) in enumerate(pairs):
        t_stat, p_val = stats.ttest_rel(svm_df[coef_cols[i]], svm_df[coef_cols[j]])
        print(f'{coef_cols[i]} vs {coef_cols[j]}: t = {t_stat:.2f}, p = {p_val:.3e}')
        # the two adjacent pairs share the first level; the long pair goes above them
        y = y_max + step * (0.5 if level < 2 else 1.5)
        _draw_bracket(axes, i, j, y, h, _p_to_stars(p_val))
    axes.set_ylim(top=y_max + step * 2.5)

    axes.set_xticks(range(3), tick_labels, fontsize='small')
    axes.set_xlabel('')
    axes.set_ylabel('SVM Coefficient')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def figh_mvt_optimality(axes=None):
    """Adapted from plot_optimality in fig3-1_optimality_and_example_behavior.py."""
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        return_handle = True
    else:
        fig = None
        return_handle = False

    x = np.arange(0, 15, 0.1)
    x_inst = np.arange(0.1, 15, 0.1)
    axes.plot(x_inst, exp_decreasing(x_inst), color='grey', alpha=0.6, linewidth=1, label='Instantaneous')
    for context_time, color, label in zip([10, 5], BLOCK_COLORS, BLOCK_LABELS):
        y = global_reward_rate(x, context_time=context_time)
        x_opt, y_opt = x[np.argmax(y)], np.max(y)
        axes.plot(x, y, color=color, linewidth=1, label=f'Global ({label})')
        axes.scatter(x_opt, y_opt, color=color, s=10, zorder=5)
        axes.vlines(x=x_opt, ymin=0, ymax=y_opt, color=color, linestyle='--', lw=1)

    axes.set_xlim(0, 15)
    axes.set_ylim(0, 1.05)
    axes.set_xlabel('Time from Entry (s)')
    axes.set_ylabel('Reward Rate (rewards/s)')
    axes.legend(fontsize='medium', frameon=True, edgecolor='lightgrey', fancybox=False, loc='upper right', labelspacing=0.2, handlelength=1.0)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


# --- axes layout ---
def setup_axes_v1():
    fig_size = (9, 12)  # (width, height) in inches
    rows, cols = fig_size[1] * 10, fig_size[0] * 10

    # horizontal splits for each figure row (relative widths) and the margins between panels
    row_1 = [1]  # A schematic
    row_2 = [1, 2]  # B survival from entry, C hazard from entry
    row_3 = [1, 1, 1]  # D survival from reward, E hazard by block, F hazard by entry time
    row_4 = [1, 1]  # G SVM coefficients, H MVT optimality
    row_2_margins = [10]
    row_3_margins = [10, 6]
    row_4_margins = [14]

    # vertical splits (relative heights) and the margins between figure rows
    col_1 = [4, 2.5, 2.5, 3]
    col_1_margins = [9, 11, 11]

    def _splits(widths, margins, total):
        sizes = [int((total - np.sum(margins)) * w / sum(widths)) for w in widths]
        edges = np.cumsum([val for pair in zip(sizes, list(margins) + [0]) for val in pair][:-1])
        edges[-1] = total
        return np.insert(edges, 0, 0)  # [start_0, end_0, start_1, end_1, ...] once paired below

    r2 = _splits(row_2, row_2_margins, cols)
    r3 = _splits(row_3, row_3_margins, cols)
    r4 = _splits(row_4, row_4_margins, cols)
    c1 = _splits(col_1, col_1_margins, rows)

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    axes = [
        fig.add_subplot(gs[c1[0]:c1[1], :]),  # A task schematic

        fig.add_subplot(gs[c1[2]:c1[3], r2[0]:r2[1]]),  # B survival vs. time from entry
        fig.add_subplot(gs[c1[2]:c1[3], r2[2]:r2[3]]),  # C hazard vs. time from entry

        fig.add_subplot(gs[c1[4]:c1[5], r3[0]:r3[1]]),  # D survival vs. time from reward
        fig.add_subplot(gs[c1[4]:c1[5], r3[2]:r3[3]]),  # E hazard vs. time from reward, by block
        fig.add_subplot(gs[c1[4]:c1[5], r3[4]:r3[5]]),  # F hazard vs. time from reward, by entry time

        fig.add_subplot(gs[c1[6]:c1[7], r4[0]:r4[1]]),  # G SVM coefficients
        fig.add_subplot(gs[c1[6]:c1[7], r4[2]:r4[3]]),  # H MVT optimality
    ]

    lettering = 'ABCDEFGHIJKLMN'
    axes_to_letter = [0, 1, 2, 3, 4, 5, 6, 7]
    for i, ax in enumerate(axes_to_letter):
        axes[ax].text(
            0.0, 1.0, lettering[i], transform=(
                    axes[ax].transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif', weight='bold')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.05)
    axes = np.array(axes)
    return axes


def setup_axes_v2():
    """
    Row 2: survival curves (B from entry, C from reward). Row 3: hazard curves (D from entry,
    E from reward by block, F from reward by entry time). Panel widths in rows 2 and 3 are set
    after plotting by match_time_scale(), so equal lengths on the x-axis mean equal time.
    Axes are returned in the same order as setup_axes_v1 (i.e. by plotting function), so main()
    works with either layout:
        [A, surv. from entry, hazard from entry, surv. from reward, hazard by block, hazard by entry, G, H]
    """
    fig_size = (9, 12)  # (width, height) in inches
    rows, cols = fig_size[1] * 10, fig_size[0] * 10

    # starting widths ~ the x range in seconds; refined in match_time_scale()
    row_2 = [20, 10]  # B survival from entry, C survival from reward
    row_3 = [20, 10, 10]  # D hazard from entry, E hazard by block, F hazard by entry time
    row_4 = [1, 1]  # G SVM coefficients, H MVT optimality
    row_2_margins = [10]
    row_3_margins = [10, 6]
    row_4_margins = [14]

    col_1 = [4, 2.5, 2.5, 3]
    col_1_margins = [9, 11, 11]

    def _splits(widths, margins, total):
        sizes = [int((total - np.sum(margins)) * w / sum(widths)) for w in widths]
        edges = np.cumsum([val for pair in zip(sizes, list(margins) + [0]) for val in pair][:-1])
        edges[-1] = total
        return np.insert(edges, 0, 0)

    r2 = _splits(row_2, row_2_margins, cols)
    r3 = _splits(row_3, row_3_margins, cols)
    r4 = _splits(row_4, row_4_margins, cols)
    c1 = _splits(col_1, col_1_margins, rows)

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    ax_a = fig.add_subplot(gs[c1[0]:c1[1], :])  # A task schematic
    ax_b = fig.add_subplot(gs[c1[2]:c1[3], r2[0]:r2[1]])  # B survival vs. time from entry
    ax_c = fig.add_subplot(gs[c1[2]:c1[3], r2[2]:r2[3]])  # C survival vs. time from reward
    ax_d = fig.add_subplot(gs[c1[4]:c1[5], r3[0]:r3[1]])  # D hazard vs. time from entry
    ax_e = fig.add_subplot(gs[c1[4]:c1[5], r3[2]:r3[3]])  # E hazard vs. time from reward, by block
    ax_f = fig.add_subplot(gs[c1[4]:c1[5], r3[4]:r3[5]])  # F hazard vs. time from reward, by entry time
    ax_g = fig.add_subplot(gs[c1[6]:c1[7], r4[0]:r4[1]])  # G SVM coefficients
    ax_h = fig.add_subplot(gs[c1[6]:c1[7], r4[2]:r4[3]])  # H MVT optimality

    lettering = 'ABCDEFGHIJKLMN'
    for i, ax in enumerate([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h]):
        ax.text(
            0.0, 1.0, lettering[i], transform=(
                    ax.transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif', weight='bold')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.05)

    # same order as setup_axes_v1 so the plotting calls in main() don't change
    axes = np.array([ax_a, ax_b, ax_d, ax_c, ax_e, ax_f, ax_g, ax_h])
    time_scaled_rows = [[ax_b, ax_c], [ax_d, ax_e, ax_f]]
    return axes, time_scaled_rows


def match_time_scale(row_axes):
    """
    Resize the panels of one figure row so their widths are proportional to their x ranges
    (same seconds per inch), keeping the row's outer edges and the gaps between panels.
    """
    pos = [ax.get_position() for ax in row_axes]
    gaps = [pos[i + 1].x0 - pos[i].x1 for i in range(len(pos) - 1)]
    spans = [np.diff(ax.get_xlim())[0] for ax in row_axes]
    unit = (pos[-1].x1 - pos[0].x0 - sum(gaps)) / sum(spans)  # figure fraction per second
    x0 = pos[0].x0
    for ax, p, span, gap in zip(row_axes, pos, spans, gaps + [0]):
        ax.set_position([x0, p.y0, span * unit, p.height])
        x0 += span * unit + gap


def apply_time_scale(time_scaled_rows, sync_pairs=(), n_iter=4):
    """
    Iterate: rescale panel widths -> re-place legends that sit right of the curves (their
    width in axes units changed, so the x-axis may need extending) -> re-sync shared x-axes.
    Stops once the x limits no longer change.
    """
    fig = time_scaled_rows[0][0].figure
    for _ in range(n_iter):
        before = [ax.get_xlim() for row in time_scaled_rows for ax in row]
        for row in time_scaled_rows:
            match_time_scale(row)
        fig.canvas.draw()  # update the renderer with the new axes sizes
        for row in time_scaled_rows:
            for ax in row:
                kws = getattr(ax, '_legend_right_kws', None)
                if kws is not None:
                    legend_right_of_curves(ax, **kws)
        for follower, leader in sync_pairs:
            follower.set_xlim(leader.get_xlim())
        if before == [ax.get_xlim() for row in time_scaled_rows for ax in row]:
            break
    for row in time_scaled_rows:
        match_time_scale(row)


def main(recompute=False, save_fig=False, layout='v2'):
    # --- data preparation ---
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]

    tic = time.time()
    trial_df, reward_df = load_or_extract_survival_data(animal_ids, recompute=recompute)
    print(f'survival data: {len(trial_df)} trials, {len(reward_df)} reward intervals '
          f'({time.time() - tic:.2f} s)')

    tic = time.time()
    svm_df = load_or_compute_svm_coefficients(animal_ids, recompute=recompute)
    print(f'SVM coefficients: {len(svm_df)} sessions ({time.time() - tic:.2f} s)')

    schematic_path = os.path.join(config.MAIN_DATA_ROOT, config.PAPER_OUTPUT_SUBDIR, 'behavior_task_schematic.png')
    # --- end of data preparation ---

    # --- set up axes and add figures to axes ---
    if layout == 'v1':
        axes, time_scaled_rows = setup_axes_v1(), None
    else:
        axes, time_scaled_rows = setup_axes_v2()

    figa_task_schematic(schematic_path, ax=axes[0])
    figb_survival_from_entry(trial_df, axes=axes[1])
    figc_hazard_from_entry(trial_df, min_at_risk=40, axes=axes[2])
    figd_survival_from_reward(reward_df, axes=axes[3])
    fige_hazard_from_reward_by_block(reward_df, min_at_risk=30, axes=axes[4])
    figf_hazard_from_reward_by_entry_time(reward_df, min_at_risk=30, axes=axes[5])
    figg_svm_coefficients(svm_df, axes=axes[6])
    figh_mvt_optimality(axes=axes[7])

    # E and F share a y-axis so the hazard curves are directly comparable
    y_max = max(axes[4].get_ylim()[1], axes[5].get_ylim()[1])
    for ax in axes[4:6]:
        ax.set_ylim(0, y_max)

    if time_scaled_rows is None:
        # v1: F's x-axis may have been widened to fit its legend, so E follows it
        axes[4].set_xlim(axes[5].get_xlim())
    else:
        # v2: equal seconds per inch within rows 2 and 3, which already makes E and F comparable,
        # so E keeps its own x range instead of inheriting the space F needs for its legend
        apply_time_scale(time_scaled_rows)

    for ax in axes:
        if getattr(ax, '_legend_overlaps', False):
            print('Warning: a right-aligned legend overlaps its curves; increase legend_x_right for that panel.')

    if save_fig:
        out_dir = os.path.join(config.MAIN_DATA_ROOT, config.PAPER_OUTPUT_SUBDIR)
        plt.savefig(os.path.join(out_dir, 'elife_fig1_v2.png'), dpi=300)
        plt.savefig(os.path.join(out_dir, 'elife_fig1_v2.pdf'))
    plt.show()
    # --- end of adding figures to axes


if __name__ == '__main__':
    main(recompute=False, save_fig=True)