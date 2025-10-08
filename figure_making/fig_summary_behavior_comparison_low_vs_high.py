import numpy as np
import pandas as pd
import data_loader
import helper
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_compare_leavetime(trial_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        return_handle = True
    else:
        fig = None
        return_handle = False

    df = trial_df.copy()
    df['exp_leave_time'] = df['exp_exit'] - df['exp_entry']
    median_df = df.groupby(['phase', 'animal'])['exp_leave_time'].median().reset_index()

    # box plot and dots
    palette = {
        '0.4': sns.color_palette('Set2')[0],
        '0.8': sns.color_palette('Set2')[1]
    }
    sns.boxplot(
        x='phase',
        y='exp_leave_time',
        data=median_df,
        order=['0.4', '0.8'],
        palette=palette,
        ax=ax,
        width=0.4,
        showfliers=False,
        boxprops=dict(alpha=0.7)
    )
    offset = 0.25
    median_df['x_offset'] = median_df['phase'].map({'0.4': 0 + offset,
                                                    '0.8': 1 - offset})
    sns.lineplot(
        x='x_offset',
        y='exp_leave_time',
        data=median_df,
        units='animal',
        estimator=None,
        ax=ax,
        color='gray',
        alpha=0.6,
        marker='o',
        markersize=6
    )
    ax.set_ylabel('Median Leave Time (sec)')
    ax.set_xlabel('Context Reward Rate')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['low', 'high'])

    # stats & annotation: Wilcoxon Signed-rank test
    group1 = median_df.loc[median_df['phase'] == '0.4', 'exp_leave_time'].to_numpy()
    group2 = median_df.loc[median_df['phase'] == '0.8', 'exp_leave_time'].to_numpy()
    stat, p_value = wilcoxon(group1, group2)
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'  # for "not significant"
    y_max = median_df['exp_leave_time'].max()
    y_offset = 0.1 * y_max
    bar_height = y_max + y_offset
    bar_tips_height = bar_height - (0.02 * y_max)
    x1, x2 = 0, 1
    ax.plot([x1, x1, x2, x2], [bar_tips_height, bar_height, bar_height, bar_tips_height], c='dimgray')
    ax.text((x1 + x2) * 0.5, bar_height, sig_symbol, ha='center', va='bottom', color='black')
    ax.set_ylim(top=bar_height + y_offset * 0.5)

    if return_handle:
        plt.tight_layout()
        fig.show()
        return fig, ax

def plot_kaplan_meier(trial_df, ax=None, add_labels=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        return_handle = True
    else:
        fig = None
        return_handle = False

    trial_df['leave_time'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df['event_observed'] = 1

    kmf = KaplanMeierFitter()
    color_palette = sns.color_palette("Set2")
    groups = {
        '0.8': {'label': 'Observed in High', 'color': color_palette[1]},
        '0.4': {'label': 'Observed in Low', 'color': color_palette[0]}
    }
    for phase, settings in groups.items():
        mask = trial_df['phase'] == phase
        # Use the `add_labels` flag to decide whether to add a legend label
        label = settings['label'] if add_labels else None
        kmf.fit(
            durations=trial_df.loc[mask, 'leave_time'],
            event_observed=trial_df.loc[mask, 'event_observed'],
            label=label
        )
        kmf.plot_survival_function(ax=ax, c=settings['color'], xlabel='')

    if return_handle:
        plt.tight_layout()
        fig.show()
        return fig, ax


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
    global_reward_rate_low = _global_reward_rate(x, bg_stay_low, travel_bg2exp, travel_exp2bg)
    global_reward_rate_high = _global_reward_rate(x, bg_stay_high, travel_bg2exp, travel_exp2bg)
    optimal_low = x[np.argmax(global_reward_rate_low)]
    optimal_high = x[np.argmax(global_reward_rate_high)]
    return np.round(optimal_low, 1), np.round(optimal_high, 1)

def _global_reward_rate(x, bg_stay, travel_bg2exp, travel_exp2bg, cumulative=8., starting=1.):
    a = starting
    b = a / cumulative
    local_gain = cumulative * (1 - np.exp(-b * x))
    global_gain = local_gain + 4
    rou_g = global_gain / (x + bg_stay + travel_bg2exp + travel_exp2bg)
    return rou_g


def main():
    # --- 1. Setup Animals and Figure ---
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
    color_palette = sns.color_palette("Set2")

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    axes_flat = axes.flatten()  # Flatten the 4x2 array of axes for easy looping

    # --- 2. Loop Through Animals and Plot ---
    for i, animal_id in enumerate(animal_ids):
        ax = axes_flat[i]  # Get the current subplot to draw on
        print(f"Processing data for {animal_id}...")

        # Load data for the current animal
        day_0 = animal_info[animal_id]['day_0']
        sex = animal_info[animal_id]['sex']
        master_df = data_loader.load_dataframes_for_animal_summary(
            [animal_id], 'trial_df', day_0=day_0, hemisphere_qc=0, file_format='parquet'
        )

        # --- Plot survival curves ---
        # `add_labels=(i==0)` means we only add legend labels for the very first plot.
        plot_kaplan_meier(master_df, ax=ax, add_labels=(i == 0))

        # --- Calculate and plot optimal times ---
        optimal_low, optimal_high = calculate_adjusted_optimal(master_df)

        # Add labels only for the first plot to contribute to the shared legend
        if i == 0:
            ax.axvline(x=optimal_high, color=color_palette[1], linestyle='--', label='Optimal in High')
            ax.axvline(x=optimal_low, color=color_palette[0], linestyle='--', label='Optimal in Low')
        else:
            ax.axvline(x=optimal_high, color=color_palette[1], linestyle='--')
            ax.axvline(x=optimal_low, color=color_palette[0], linestyle='--')

        # --- Perform log-rank test and add p-value text ---
        results_summary = perform_log_rank_test(master_df)
        p_value = results_summary["p"].iloc[0]
        p_text = f'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
        ax.text(0.95, 0.95, p_text, ha='right', va='top', transform=ax.transAxes)
        ax.set_title(f'{animal_id}, {sex}')
        ax.get_legend().remove()

    # --- 3. Final Figure Adjustments ---
    plt.setp(axes, xlim=(0, 25), ylim=(0, 1.05))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize='small', bbox_to_anchor=(0.28, 0.85))
    fig.supxlabel('Time from Entry (sec)')
    fig.supylabel('Proportion of Trials Still in Port')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # rect leaves space for suptitle if needed

    # Save the final figure
    plt.savefig("all_animals_survival_curves.png", dpi=300)
    plt.show()



if __name__ == '__main__':
    # main()
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    master_df1 = data_loader.load_dataframes_for_animal_summary(animal_ids, 'trial_df',
                                                                day_0='2023-11-30', hemisphere_qc=0, file_format='parquet')

    animal_ids = ["RK007", "RK008"]
    master_df2 = data_loader.load_dataframes_for_animal_summary(animal_ids, 'trial_df',
                                                                day_0='2025-06-17', hemisphere_qc=0, file_format='parquet')
    master_trial_df = pd.concat([master_df1, master_df2], ignore_index=True)

    boxplot_compare_leavetime(trial_df=master_trial_df)
