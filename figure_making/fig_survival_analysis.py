import numpy as np
import data_loader
import helper
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kaplan_meier(trial_df, title_head=None):
    trial_df['leave_time'] = trial_df['exp_exit'] - trial_df['exp_entry']
    trial_df['event_observed'] = 1

    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)
    color_palette = sns.color_palette("Set2")
    groups = {
        '0.8': {'label': 'high', 'color': color_palette[1]},
        '0.4': {'label': 'low', 'color': color_palette[0]}
    }
    for phase, settings in groups.items():
        mask = trial_df['phase'] == phase
        kmf.fit(
            durations=trial_df.loc[mask, 'leave_time'],
            event_observed=trial_df.loc[mask, 'event_observed'],
            label=settings['label']
        )
        kmf.plot_survival_function(ax=ax, c=settings['color'])
    if title_head is not None:
        plt.title(f"{title_head}: K-M Estimate of Survival")
    else:
        plt.title('Kaplan-Meier Estimate of Survival')
    plt.xlabel('Time from Entry (sec)')
    plt.ylabel('Stay Probability')


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
    # animal_str = 'SZ036'
    # for i in range(0, 15):
    #     session_name = '2024-01-08T13_52'
    #     trial_df = data_loader.load_session_dataframe(animal_str, 'trial_df',
    #                                                   session_id=i,
    #                                                   file_format='parquet')
    #     plot_kaplan_meier(trial_df)


    for animal in ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]:
        animal_ids = [animal]
        master_df = data_loader.load_dataframes_for_animal_summary(animal_ids, 'trial_df',
                                                                day_0='2023-11-30', hemisphere_qc=0, file_format='parquet')
        optimal_low, optimal_high = calculate_adjusted_optimal(master_df)
        color_palette = sns.color_palette("Set2")
        fig, ax = plt.subplots()
        plot_kaplan_meier(master_df, title_head=animal_ids[0])
        plt.axvline(x=optimal_high, color=color_palette[1], linestyle='--', label=f'Optimal for high')
        plt.axvline(x=optimal_low, color=color_palette[0], linestyle='--', label=f'Optimal for low')
        plt.legend()
        plt.ylim([0, 1])
        results = perform_log_rank_test(master_df)
        p = results["p"].to_numpy()[0]
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        ax.text(0.6 * x_max, 0.5 * y_max, f"p=\n{p}")
        plt.savefig(f"{animal_ids[0]}_K-M_survival_curves.png")
        fig.show()
        plt.close()

    for animal in ["RK007", "RK008"]:
        animal_ids = [animal]
        master_df = data_loader.load_dataframes_for_animal_summary(animal_ids, 'trial_df',
                                                                day_0='2025-06-17', hemisphere_qc=0, file_format='parquet')
        optimal_low, optimal_high = calculate_adjusted_optimal(master_df)
        color_palette = sns.color_palette("Set2")
        fig, ax = plt.subplots()
        plot_kaplan_meier(master_df, title_head=animal_ids[0])
        plt.axvline(x=optimal_high, color=color_palette[1], linestyle='--', label=f'Optimal for high')
        plt.axvline(x=optimal_low, color=color_palette[0], linestyle='--', label=f'Optimal for low')
        plt.legend()
        plt.ylim([0, 1])
        results = perform_log_rank_test(master_df)
        p = results["p"].to_numpy()[0]
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        ax.text(0.6 * x_max, 0.5 * y_max, f"p=\n{p}")
        plt.savefig(f"{animal_ids[0]}_K-M_survival_curves.png")
        fig.show()
        plt.close()



if __name__ == '__main__':
    main()
