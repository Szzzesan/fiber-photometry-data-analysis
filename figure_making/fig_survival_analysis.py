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

def main():
    # animal_str = 'SZ036'
    # for i in range(0, 15):
    #     session_name = '2024-01-08T13_52'
    #     trial_df = data_loader.load_session_dataframe(animal_str, 'trial_df',
    #                                                   session_id=i,
    #                                                   file_format='parquet')
    #     plot_kaplan_meier(trial_df)


    # for animal in ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]:
    #     animal_ids = [animal]
    #     master_df = data_loader.load_dataframes_for_animal_summary(animal_ids, 'trial_df',
    #                                                             day_0='2023-11-30', hemisphere_qc=0, file_format='parquet')
    #     fig, ax = plt.subplots()
    #     plot_kaplan_meier(master_df, title_head=animal_ids[0])
    #     results = perform_log_rank_test(master_df)
    #     p = results["p"].to_numpy()[0]
    #     x_min, x_max = plt.xlim()
    #     y_min, y_max = plt.ylim()
    #     ax.text(0.7 * x_max, 0.8 * y_max, f"p=\n{p}")
    #     plt.savefig(f"{animal_ids[0]}_K-M_survival_curves.png")
    #     fig.show()
    #     plt.close()

    for animal in ["RK007", "RK008"]:
        animal_ids = [animal]
        master_df = data_loader.load_dataframes_for_animal_summary(animal_ids, 'trial_df',
                                                                day_0='2025-06-17', hemisphere_qc=0, file_format='parquet')
        fig, ax = plt.subplots()
        plot_kaplan_meier(master_df, title_head=animal_ids[0])
        results = perform_log_rank_test(master_df)
        p = results["p"].to_numpy()[0]
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        ax.text(0.7 * x_max, 0.8 * y_max, f"p=\n{p}")
        plt.savefig(f"{animal_ids[0]}_K-M_survival_curves.png")
        fig.show()
        plt.close()



if __name__ == '__main__':
    main()
