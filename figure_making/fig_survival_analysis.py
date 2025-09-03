import data_loader
import helper
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kaplan_meier(trial_df):
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
    plt.title('Kaplan-Meier Estimate of Survival')
    plt.xlabel('Time from Entry (sec)')
    plt.ylabel('Stay Probability')
    plt.grid(True)
    plt.show()

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

def main():
    animal_str = 'SZ036'
    for i in range(0, 15):
        session_name = '2024-01-08T13_52'
        trial_df = data_loader.load_session_dataframe(animal_str, 'trial_df',
                                                      session_id=i,
                                                      file_format='parquet')
        plot_kaplan_meier(trial_df)



if __name__ == '__main__':
    main()
