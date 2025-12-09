import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_nonreward_DA(nonreward_DA_df, fig_title, bin_size=0.5):
    binned_results_all, binned_results_split = bin_nonreward_DA(nonreward_DA_df, bin_size=bin_size)

    phases = ['0.4', '0.8']
    color_map = {
        '0.4': sns.color_palette('Set2')[0],  # Blue/Green
        '0.8': sns.color_palette('Set2')[1]  # Orange/Yellow
    }
    label_map = {
        '0.4': 'low',  # Blue/Green
        '0.8': 'high'  # Orange/Yellow
    }
    errorbar_params = {
        'fmt': 'o',
        'ms': 3,
        'capsize': 5,
        'elinewidth': 1.5,
        'linestyle': '--'
    }

    for branch in ['green_left', 'green_right']:
        fig, axes = plt.subplots(2, 1, sharex=True)
        y_mean_all = binned_results_all[(branch, 'mean')]
        y_sem_all = binned_results_all[(branch, 'sem')]
        axes[0].errorbar(
            x=binned_results_all['bin_center'],
            y=y_mean_all,
            yerr=y_sem_all,
            color='darkblue',
            ecolor='gray',
            **errorbar_params
        )
        axes[0].set_ylabel('Mean DA (z-score)')

        for phase_val in phases:
            group_data = binned_results_split.loc[binned_results_split['phase'] == phase_val]
            current_color = color_map[phase_val]
            label = label_map[phase_val]
            y_mean_split = group_data[(branch, 'mean')]
            y_sem_split = group_data[(branch, 'sem')]

            axes[1].errorbar(
                x=group_data['bin_center'],
                y=y_mean_split,
                yerr=y_sem_split,
                color=current_color,
                ecolor=current_color,
                label=label,
                **errorbar_params
            )

        axes[1].set_ylabel('Mean DA (z-score)')
        axes[1].set_xlabel('Time Bin from Entry (sec)')
        axes[1].legend()
        for ax in axes.flat:
            ax.set_ylim([-1.5, 1.5])
            ax.grid(axis='y', linestyle=':', alpha=0.6)
        plt.suptitle(fig_title + f' {branch[6:]}', y=0.95)
        # plt.suptitle(f'{self.animal} {self.signal_dir[-23:-7]} {branch[6:]}: non-reward DA', y=0.95)
        plt.tight_layout()
        plt.show()

def bin_nonreward_DA(nonreward_DA_df, bin_size=0.5):
    max_time = nonreward_DA_df['time_in_port'].max()
    bins = np.arange(0, max_time + bin_size, bin_size)
    nonreward_DA_df['time_bin'] = pd.cut(nonreward_DA_df['time_in_port'], bins=bins,
                                                   include_lowest=True)

    def sem(series):
        return series.std() / np.sqrt(len(series))

    binned_results_all = nonreward_DA_df.groupby('time_bin')[['green_right', 'green_left']].agg(
        ['mean', ('sem', sem)]).reset_index()
    binned_results_all['bin_center'] = np.arange(bin_size / 2, max_time + bin_size / 2, bin_size)

    binned_results_split = nonreward_DA_df.groupby(['time_bin', 'phase'])[
        ['green_right', 'green_left']].agg(
        ['mean', ('sem', sem)]).reset_index()
    binned_results_split['bin_center'] = np.repeat(np.arange(bin_size / 2, max_time + bin_size / 2, bin_size), 2)
    return binned_results_all, binned_results_split
