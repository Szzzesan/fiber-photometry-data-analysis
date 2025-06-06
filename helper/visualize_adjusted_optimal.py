import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def visualize_adjusted_optimal(intervals_df, save=0, save_path=None):
    df_plot = intervals_df
    # region find where the block changes
    patch_begin = df_plot.groupby('group').trial.min().to_numpy()
    patch_end = df_plot.groupby('group').trial.max().to_numpy()
    patch_block = df_plot.groupby('group').block.agg(pd.Series.mode)
    # endregion
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(df_plot['trial'], df_plot['optimal_leave'], c='darkgrey', label='Adjusted Optimal', zorder=2)
    ax.bar(df_plot['trial'], df_plot['leave_from_entry'], width=0.4, color='b', label='Leave From Entry', zorder=1)
    ax.scatter(df_plot['trial'].loc[(~df_plot['leave_from_reward'].isna()) & df_plot['trial_time']],
               df_plot['trial_time'].loc[(~df_plot['leave_from_reward'].isna()) & df_plot['trial_time']], c='r',
               marker='.', zorder=2, label='reward')
    for i in range(len(patch_block)):
        if patch_block.iloc[i] == '0.4':
            patch_color = sns.color_palette("Set2")[0]
        else:
            patch_color = sns.color_palette("Set2")[1]
        ax.axvspan(patch_begin[i] - 0.3, patch_end[i] + 0.3, color=patch_color, alpha=0.5, zorder=0)
    ax.legend()
    ax.set_title('Actual Leave vs Adjusted Optimal')
    fig.show()

    if save:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path, 'behavior_vs_adjusted_optimal' + '.png')
        fig.savefig(fig_name)
