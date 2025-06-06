import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import r2_score


def plot_DA_correlated_with_duration(transient_df, x_col_name, y_col_name, x_label, y_label, xlim, ylim, title,
                                     left_or_right, save_path, save_name, plot=0, save=0):
    if len(transient_df.index) > 1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=transient_df, x=x_col_name, y=y_col_name, hue='block', palette='Set2', legend=None)
        x = transient_df[x_col_name].to_numpy().reshape(-1, 1)
        y = transient_df[y_col_name].to_numpy().reshape(-1, 1)

        if x.size >= 10:
            reg = LinearRegression().fit(x, y)
            estimated_y = reg.predict(x)
            r2 = reg.score(x, y)
            plt.plot(x, estimated_y)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        ax.set_facecolor("white")
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_color('dimgrey')
        plt.grid(False)
        plt.title(title)
        if plot:
            fig.show()
        if save:
            isExist = os.path.exists(save_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(save_path)
                print("A new directory is created!")
            fig_name = os.path.join(save_path, save_name + '_' + left_or_right + '.png')
            fig.savefig(fig_name)

        if x.size >= 10:
            return r2
        else:
            return None


def visualize_trial_by_trial(transient, dFF0, col_name_in_dFF0, session_label, save_path, left_or_right, plot=0,
                             save=0, task='single_reward'):
    df_to_swarm = transient[['n2p', 'p2n', 'r2p', 'p2r', 'r2p2r', 'e2p', 'l2p', 'p2l', 'x2p', 'p2x']].copy().melt(
        var_name='Interval Type', value_name='Duration (sec)')
    # sns.swarmplot(data=df_to_swarm.melt(), x='variable', y='value')
    # plt.show()

    # plt.style.use('ggplot')
    # sns.swarmplot(data=df_to_swarm, x='Interval Type', y='Duration (sec)', size=3, alpha=0.6)
    # # sns.boxplot(data=df_to_swarm, x='Interval Type', y='Duration (sec)', boxprops={"facecolor": 'none'}, showcaps=0, notch=1,
    # #             fliersize=0)
    # sns.violinplot(data=df_to_swarm, x='Interval Type', y='Duration (sec)', color='powderblue', saturation=0, scale='count')
    # plt.show()

    transient_plot = transient[(~transient['r2p'].isna()) & (transient.port == 1)]
    transient_plot['height'] = transient_plot['height'] * 100
    if task == 'multi_reward':
        # region Temporary data cleaning
        transient_plot = transient_plot[transient_plot['num_reward_in_halfsec_before'] < 2]
        transient_plot = transient_plot[transient_plot['num_reward_in_halfsec_after'] < 2]
        transient_plot = transient_plot[transient_plot['is_from_valid_trial']]
        # endregion
    # region all rewards
    r2_R = plot_DA_correlated_with_duration(transient_plot, x_col_name='ts_reward', y_col_name='height',
                                                 x_label='Reward Time since Last Reward (sec)',
                                                 y_label='Peak Height (dF/F0 in %)',
                                                 xlim=[0, 8], ylim=[0, 14],
                                                 title=session_label + ' ' + left_or_right + ' DA ~ IRI',
                                                 left_or_right=left_or_right, save_path=save_path,
                                                 save_name='DA-IRI',
                                                 plot=plot, save=save)
    r2_N = plot_DA_correlated_with_duration(transient_plot, x_col_name='ts_entry', y_col_name='height',
                                                 x_label='Reward Time since Entry (sec)',
                                                 y_label='Peak Height (dF/F0 in %)',
                                                 xlim=[0, 3], ylim=[0, 14],
                                                 title=session_label + ' ' + left_or_right + ' DA ~ Entry-Reward interval',
                                                 left_or_right=left_or_right, save_path=save_path,
                                                 save_name='DA-NRI',
                                                 plot=plot, save=save)
    r2_X = plot_DA_correlated_with_duration(transient_plot, x_col_name='tt_exit', y_col_name='height',
                                                 x_label='Reward Time to Exit (sec)',
                                                 y_label='Peak Height (dF/F0 in %)',
                                                 xlim=[0, 8], ylim=[0, 14],
                                                 title=session_label + ' ' + left_or_right + ' DA ~ Reward-Exit interval',
                                                 left_or_right=left_or_right, save_path=save_path,
                                                 save_name='DA-RXI',
                                                 plot=plot, save=save)
    # endregion

    if task == 'multi_reward':
        # region exclude 1st reward
        df_reward_exc1 = transient_plot[~transient_plot.is_1st_reward]
        r2_exc1_R = plot_DA_correlated_with_duration(df_reward_exc1, x_col_name='ts_reward', y_col_name='height',
                                                     x_label='Reward Time since Last Reward (sec)',
                                                     y_label='Peak Height (dF/F0 in %)',
                                                     xlim=[0, 8.5], ylim=[0, 6],
                                                     title=session_label + ' ' + left_or_right + ' DA ~ IRI (exc 1st reward)',
                                                     left_or_right=left_or_right, save_path=save_path,
                                                     save_name='DA-IRI_exc1',
                                                     plot=plot, save=save)
        r2_exc1_N = plot_DA_correlated_with_duration(df_reward_exc1, x_col_name='ts_entry', y_col_name='height',
                                                     x_label='Reward Time since Entry (sec)',
                                                     y_label='Peak Height (dF/F0 in %)',
                                                     xlim=[0, 8.5], ylim=[0, 6],
                                                     title=session_label + ' ' + left_or_right + ' DA ~ Entry-Reward interval (exc 1st reward)',
                                                     left_or_right=left_or_right, save_path=save_path,
                                                     save_name='DA-NRI_exc1',
                                                     plot=plot, save=save)
        r2_exc1_X = plot_DA_correlated_with_duration(df_reward_exc1, x_col_name='tt_exit', y_col_name='height',
                                                     x_label='Reward Time to Exit (sec)',
                                                     y_label='Peak Height (dF/F0 in %)',
                                                     xlim=[0, 8.5], ylim=[0, 6],
                                                     title=session_label + ' ' + left_or_right + ' DA ~ Reward-Exit interval (exc 1st reward)',
                                                     left_or_right=left_or_right, save_path=save_path,
                                                     save_name='DA-RXI_exc1',
                                                     plot=plot, save=save)
        # endregion

        # region Only looking at first reward of each trial
        df_reward_1st = transient_plot[transient_plot.is_1st_reward]
        r2_1streward_R = plot_DA_correlated_with_duration(df_reward_1st, x_col_name='ts_reward', y_col_name='height',
                                                          x_label='Reward Time since Last Reward (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ IRI (1st reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-IRI_1streward',
                                                          plot=plot, save=save)
        r2_1streward_N = plot_DA_correlated_with_duration(df_reward_1st, x_col_name='ts_entry', y_col_name='height',
                                                          x_label='Reward Time since Entry (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ Entry-Reward interval (1st reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-NRI_1streward',
                                                          plot=plot, save=save)
        r2_1streward_X = plot_DA_correlated_with_duration(df_reward_1st, x_col_name='tt_exit', y_col_name='height',
                                                          x_label='Reward Time to Exit (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ Reward-Exit interval (1st reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-RXI_1streward',
                                                          plot=plot, save=save)
        # endregion

        # region Only looking at the end reward of each trial
        df_reward_end = transient_plot[transient_plot.is_end_reward]
        r2_endreward_R = plot_DA_correlated_with_duration(df_reward_end, x_col_name='ts_reward', y_col_name='height',
                                                          x_label='Reward Time since Last Reward (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ IRI (end reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-IRI_endreward',
                                                          plot=plot, save=save)
        r2_endreward_N = plot_DA_correlated_with_duration(df_reward_end, x_col_name='ts_entry', y_col_name='height',
                                                          x_label='Reward Time since Entry (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ Entry-Reward interval (end reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-NRI_endreward',
                                                          plot=plot, save=save)
        r2_endreward_X = plot_DA_correlated_with_duration(df_reward_end, x_col_name='tt_exit', y_col_name='height',
                                                          x_label='Reward Time to Exit (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ Reward-Exit interval (end reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-RXI_endreward',
                                                          plot=plot, save=save)
        # endregion

        # region middle rewards
        df_reward_middle = transient_plot[(~transient_plot.is_1st_reward) & (~transient_plot.is_end_reward)]
        r2_midreward_R = plot_DA_correlated_with_duration(df_reward_middle, x_col_name='ts_reward', y_col_name='height',
                                                          x_label='Reward Time since Last Reward (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ IRI (mid reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-IRI_midreward',
                                                          plot=plot, save=save)
        r2_midreward_N = plot_DA_correlated_with_duration(df_reward_middle, x_col_name='ts_entry', y_col_name='height',
                                                          x_label='Reward Time since Entry (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ Entry-Reward interval (mid reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-NRI_midreward',
                                                          plot=plot, save=save)
        r2_midreward_X = plot_DA_correlated_with_duration(df_reward_middle, x_col_name='tt_exit', y_col_name='height',
                                                          x_label='Reward Time to Exit (sec)',
                                                          y_label='Peak Height (dF/F0 in %)',
                                                          xlim=[0, 8.5], ylim=[0, 6],
                                                          title=session_label + ' ' + left_or_right + ' DA ~ Reward-Exit interval (mid reward)',
                                                          left_or_right=left_or_right, save_path=save_path,
                                                          save_name='DA-RXI_midreward',
                                                          plot=plot, save=save)
        # endregion

        return r2_midreward_R, r2_midreward_N, r2_midreward_X, \
            r2_1streward_R, r2_1streward_N, r2_1streward_X, \
            r2_endreward_R, r2_endreward_N, r2_endreward_X, \
            r2_exc1_R, r2_exc1_N, r2_exc1_X, \
            r2_R, r2_N, r2_X

    elif task == 'single_reward':
        return r2_R, r2_N, r2_X

    # # region 3D visualization between time since entry, time since last reward, and da response
    # fig, ax=plt.subplots()
    # sns.scatterplot(data=transient_plot, x='ts_reward', y='ts_entry', hue='height', style='is_1st_reward')
    # plt.xlabel('Time since Last Reward (sec)')
    # plt.ylabel('Time since Entry (sec)')
    # plt.xlim([0, 10])
    # plt.ylim([0, 15])
    #
    # ax.set_facecolor("white")
    # ax.spines['left'].set_color('dimgrey')
    # ax.spines['bottom'].set_color('dimgrey')
    # plt.grid(False)
    # plt.show()
    # endregion

    # region plot average traces split by time since last reward
    # reward_preceded_transient = transient[~transient['r2p'].isna()].sort_values('ts_reward', ignore_index=True)
    # transient_df_collection = {}
    # for i in range(len(reward_preceded_transient.index)):
    #     rt = reward_preceded_transient.reward_time[i]
    #     is_peri_reward_in_dFF0 = (dFF0['time_recording'] > rt - 1) & (dFF0['time_recording'] < rt + 5)
    #     time_peri_reward = dFF0['time_recording'].loc[is_peri_reward_in_dFF0] - rt
    #
    #     da_peri_reward = dFF0[col_name_in_dFF0].loc[is_peri_reward_in_dFF0]
    #     transient_df_collection[i] = pd.DataFrame(data={'time_post_reward': time_peri_reward})
    #     bins = np.arange(-1, 5 + 0.033, 0.033)
    #     transient_df_collection[i]['time_bin'] = pd.cut(transient_df_collection[i].time_post_reward, bins, right=False)
    #     transient_df_collection[i][f'dff0_{i}'] = da_peri_reward
    #     transient_df_collection[i] = transient_df_collection[i].groupby('time_bin', as_index=False).mean()
    #     transient_df_collection[i]['time'] = bins[:-1]
    #     transient_df_collection[i].drop(columns=['time_bin', 'time_post_reward'], inplace=True)
    #
    # df_all = transient_df_collection[0]
    # for d in range(1, len(transient_df_collection)):
    #     df = transient_df_collection[d]
    #     df_all = df_all.merge(df, on='time')
    # first_col = df_all.pop('time')
    # df_all.insert(0, 'time', first_col)
    # transient_num = len(df_all.columns)-1
    # split_1 = int(transient_num / 3) + 1
    # split_2 = transient_num - int(transient_num /3) + 1
    # mean_val_0 = df_all.iloc[:, 1:split_1].mean(axis=1)
    # mean_val_1 = df_all.iloc[:, split_1:split_2].mean(axis=1)
    # mean_val_2 = df_all.iloc[:, split_2:].mean(axis=1)
    # fig, ax = plt.subplots()
    # ax.plot(df_all['time'], mean_val_0, color='black', label='short IRI')
    # ax.plot(df_all['time'], mean_val_1, color='dimgrey', label='mid IRI')
    # ax.plot(df_all['time'], mean_val_2, color='darkgrey', label='long IRI')
    # ax.set_facecolor("white")
    # ax.spines['left'].set_color('dimgrey')
    # ax.spines['bottom'].set_color('dimgrey')
    # ax.legend()
    # plt.grid(False)
    # plt.xlabel('Time post reward (sec)')
    # plt.ylabel('dF/F0')
    # plt.ylim([-0.01, 0.04])
    # plt.show()
    # endregion
