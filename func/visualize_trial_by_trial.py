import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score


def visualize_trial_by_trial(transient, dFF0, col_name_in_dFF0):
    df_to_swarm = transient[['n2p', 'p2n', 'r2p', 'p2r', 'r2p2r', 'e2p', 'l2p', 'p2l', 'x2p', 'p2x']].copy().melt(var_name='Interval Type', value_name='Duration (sec)')
    # sns.swarmplot(data=df_to_swarm.melt(), x='variable', y='value')
    # plt.show()

    plt.style.use('ggplot')
    sns.swarmplot(data=df_to_swarm, x='Interval Type', y='Duration (sec)', size=3, alpha=0.6)
    # sns.boxplot(data=df_to_swarm, x='Interval Type', y='Duration (sec)', boxprops={"facecolor": 'none'}, showcaps=0, notch=1,
    #             fliersize=0)
    sns.violinplot(data=df_to_swarm, x='Interval Type', y='Duration (sec)', color='powderblue', saturation=0, scale='count')
    plt.show()

    # plt.style.use('ggplot')
    # transient_plot = transient[(~transient['r2p'].isna()) & (transient.port == 1)]
    # sns.scatterplot(data=transient_plot, x='ts_reward', y='height', palette='Set2', style='is_1st_reward')
    # plt.xlabel('Time since Last Reward (sec)')
    # plt.ylabel('Peak Height')
    # plt.xlim([0, 10])
    # plt.ylim([0.008, 0.1])
    # plt.show()
    #
    # plt.style.use('ggplot')
    # sns.scatterplot(data=transient_plot, x='ts_entry', y='height', palette='Set2', style='is_1st_reward')
    # plt.xlabel('Time since Entry (sec)')
    # plt.ylabel('Peak Height')
    # plt.xlim([0, 10])
    # plt.ylim([0.008, 0.1])
    # plt.show()
    #
    # plt.style.use('ggplot')
    # sns.scatterplot(data=transient_plot, x='ts_entry_or_reward', y='height', palette='Set2', style='is_1st_reward')
    # plt.xlabel('Time since Entry/Reward (sec)')
    # plt.ylabel('Peak Height')
    # plt.xlim([0, 10])
    # plt.ylim([0.008, 0.1])
    # plt.show()

    transient_plot = transient[(~transient['r2p'].isna()) & (transient.port == 1)]
    transient_plot = transient_plot[transient_plot.is_1st_reward]
    # region Temporary data clipping - only applicable for one session
    condition1 = (transient_plot.trial == 7) & (transient_plot.is_1st_reward)
    condition2 = (transient_plot.trial == 20) & (transient_plot.is_1st_reward)
    transient_plot = transient_plot[(~condition1) & (~condition2)]
    # endregion
    fig, ax = plt.subplots()
    sns.scatterplot(data=transient_plot, x='ts_reward', y='height', hue='block', palette='Set2', marker='x', legend=None)
    x_1 = transient_plot['ts_reward'].to_numpy().reshape(-1,1)
    y = transient_plot['height'].to_numpy().reshape(-1,1)
    huber_r = HuberRegressor(alpha=0.0001, epsilon=1).fit(x_1, y)
    estimated_y = huber_r.predict(x_1)
    r2_reward = r2_score(y, estimated_y)
    plt.plot(x_1, estimated_y)
    ax.set_xlabel('Time since Last Reward (sec)')
    ax.set_ylabel('Peak Height')
    ax.set_xlim([0, 10])
    ax.set_ylim([0.008, 0.1])
    ax.set_facecolor("white")
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    plt.grid(False)
    fig.show()

    fig, ax = plt.subplots()
    sns.scatterplot(data=transient_plot, x='ts_entry', y='height', hue='block', palette='Set2', marker='x', legend=None)
    x_2 = transient_plot['ts_entry'].to_numpy().reshape(-1,1)
    y = transient_plot['height'].to_numpy().reshape(-1,1)
    huber_n = HuberRegressor(alpha=0.0001, epsilon=1).fit(x_2, y)
    estimated_y = huber_n.predict(x_2)
    r2_entry = r2_score(y, estimated_y)
    plt.plot(x_2, estimated_y)
    plt.xlabel('Time since Entry (sec)')
    plt.ylabel('Peak Height')
    plt.xlim([0, 15])
    plt.ylim([0.008, 0.1])
    ax.set_facecolor("white")
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    plt.grid(False)
    plt.show()
    print('Hello')
    return r2_reward
    # fig, ax = plt.subplots()
    # sns.scatterplot(data=transient_plot, x='ts_entry_or_reward', y='height', hue='block', palette='Set2', style='is_1st_reward', legend=None)
    # plt.xlabel('Time since Entry/Reward (sec)')
    # plt.ylabel('Peak Height')
    # plt.xlim([0, 10])
    # plt.ylim([0.008, 0.1])
    # ax.set_facecolor("white")
    # ax.spines['left'].set_color('dimgrey')
    # ax.spines['bottom'].set_color('dimgrey')
    # plt.grid(False)
    # # plt.legend(bbox_to_anchor=(1.01, 1.01), loc='lower left')
    # plt.show()
    #
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



