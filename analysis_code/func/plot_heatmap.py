import numpy as np
import pandas as pd
import math
import func
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import os


def get_matched_behavior_time_from_condition(pi_events, time0_condition, filterleft_condition, filterright_condition):
    time0_idx = time0_condition.index[time0_condition].to_numpy()
    filterleft_idx = filterleft_condition.index[filterleft_condition].to_numpy()
    filterright_idx = filterright_condition.index[filterright_condition].to_numpy()
    if filterleft_idx.size != filterright_idx.size:
        outer_filter = np.subtract.outer(filterright_idx, filterleft_idx)
        f_r_idx = [
            np.nan if (np.where(outer_filter[:, col] > 0)[0].size == 0) else np.where(outer_filter[:, col] > 0)[0].min()
            for col in list(range(filterleft_idx.size))]
        f_l_idx = [
            np.nan if (np.where(outer_filter[row, :] > 0)[0].size == 0) else np.where(outer_filter[row, :] > 0)[0].max()
            for row in list(range(filterright_idx.size))]
        f_l_set = {x for x in set(f_l_idx) if x == x}
        f_r_set = {x for x in set(f_r_idx) if x == x}
        f_r_idx = list(f_r_set)
        f_l_idx = list(f_l_set)
        filterleft_idx = filterleft_idx[f_l_idx]
        filterright_idx = filterright_idx[f_r_idx]
    if not (filterright_idx > filterleft_idx).all():
        print('There are one or more rows in which the FILTER boundaries are reversed.')
    outer_left = np.subtract.outer(time0_idx, filterleft_idx)
    outer_right = np.subtract.outer(filterright_idx, time0_idx)
    product_for_time0 = np.matmul(outer_left > 0, outer_right > 0)
    product_for_filter = np.matmul(outer_right > 0, outer_left > 0)
    f_idx = np.where(product_for_filter.diagonal() == True)
    zero_idx = np.where(product_for_time0.diagonal() == True)
    time0 = pi_events['time_recording'].iloc[time0_idx[zero_idx]].to_numpy()
    time_filterleft = pi_events['time_recording'].iloc[filterleft_idx[f_idx]].to_numpy()
    time_filterright = pi_events['time_recording'].iloc[filterright_idx[f_idx]].to_numpy()
    trial = pi_events['trial'].iloc[time0_idx[zero_idx]].to_numpy()
    phase = pi_events['phase'].iloc[time0_idx[zero_idx]].to_numpy()
    return time0, time_filterleft, time_filterright, trial, phase


def get_trial_order(pi_events, orderleft_condition, orderright_condition, trial):
    order = np.array(list(range(len(trial))))
    if (orderleft_condition is None) & (orderright_condition is None):
        pass
    elif (orderleft_condition is None) | (orderright_condition is None):
        print('input orderleft or orderright condition must be specified')
    else:
        # re-order the time0, time_filterleft, time_filterright according to how trial is re-ordered
        order_df = pd.DataFrame(columns=['trial', 'edgeleft', 'edgeright', 'distance', 'order'])
        order_df['trial'] = trial
        order_df['edgeleft'] = pi_events['time_recording'].loc[
            orderleft_condition & (pi_events.trial.isin(trial))].to_numpy()
        order_df['edgeright'] = pi_events['time_recording'].loc[
            orderright_condition & (pi_events.trial.isin(trial))].to_numpy()
        if len(order_df.index) != trial.size:
            print('The order array and the trial array should have the same length!')
        if not (order_df['edgeright'] > order_df['edgeleft']).to_numpy().all():
            print('There are one or more rows in which the ORDER boundaries are reversed.')
        order_df['distance'] = order_df['edgeright'] - order_df['edgeleft']
        order_df.sort_values(by='distance', inplace=True)
        order_df['order'] = np.array(list(range(len(order_df.index))))
        order_df.sort_index(inplace=True)
        order = order_df['order'].to_numpy()
    return order


def construct_matrix_for_heatmap(pi_events, dFF0, branch, vmin, vmax, time0_condition, filterleft_condition,
                                 filterright_condition, orderleft_condition=None, orderright_condition=None,
                                 time0_name=None):
    time0, time_filterleft, time_filterright, trial, phase = get_matched_behavior_time_from_condition(pi_events,
                                                                                                      time0_condition,
                                                                                                      filterleft_condition,
                                                                                                      filterright_condition)
    order = get_trial_order(pi_events, orderleft_condition, orderright_condition, trial)
    arrlinds = order.argsort()
    time0 = time0[arrlinds]
    time_filterleft = time_filterleft[arrlinds]
    time_filterright = time_filterright[arrlinds]
    trial = trial[arrlinds]
    phase = phase[arrlinds]

    filter_time_dist_l = time0 - time_filterleft
    filter_time_dist_r = time_filterright - time0
    filter_frame_dist_l = [math.ceil(item / 0.025) for item in filter_time_dist_l]
    filter_frame_dist_r = [math.ceil(item / 0.025) for item in filter_time_dist_r]
    plot_frame_dist_l = math.ceil(abs(vmin) / 0.025)
    plot_frame_dist_r = math.ceil(abs(vmax) / 0.025)
    mat_col_num = plot_frame_dist_l + plot_frame_dist_r
    mat_row_num = trial.size
    np_mat = np.empty((mat_row_num, mat_col_num))
    np_mat[:] = np.nan
    for i in range(mat_row_num):
        time0_idx = func.find_closest_value(dFF0['time_recording'], time0[i])
        offset_l = min(plot_frame_dist_l, filter_frame_dist_l[i])
        np_mat[i, (plot_frame_dist_l - offset_l):plot_frame_dist_l] = dFF0[branch].iloc[
                                                                      (time0_idx - offset_l):time0_idx].to_numpy()
        offset_r = min(plot_frame_dist_r, filter_frame_dist_r[i])
        np_mat[i, plot_frame_dist_l:(plot_frame_dist_l + offset_r)] = dFF0[branch].iloc[time0_idx:(
                time0_idx + offset_r)]

    df_for_heatmap = pd.DataFrame(columns=np.arange(vmin, vmax, 0.025), index=list(range(trial.size)),
                                  data=np_mat * 100)
    df_for_heatmap.branch = branch
    df_for_heatmap.time0 = time0_name
    df_trial_info = pd.DataFrame({'trial': trial, 'phase': phase})
    return df_for_heatmap, df_trial_info


def plot_heatmap_from_matrix(df_for_heatmap, df_trial_info, cbarmin, cbarmax, split_block=0, save=0, save_path=None):
    dFF0_mean_slow = df_for_heatmap[df_trial_info.phase == '0.4'].mean(axis=0).to_numpy()
    dFF0_std_slow = df_for_heatmap[df_trial_info.phase == '0.4'].std(axis=0).to_numpy()
    dFF0_mean_fast = df_for_heatmap[df_trial_info.phase == '0.8'].mean(axis=0).to_numpy()
    dFF0_std_fast = df_for_heatmap[df_trial_info.phase == '0.8'].std(axis=0).to_numpy()
    dFF0_mean = df_for_heatmap.mean(axis=0).to_numpy()
    dFF0_std = df_for_heatmap.std(axis=0).to_numpy()

    # region Specify the parameters of the plot
    figsize = (15, 15)
    xtick_increment = 40
    # the index of the position of xticks
    xticks = np.arange(0, len(df_for_heatmap.columns) - 1, xtick_increment, dtype=int)
    # the content of labels of these xticks
    xticklabels = [round(df_for_heatmap.columns[idx], 1) for idx in xticks]
    colormap = plt.colormaps['crest'].reversed()
    cbar_tick_increment = int((cbarmax - cbarmin) / 4)
    if cbar_tick_increment == 0:
        cbar_tick_increment = 1
    cbar_tick_labels = list(np.arange(cbarmin, cbarmax, cbar_tick_increment))

    block_color = sns.color_palette('Set2')
    # endregion

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax0 = sns.heatmap(df_for_heatmap, cmap=colormap, vmin=cbarmin, vmax=cbarmax,
                      cbar_kws={'ticks': cbar_tick_labels, 'label': 'dFF0 (%)'})
    cbar = ax0.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    ax0.figure.axes[-1].yaxis.label.set_size(20)
    ax0.set(xlabel=None)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels, rotation=0)
    ax0.tick_params(axis='both', which='major', labelsize=15)
    plt.yticks([])
    ax1 = plt.subplot(gs[1])
    l, b, w, h = ax1.get_position().bounds
    ax1.set_position([l, b, 0.8 * w, h])
    if split_block:
        ax1.plot(df_for_heatmap.columns.values, dFF0_mean_slow, c=block_color[0], zorder=10)
        ax1.fill_between(df_for_heatmap.columns.values, dFF0_mean_slow - dFF0_std_slow, dFF0_mean_slow + dFF0_std_slow,
                         color=block_color[0], alpha=0.2, zorder=5)
        ax1.plot(df_for_heatmap.columns.values, dFF0_mean_fast, c=block_color[1], zorder=10)
        ax1.fill_between(df_for_heatmap.columns.values, dFF0_mean_fast - dFF0_std_fast, dFF0_mean_fast + dFF0_std_fast,
                         color=block_color[1], alpha=0.2, zorder=5)
    else:
        ax1.plot(df_for_heatmap.columns.values, dFF0_mean, c='k', zorder=10)
        ax1.fill_between(df_for_heatmap.columns.values, dFF0_mean - dFF0_std, dFF0_mean + dFF0_std,
                         color='darkgrey', alpha=0.2, zorder=5)
    ax1.set_xlim([df_for_heatmap.columns.values.min(), df_for_heatmap.columns.values.max() + 0.025])
    ax1.set_ylim([cbarmin, cbarmax * 0.67])
    ax1.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Time (sec)', fontsize=20)
    plt.ylabel('dF/F0 (%)', fontsize=20)
    plt.suptitle(f'{df_for_heatmap.branch} aligned by {df_for_heatmap.time0}', x=0.43, y=0.93,
                 fontsize=20, weight=10)
    fig.show()
    if save:
        fig_name = f'{df_for_heatmap.branch}_heatmap_aligned_by_{df_for_heatmap.time0}'
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path, fig_name + '.png')
        fig.savefig(fig_name)


def plot_heatmap(pi_events, dFF0, branch, cbarmin, cbarmax, save=0, save_path=None):
    condition_exp_reward = (pi_events['port'] == 1) & (pi_events['value'] == 1) & (pi_events['key'] == 'reward') & (
            pi_events['reward_order_in_trial'] == 1)
    condition_exp_entry = (pi_events['port'] == 1) & (pi_events['value'] == 1) & (pi_events['key'] == 'head') & (
        pi_events.is_valid)
    condition_exp_exit = (pi_events['port'] == 1) & (pi_events['value'] == 0) & (pi_events['key'] == 'head') & (
        pi_events.is_valid)
    #
    # df_for_heatmap_exp_reward, df_exp_r_trial_info = construct_matrix_for_heatmap(pi_events, dFF0, branch=branch,
    #                                                                               vmin=-5, vmax=3,
    #                                                                               time0_condition=condition_exp_reward,
    #                                                                               filterleft_condition=condition_exp_entry,
    #                                                                               filterright_condition=condition_exp_exit,
    #                                                                               orderleft_condition=condition_exp_entry,
    #                                                                               orderright_condition=condition_exp_reward,
    #                                                                               time0_name='ExpReward')
    #
    # plot_one_heatmap(df_for_heatmap_exp_reward, df_exp_r_trial_info, cbarmin=cbarmin, cbarmax=cbarmax, split_block=0,
    #                  save=save, save_path=save_path)

    condition_bg_reward = (pi_events['port'] == 2) & (pi_events['value'] == 1) & (pi_events['key'] == 'reward') & (
            pi_events['reward_order_in_trial'] == 1)
    condition_bg_entry = (pi_events['port'] == 2) & (pi_events['value'] == 1) & (pi_events['key'] == 'trial') & (
        pi_events.is_valid_trial)
    condition_bg_exit = (pi_events['port'] == 2) & (pi_events['value'] == 0) & (pi_events['key'] == 'head') & (
        pi_events.is_valid_trial)
    df_for_heatmap_bg, df_bg_r_trial_info = construct_matrix_for_heatmap(pi_events, dFF0, branch=branch,
                                                                         vmin=-3, vmax=12,
                                                                         time0_condition=condition_bg_entry,
                                                                         filterleft_condition=condition_exp_exit,
                                                                         filterright_condition=condition_bg_exit,
                                                                         orderleft_condition=None,
                                                                         orderright_condition=None,
                                                                         time0_name='BgReward')

    plot_heatmap_from_matrix(df_for_heatmap_bg, df_bg_r_trial_info, cbarmin=cbarmin, cbarmax=cbarmax * 0.4, split_block=1,
                             save=save, save_path=save_path)
