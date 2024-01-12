import pandas as pd
import numpy as np
import func
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


def find_inner_interval(interval1, interval2):
    if np.isnan(list(interval1)).any() or np.isnan(list(interval2)).any():
        return np.nan
    else:
        l = max(interval1[0], interval2[0])
        r = min(interval1[1], interval2[1])
        if l < r:
            return [l, r]
        else:
            return np.nan


def order_of_trial(df, trial):
    s = df.trial
    s = s.drop_duplicates()
    sorted = s.sort_values()
    sorted = sorted.reset_index()
    o = sorted.index[sorted.trial == trial].to_numpy()[0]
    return o


def sort_heatmap_by_filterborder(matrix_to_sort, before_or_after='before'):
    if before_or_after == 'before':
        df_part = matrix_to_sort.iloc[:, matrix_to_sort.columns < 0]
    else:
        df_part = matrix_to_sort.iloc[:, matrix_to_sort.columns > 0]
    num_nan_in_row = np.zeros(len(df_part.index))
    for idx in range(len(num_nan_in_row)):
        num_nan_in_row[idx] = df_part.iloc[idx].isnull().sum()
    trial_order = np.argsort(np.argsort(num_nan_in_row))
    matrix_to_sort['order'] = trial_order + 1
    matrix_to_sort.sort_values(by='order', inplace=True, ascending=False)
    matrix_to_sort.index = matrix_to_sort.order.max() + 1 - matrix_to_sort.order
    matrix_to_sort.drop(columns='order', inplace=True)
    return matrix_to_sort


def sort_scatter_by_filterborder(scatter_df, before_or_after='before'):
    if before_or_after == 'before':
        is_right_direction = scatter_df.time < 0
    else:
        is_right_direction = scatter_df.time > 0
    all_trials = np.unique(scatter_df.trial.to_numpy())
    trial_num = len(all_trials)
    diff = np.zeros(trial_num)
    for i in range(len(diff)):
        if before_or_after == 'before':
            diff[i] = scatter_df.time[is_right_direction & (scatter_df.trial == all_trials[i])].max()
        else:
            diff[i] = scatter_df.time[is_right_direction & (scatter_df.trial == all_trials[i])].min()
    trial_order = np.argsort(np.argsort(-diff))
    trial_order_duplicated = np.zeros(len(scatter_df.index))
    for i in range(len(trial_order_duplicated)):
        trial_order_duplicated[i] = trial_order[all_trials == scatter_df.trial[i]]
    scatter_df.trial = trial_order_duplicated
    return scatter_df


def get_signal_per_bin(signal_data, branch, aligner_event_time, plot_interval, filter_interval=None, bin_size=0.033):
    col_name = 'green_' + branch
    df = pd.DataFrame({'time_recording': signal_data.time_recording, 'dFF0': signal_data[col_name]})
    df = df[
        (df.time_recording > aligner_event_time + plot_interval[0]) & (
                df.time_recording < aligner_event_time + plot_interval[1])]
    df.reset_index(drop=True, inplace=True)
    df['time_aligned'] = df.time_recording - aligner_event_time
    bins = np.arange(plot_interval[0], plot_interval[1] + bin_size, bin_size)
    df['time_bin'] = pd.cut(df.time_aligned, bins, right=False)
    df = df.drop(['time_aligned', 'time_recording'], axis=1)
    grouped = df.groupby(['time_bin']).mean()
    time_label = bins[:-1]
    if filter_interval is None:
        inner_interval = plot_interval
    else:
        if (filter_interval[0] < aligner_event_time < filter_interval[1]):
            inner_interval = find_inner_interval(plot_interval, filter_interval - aligner_event_time)
        else:
            inner_interval = np.nan
    if ~np.isnan(inner_interval).any():
        binned_dFF0 = pd.DataFrame({'time': time_label, 'dFF0': grouped.dFF0.values})
        indices_in_range = np.where(
            (binned_dFF0.time.to_numpy() <= inner_interval[1]) & (binned_dFF0.time.to_numpy() >= inner_interval[0]))
        # todo: There must be sth. wrong with the length of filter_intervals. Fix it
        binned_dFF0.drop(range(indices_in_range[0][0] + 1), inplace=True)
        binned_dFF0.drop(range(indices_in_range[0][-1] + 1, max(binned_dFF0.index) + 1), inplace=True)
    else:
        binned_dFF0 = pd.DataFrame(columns=['time', 'dFF0'])
    binned_dFF0['trial'] = ''
    return binned_dFF0


def get_heatmap_matrix(dFF0, pi_events, condition, branch, plot_interval, filter_intervals,
                       bin_size=1 / 30, sort='False', sort_direction='before'):
    event_rows = pi_events[condition].reset_index(drop=True)
    dFF0_for_heatmap = pd.DataFrame(columns=['time', 'dFF0', 'trial'])
    for i in range(len(event_rows)):
        event_time = event_rows.time_recording[i]
        in_range_bool = [filter_intervals[j][0] < event_time < filter_intervals[j][1] for j in
                         range(len(filter_intervals))]
        interval_idx = [i for i, x in enumerate(in_range_bool) if x]
        if len(interval_idx) > 0:
            binned_dFF0 = get_signal_per_bin(dFF0, branch, event_time, plot_interval=plot_interval,
                                             filter_interval=filter_intervals[interval_idx][0], bin_size=bin_size)
            if len(binned_dFF0.index) > 0:
                binned_dFF0['trial'] = event_rows.trial[i]
            dFF0_for_heatmap = pd.concat([dFF0_for_heatmap, binned_dFF0], axis=0)

    dFF0_for_heatmap.dFF0 = dFF0_for_heatmap.dFF0.astype(float)
    dFF0_for_heatmap = dFF0_for_heatmap.pivot_table(index='trial', columns='time', values='dFF0')
    if sort:
        dFF0_for_heatmap = sort_heatmap_by_filterborder(dFF0_for_heatmap, sort_direction)
    return dFF0_for_heatmap


def get_marker_each_row(pi_events, marker_event, aligner_event_time, interval):
    within_search_range = (pi_events.time_recording > aligner_event_time + interval[0]) & (
            pi_events.time_recording < aligner_event_time + interval[1])
    if marker_event == 'entry':
        is_marker_event = (pi_events.key == 'head') & (pi_events.value == 1)
    elif marker_event == 'exit':
        is_marker_event = (pi_events.key == 'head') & (pi_events.value == 0)
    else:
        is_marker_event = (pi_events.key == marker_event) & (pi_events.value == 1)
    df = pd.DataFrame(columns=['marker_time', 'time', 'trial', 'event'])
    df['marker_time'] = pi_events.time_recording[within_search_range & is_marker_event]
    df['trial'] = pi_events.trial[within_search_range & is_marker_event]
    df['time'] = df['marker_time'] - aligner_event_time
    df['event'] = marker_event
    return df


def get_marker_for_scatters(pi_events, pi_trials, aligner_event, sequence, plot_interval, filter_intervals, bin_width,
                            sort='False', sort_direction='before'):
    aligner_event_time_series = np.empty(len(pi_trials.index))
    for idx in pi_trials.index:
        if (len(pi_trials[aligner_event].iloc[idx]) > sequence) & (len(pi_trials[aligner_event].iloc[idx]) > 0):
            aligner_event_time_series[idx] = pi_trials[aligner_event].iloc[idx][sequence]
        else:
            aligner_event_time_series[idx] = np.nan
    df_for_scatter = pd.DataFrame(columns=['marker_time', 'time', 'trial', 'event'])
    for marker_event in ['entry', 'exit', 'reward', 'lick']:
        for i in range(len(aligner_event_time_series)):
            if ~np.isnan(list(filter_intervals[i, :])).any():
                inner_interval = find_inner_interval(plot_interval,
                                                     filter_intervals[i, :] - aligner_event_time_series[i])
                if ~np.isnan(inner_interval).any():
                    inner_interval = [inner_interval[0] - bin_width, inner_interval[1] + bin_width]
                    df = get_marker_each_row(pi_events, marker_event, aligner_event_time_series[i],
                                             interval=inner_interval)
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
            df_for_scatter = pd.concat([df_for_scatter, df], axis=0)
    df_for_scatter.drop(['marker_time'], axis=1, inplace=True)
    df_for_scatter.reset_index(drop=True, inplace=True)
    if sort:
        df_for_scatter = sort_scatter_by_filterborder(df_for_scatter, before_or_after=sort_direction)
    df_for_scatter['y_position'] = ''
    # region fix the positions of the markers on the y axis
    for idx in range(len(df_for_scatter.index)):
        df_for_scatter.y_position[idx] = order_of_trial(df_for_scatter, df_for_scatter.trial[idx])
    # endregion
    return df_for_scatter


def sensor_raster_plot(dFF0, pi_events, pi_trials, condition,
                       branch='right',
                       port='bg',
                       aligned_by='rewards',
                       sequence=0,
                       filter_intervals=None,
                       bin_size=1 / 30,
                       plot_interval=[-1, 5],
                       fig_size=(10, 10),
                       plot_markers='True',
                       save=False, save_path=None,
                       sort='False', sort_direction='before'):
    dFF0_for_heatmap = get_heatmap_matrix(dFF0, pi_events, condition, branch,
                                          plot_interval=plot_interval, filter_intervals=filter_intervals,
                                          bin_size=1 / 30, sort=sort, sort_direction=sort_direction)
    dFF0_mean = dFF0_for_heatmap.mean(axis=0)
    df_for_scatters = get_marker_for_scatters(pi_events, pi_trials, aligner_event=port + '_' + aligned_by,
                                              sequence=sequence, plot_interval=plot_interval,
                                              filter_intervals=filter_intervals, bin_width=bin_size, sort=sort,
                                              sort_direction=sort_direction)
    lick_df = df_for_scatters[df_for_scatters.event == 'lick']
    df_for_scatters.drop(df_for_scatters.index[df_for_scatters.event == 'lick'], inplace=True)

    # region Specify the parameters of the plot
    xtick_increment = int(1 / bin_size)
    # the index of the position of xticks
    xticks = np.arange(0, len(dFF0_for_heatmap.columns) - 1, xtick_increment, dtype=int)
    # the content of labels of these xticks
    xticklabels = [round(dFF0_for_heatmap.columns[idx], 1) for idx in xticks]
    colormap = plt.cm.get_cmap('crest').reversed()
    # endregion

    # region Plot the heatmap and the histogram
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax0 = sns.heatmap(dFF0_for_heatmap, cmap=colormap, vmin=-0.03, vmax=0.09,
                      cbar_kws={'ticks': [-0.03, 0, 0.03, 0.06, 0.09]})
    if plot_markers:
        sns.scatterplot(x=(df_for_scatters.time - plot_interval[0]) / bin_size,
                        y=df_for_scatters.y_position + 0.5, hue=df_for_scatters.event, style=df_for_scatters.event,
                        edgecolor='none', palette='Set2')
    ax0.legend(loc='lower right', bbox_to_anchor=(1.12, 1.01))
    ax0.set(xlabel=None)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels, rotation=0)
    ax1 = plt.subplot(gs[1])
    plt.hist(lick_df.time, color='grey', alpha=0.5,
             bins=len(dFF0_for_heatmap.index) * (plot_interval[1] - plot_interval[0]), zorder=0)
    plt.ylim([0, 20])
    l, b, w, h = ax1.get_position().bounds
    ax1.set_position([l, b, 0.8 * w, h])
    ax1.set_ylabel('lick histogram')
    ax2 = ax1.twinx()
    plt.plot(dFF0_mean, zorder=1)
    ax2.set_position([l, b, 0.8 * w, h])
    plt.xlim([min(dFF0_mean.index), max(dFF0_mean.index)])
    plt.ylim([-0.03, 0.06])
    plt.xlabel('Time (sec)')
    plt.ylabel('dF/F0')
    plt.suptitle(branch + ' branch aligned by ' + port + ' ' + aligned_by + '#' + str(sequence + 1), x=0.43, y=0.93,
                 fontsize=20, weight=10)
    plt.show()

    if save:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path,
                                branch + '_' + port + '_' + aligned_by + '_No' + str(sequence + 1) + str(
                                    bool(sort)) + '.png')
        fig.savefig(fig_name)

    # endregion
