import os
import func
from OneSession import OneSession
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
from L import L
import pandas as pd
from scipy import stats
from collections import namedtuple
import pingouin as pg  # packages for anova according to Google


def visualize_transient_consistency(session_obj_list, save=0, save_path=None):
    transient_occur_xsession_l = np.empty(len(session_obj_list))
    transient_occur_xsession_r = np.empty(len(session_obj_list))
    transient_midmag_xsession_l = np.empty(len(session_obj_list))
    transient_midmag_xsession_r = np.empty(len(session_obj_list))
    for i in range(len(session_obj_list)):
        transient_occur_xsession_l[i] = session_obj_list[i].transient_occur_l
        transient_occur_xsession_r[i] = session_obj_list[i].transient_occur_r
        if session_obj_list[i].transient_midmag_l is not None:
            transient_midmag_xsession_l[i] = session_obj_list[i].transient_midmag_l * 100
        if session_obj_list[i].transient_midmag_r is not None:
            transient_midmag_xsession_r[i] = session_obj_list[i].transient_midmag_r * 100
    session_selected_l = np.where(
        (transient_midmag_xsession_l >= np.nanpercentile(transient_midmag_xsession_l, 25) - 0.5) & (
                transient_midmag_xsession_l <= np.nanpercentile(transient_midmag_xsession_l,
                                                                75) + 0.8))
    session_selected_r = np.where(
        (transient_midmag_xsession_r >= np.nanpercentile(transient_midmag_xsession_r, 25) - 0.5) & (
                transient_midmag_xsession_r <= np.nanpercentile(transient_midmag_xsession_r,
                                                                75) + 0.8))
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(hspace=0)
    if sum(transient_occur_xsession_l == None) != len(transient_occur_xsession_l):
        ax[0].plot(transient_occur_xsession_l, label='left hemisphere')
        ax[1].plot(transient_midmag_xsession_l, label='left hemisphere')
    if sum(transient_occur_xsession_r == None) != len(transient_occur_xsession_r):
        ax[0].plot(transient_occur_xsession_r, label='right hemisphere')
        ax[1].plot(transient_midmag_xsession_r, label='right hemisphere')
    ax[1].axhspan(np.nanpercentile(transient_midmag_xsession_l, 25) - 0.5,
                  np.nanpercentile(transient_midmag_xsession_l, 75) + 0.8, color='b', alpha=0.2, zorder=-10)
    ax[1].axhspan(np.nanpercentile(transient_midmag_xsession_r, 25) - 0.5,
                  np.nanpercentile(transient_midmag_xsession_r, 75) + 0.8, color='orange', alpha=0.2, zorder=-10)
    ax[0].scatter(session_selected_l, transient_occur_xsession_l[session_selected_l], color='b', marker='*', zorder=10)
    ax[0].scatter(session_selected_r, transient_occur_xsession_r[session_selected_r], color='orange', marker='*',
                  zorder=10)
    ax[1].scatter(session_selected_l, transient_midmag_xsession_l[session_selected_l], color='b', marker='*', zorder=10)
    ax[1].scatter(session_selected_r, transient_midmag_xsession_r[session_selected_r], color='orange', marker='*',
                  zorder=10)
    ax[0].legend()
    ax[0].set_ylabel('Transient Occurance Rate (/min)', fontsize=15)
    ax[0].set_ylim([0, 25])
    ax[1].set_ylabel('Median Transient Magnitude (%)', fontsize=15)
    ax[1].set_ylim([0, 15])
    ax[1].set_xlabel('Session', fontsize=20)
    fig.suptitle(f'{session_obj_list.animal}: Transient Consistency Analysis', fontsize=30)
    fig.show()
    if save:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path, 'TransientConsistencyXSessions' + '.png')
        fig.savefig(fig_name)
    return session_selected_l, session_selected_r


def visualize_NRI_vs_amplitude_families(ani_summary, variable='time_in_port', block_split=False, normalized=False,
                                        save=False,
                                        save_path=None):
    # variable has two options: 'time_in_port' and 'IRI_prior'
    num_session = len([x for x in ani_summary.session_obj_list if x is not None])
    median_delay_arr = np.zeros((5, num_session))
    amp_arr_low = np.zeros((5, num_session))
    amp_arr_high = np.zeros((5, num_session))
    average_amplitude_arr = np.zeros((5, num_session))

    if block_split:
        # cmap = plt.get_cmap('plasma')
        # new_cmap = cmap.resampled(len(session_obj_list))
        block_palette = sns.color_palette('Set2')
        low_palette = sns.light_palette(block_palette[0], 1.5 * len(ani_summary.session_obj_list))
        high_palette = sns.light_palette(block_palette[1], 1.5 * len(ani_summary.session_obj_list))

    for branch in {'ipsi', 'contra'}:
        fig = plt.figure()
        idx_arr = 0
        for i in range(len(ani_summary.session_obj_list)):
            if ani_summary.session_obj_list[i] == None:
                continue
            else:
                if branch == 'contra':
                    if variable == 'time_in_port':
                        interval_amp_df = ani_summary.session_obj_list[i].NRI_amplitude_l
                    elif variable == 'IRI_prior':
                        interval_amp_df = ani_summary.session_obj_list[i].IRI_amplitude_l
                else:
                    if variable == 'time_in_port':
                        interval_amp_df = ani_summary.session_obj_list[i].NRI_amplitude_r
                    elif variable == 'IRI_prior':
                        interval_amp_df = ani_summary.session_obj_list[i].IRI_amplitude_r
                if interval_amp_df is None:
                    continue
                else:
                    median_delay_arr[:, idx_arr] = interval_amp_df[
                        'median_interval'].to_numpy()  # 'delay' and 'interval' mean the same thing here. Just don't get confused

                    # store the average amplitude of each quintile from each session into a big 2d matrix, so that we can average them across session later
                    amp_arr_low[:, idx_arr] = interval_amp_df['amp_low'].to_numpy()
                    amp_arr_high[:, idx_arr] = interval_amp_df['amp_high'].to_numpy()
                    average_amplitude_arr[:, idx_arr] = interval_amp_df['amp_all'].to_numpy()

                    idx_arr += 1

                    if block_split & normalized:
                        weight = np.round(0.8 - (i + 1) / (len(ani_summary.session_obj_list) + 1) * 0.3, decimals=2)
                        normalized_interval_amplitude_l = interval_amp_df['amp_low'] / \
                                                          interval_amp_df['amp_high']
                        plt.plot(interval_amp_df['median_interval'],
                                 normalized_interval_amplitude_l, c=str(weight), marker='o')
                    elif block_split & (not normalized):
                        plt.plot(interval_amp_df['median_interval'],
                                 interval_amp_df['amp_low'], c=low_palette[i], marker='o')
                        plt.plot(interval_amp_df['median_interval'],
                                 interval_amp_df['amp_high'], c=high_palette[i], marker='o')
                    else:
                        weight = np.round(0.8 - (i + 1) / (len(ani_summary.session_obj_list) + 1) * 0.3, decimals=2)
                        plt.plot(interval_amp_df['median_interval'],
                                 interval_amp_df['amp_all'], c=str(weight), marker='o')

        if block_split & normalized:
            normalized_amp_arr_low = amp_arr_low / amp_arr_high
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(normalized_amp_arr_low, axis=1), c=block_palette[0],
                     marker='o', markersize=10, zorder=10)
            plt.plot(np.median(median_delay_arr, axis=1), np.ones(5), c=block_palette[1], marker='o',
                     markersize=10, zorder=10)
        elif block_split & (not normalized):
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(amp_arr_low, axis=1), c=block_palette[0], marker='o',
                     markersize=10, zorder=10)
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(amp_arr_high, axis=1), c=block_palette[1], marker='o',
                     markersize=10, zorder=10)
        else:
            plt.plot(np.median(median_delay_arr, axis=1), np.mean(average_amplitude_arr, axis=1),
                     marker='o', markersize=10, zorder=10)

        if variable == 'time_in_port':
            plt.title(f'{ani_summary.animal}: {branch} DA Amplitude vs Time since Entry', fontsize=20)
            plt.xlabel('Median Delay off Entry (sec)', fontsize=15)
        elif variable == 'IRI_prior':
            plt.title(f'{ani_summary.animal}: {branch} DA Amplitude vs Time since Last Reward', fontsize=20)
            plt.xlabel('Median IRI (sec)', fontsize=15)
        plt.ylabel('Z-score Amplitude', fontsize=15)
        fig.show()

        # 'ani_sum' stands for animal summary.
        # This dataframe stores the median and quartiles of NRIs and the mean and SEM of DA amplitudes
        # from all sessions of one animal's one hemisphere
        if branch == 'contra':
            if variable == 'time_in_port':
                ani_summary.NRI_amp_contra['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.NRI_amp_contra['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.NRI_amp_contra['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.NRI_amp_contra['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_contra['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_contra['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.NRI_amp_contra['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.NRI_amp_contra['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.NRI_amp_contra['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)
            elif variable == 'IRI_prior':
                ani_summary.IRI_amp_contra['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.IRI_amp_contra['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.IRI_amp_contra['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.IRI_amp_contra['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_contra['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_contra['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.IRI_amp_contra['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.IRI_amp_contra['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.IRI_amp_contra['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)
        elif branch == 'ipsi':
            if variable == 'time_in_port':
                ani_summary.NRI_amp_ipsi['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.NRI_amp_ipsi['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.NRI_amp_ipsi['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.NRI_amp_ipsi['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_ipsi['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.NRI_amp_ipsi['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.NRI_amp_ipsi['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.NRI_amp_ipsi['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.NRI_amp_ipsi['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)
            elif variable == 'IRI_prior':
                ani_summary.IRI_amp_ipsi['median_interval'] = np.median(median_delay_arr, axis=1)
                ani_summary.IRI_amp_ipsi['quart1_interval'] = np.quantile(median_delay_arr, 0.25, axis=1)
                ani_summary.IRI_amp_ipsi['quart3_interval'] = np.quantile(median_delay_arr, 0.75, axis=1)
                ani_summary.IRI_amp_ipsi['mean_amp'] = np.mean(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_ipsi['SEM_amp'] = stats.sem(average_amplitude_arr, axis=1)
                ani_summary.IRI_amp_ipsi['mean_amp_low'] = np.mean(amp_arr_low, axis=1)
                ani_summary.IRI_amp_ipsi['SEM_amp_low'] = stats.sem(amp_arr_low, axis=1)
                ani_summary.IRI_amp_ipsi['mean_amp_high'] = np.mean(amp_arr_high, axis=1)
                ani_summary.IRI_amp_ipsi['SEM_amp_high'] = stats.sem(amp_arr_high, axis=1)


def multi_session_analysis(animal_str, session_list, include_branch='both'):
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
    raw_dir = os.path.join(animal_dir, 'raw_data')
    FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    xsession_figure_export_dir = os.path.join(animal_dir, 'figures')
    # Declaring namedtuple()
    df1 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df2 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df3 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    df4 = pd.DataFrame(
        columns=['median_interval', 'quart1_interval', 'quart3_interval', 'mean_amp', 'SEM_amp', 'mean_amp_low',
                 'SEM_amp_low', 'mean_amp_high', 'SEM_amp_high'])
    OneAniAllSes = namedtuple('OneAniAllSes',
                              ['animal', 'session_obj_list', 'NRI_amp_ipsi', 'NRI_amp_contra', 'IRI_amp_ipsi',
                               'IRI_amp_contra'])

    # check if the neural data files, the behavior data files, and the sync data files are of the same numbers
    if (len(FP_file_list) == len(behav_file_list)) & (len(behav_file_list) == len(TTL_file_list)):

        # If so, make a namedtuple,
        # with each including the animal name,
        # the list of objects each being one session,
        # and the summary NRI_amplitude dataframe

        # adding values to a named tuple
        ani_summary = OneAniAllSes(animal=animal_str, session_obj_list=[None] * len(FP_file_list), NRI_amp_ipsi=df1,
                                   NRI_amp_contra=df2, IRI_amp_ipsi=df3, IRI_amp_contra=df4)
    else:
        print("Error: the numbers of different data files should be equal!!")
    for i in session_list:
        try:
            ani_summary.session_obj_list[i] = OneSession(animal_str, i, include_branch=include_branch)
            # ani_summary.session_obj_list[i].examine_raw(save=1)
            ani_summary.session_obj_list[i].calculate_dFF0(plot=0, plot_middle_step=0, save=0)
            ani_summary.session_obj_list[i].process_behavior_data()
            # ani_summary.session_obj_list[i].plot_bg_heatmaps(save=1)
            # ani_summary.session_obj_list[i].plot_heatmaps(save=1)
            # ani_summary.session_obj_list[i].actual_leave_vs_adjusted_optimal(save=0)
            # ani_summary.session_obj_list[i].extract_transient(plot_zscore=0)
            # ani_summary.session_obj_list[i].visualize_correlation_scatter(save=0)
            ani_summary.session_obj_list[i].visualize_average_traces(variable='time_in_port', method='even_time',
                                                                     block_split=True,
                                                                     plot_linecharts=0,
                                                                     plot_histograms=0)
        except:
            print(f"skipped session {i} because of error!!!")
            print("----------------------------------")
            continue
    # ses_selected_l, ses_selected_r = visualize_transient_consistency(session_obj_list, save=1, save_path=xsession_figure_export_dir)
    visualize_NRI_vs_amplitude_families(ani_summary, variable='time_in_port', block_split=True, normalized=False, save=False,
                                        save_path=None)
    return ani_summary


if __name__ == '__main__':
    # multi
    DAresponse = np.zeros(90)

    session_list = [1, 2, 3, 5, 7, 9, 11, 12, 14, 15, 19, 22, 23, 24, 25]
    summary_036 = multi_session_analysis('SZ036', session_list, include_branch='both')

    session_list = [0, 1, 2, 4, 5, 6, 8, 9, 11, 15, 16, 17, 18, 19, 20, 22, 24, 25, 27, 28, 29, 31, 32, 33, 35]
    summary_037 = multi_session_analysis('SZ037', session_list, include_branch='both')

    session_list = [1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32,
                    33, 34, 35, 36, 37, 38]
    summary_038 = multi_session_analysis('SZ038', session_list, include_branch='both')

    session_list = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22]
    summary_039 = multi_session_analysis('SZ039', session_list, include_branch='only_left')

    # session_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]
    # summary_041 = multi_session_analysis('SZ041', session_list, include_branch='only_right')
    session_list = [3, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 21, 22, 24, 26, 27, 28, 30]
    summary_042 = multi_session_analysis('SZ042', session_list, include_branch='both')

    session_list = [0, 1, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23]
    summary_043 = multi_session_analysis('SZ043', session_list, include_branch='only_right')

    # 1-way ANOVA with repeated measures, the within-subject factor being time invested in exp port before rewarded
    # DAresponse[0:5] = summary_036.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[5:10] = summary_036.NRI_amp_ipsi.mean_amp.to_numpy()
    # DAresponse[10:15] = summary_037.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[15:20] = summary_037.NRI_amp_ipsi.mean_amp.to_numpy()
    # DAresponse[20:25] = summary_038.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[25:30] = summary_038.NRI_amp_ipsi.mean_amp.to_numpy()
    # DAresponse[30:35] = summary_039.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[35:40] = summary_042.NRI_amp_contra.mean_amp.to_numpy()
    # DAresponse[40:45] = summary_043.NRI_amp_ipsi.mean_amp.to_numpy()
    # subject_np = np.repeat(
    #     ['036_contra', '036_ipsi', '037_contra', '037_ipsi', '038_contra', '038_ipsi', '039_contra', '042_contra',
    #      '043_ipsi'], 5)
    # time_invested = [1, 3, 5, 7, 9] * 9
    # df = pd.DataFrame({
    #     'Subject': subject_np,
    #     'TimeInvested': time_invested,
    #     'DAresponse': DAresponse
    # })
    # aov_1w = pg.rm_anova(dv='DAresponse', within='TimeInvested', subject='Subject', data=df, detailed=True)

    # 2-way ANOVA with repeated measures
    DAresponse[0:5] = summary_036.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[5:10] = summary_036.IRI_amp_ipsi.mean_amp_low.to_numpy()
    DAresponse[10:15] = summary_037.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[15:20] = summary_037.IRI_amp_ipsi.mean_amp_low.to_numpy()
    DAresponse[20:25] = summary_038.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[25:30] = summary_038.IRI_amp_ipsi.mean_amp_low.to_numpy()
    DAresponse[30:35] = summary_039.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[35:40] = summary_042.IRI_amp_contra.mean_amp_low.to_numpy()
    DAresponse[40:45] = summary_043.IRI_amp_ipsi.mean_amp_low.to_numpy()

    DAresponse[45:50] = summary_036.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[50:55] = summary_036.IRI_amp_ipsi.mean_amp_high.to_numpy()
    DAresponse[55:60] = summary_037.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[60:65] = summary_037.IRI_amp_ipsi.mean_amp_high.to_numpy()
    DAresponse[65:70] = summary_038.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[70:75] = summary_038.IRI_amp_ipsi.mean_amp_high.to_numpy()
    DAresponse[75:80] = summary_039.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[80:85] = summary_042.IRI_amp_contra.mean_amp_high.to_numpy()
    DAresponse[85:90] = summary_043.IRI_amp_ipsi.mean_amp_high.to_numpy()

    subject_np = np.tile(np.repeat(
        ['036_contra', '036_ipsi', '037_contra', '037_ipsi', '038_contra', '038_ipsi', '039_contra', '042_contra',
         '043_ipsi'], 5), 2)
    time_invested = [1, 3, 5, 7, 9] * 18
    block = np.repeat(['low', 'high'], 45)
    df = pd.DataFrame({
        'Subject': subject_np,
        'TimeInvested': time_invested,
        'Block': block,
        'DAresponse': DAresponse
    })
    aov_2w = pg.rm_anova(dv='DAresponse', within=['TimeInvested', 'Block'], subject='Subject', data=df, detailed=True)
    fig = plt.figure()
    ani_sum = summary_036
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='036 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='036 contra')
    ani_sum = summary_037
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='037 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='037 contra')
    ani_sum = summary_038
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='038 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='038 contra')
    ani_sum = summary_039
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='039 contra')
    # ani_sum = summary_041
    # plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
    #              yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='041 ipsi')
    # plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
    #              yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='041 contra')
    ani_sum = summary_042
    # plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
    #              yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='042 ipsi')
    plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
                 yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='042 contra')
    ani_sum = summary_043
    plt.errorbar(ani_sum.NRI_amp_ipsi['median_interval'], ani_sum.NRI_amp_ipsi['mean_amp'],
                 yerr=ani_sum.NRI_amp_ipsi['SEM_amp'], fmt='o-', capsize=5, label='043 ipsi')
    # plt.errorbar(ani_sum.NRI_amp_contra['median_interval'], ani_sum.NRI_amp_contra['mean_amp'],
    #              yerr=ani_sum.NRI_amp_contra['SEM_amp'], fmt='o-', capsize=5, label='043 contra')
    fig.legend()
    fig.show()
    print('Hello')
    # session_list = list(range(15))
    # session_obj_list = multi_session_analysis('SZ050', session_list, include_branch='both')
    # session_list = list(range(16))
    # session_obj_list = multi_session_analysis('SZ051', session_list, include_branch='both')
    # session_list = list(range(15))
    # session_obj_list = multi_session_analysis('SZ052', session_list, include_branch='both')
    # session_list = list(range(21))
    # session_obj_list = multi_session_analysis('SZ055', session_list, include_branch='both')

    # single
    # session_list = list(range(16))
    # session_obj_list = multi_session_analysis('SZ044', session_list, include_branch='both')
    # session_list = list(range(12))
    # session_obj_list = multi_session_analysis('SZ045', session_list, include_branch='only_left')
    # session_list = list(range(17))
    # session_obj_list = multi_session_analysis('SZ046', session_list, include_branch='both')
    # session_list = list(range(24))
    # session_obj_list = multi_session_analysis('SZ047', session_list, include_branch='both')
    # session_list = list(range(21))
    # session_obj_list = multi_session_analysis('SZ048', session_list, include_branch='both')
    # session_list = list(range(17))
    # session_obj_list = multi_session_analysis('SZ053', session_list, include_branch='both')
    # session_list = list(range(17))
    # session_obj_list = multi_session_analysis('SZ054', session_list, include_branch='both')
    # session_list = list(range(20))
    # session_obj_list = multi_session_analysis('SZ058', session_list, include_branch='both')
    # session_list = list(range(19))
    # session_obj_list = multi_session_analysis('SZ059', session_list, include_branch='both')
