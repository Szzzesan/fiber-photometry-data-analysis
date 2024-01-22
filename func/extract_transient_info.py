import pandas as pd
from func.quantify_da import quantify_da
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt


def add_event_as_column(transient_df, col_name_in_transient_df, pi_events, condition, search_range):
    transient_df[col_name_in_transient_df] = ''
    for i in range(len(transient_df.index)):
        pt = transient_df.peak_time[i]
        search_start = pt + search_range[0]
        search_end = pt + search_range[1]
        is_in_search_range = (pi_events.time_recording > search_start) & (pi_events.time_recording < search_end)
        events_within_range = pi_events.time_recording[is_in_search_range & condition].to_numpy()
        transient_df[col_name_in_transient_df].iloc[i] = events_within_range


def calculate_occurrence_given_transient(col_in_transient_df):
    event_occur = np.zeros(len(col_in_transient_df))
    for i in range(len(event_occur)):
        event_occur[i] = len(col_in_transient_df[i])
    occur_given_transient = np.count_nonzero(event_occur) / len(event_occur)
    return occur_given_transient


def get_event2peak_interval(transient, col_name, direction=1):
    event_in_range = np.empty(len(transient.index))
    event_in_range[:] = np.nan
    for i in range(len(event_in_range)):
        if len(transient[col_name].iloc[i]) > 0:
            event_in_range[i] = transient[col_name][i][0]
    event2peak_interval = (transient.peak_time - event_in_range) * direction
    return event2peak_interval


def find_closest_values(s, x):
    idx = (np.abs(s - x)).argmin()
    return idx


def get_rewardtime_related_to_event(pi_events, transient_df, created_col, condition, before_or_after):
    transient_df[created_col] = np.nan
    for i in range(len(transient_df.index)):
        if len(transient_df.prior_reward[i]) > 0:
            time_prior_reward = transient_df.prior_reward[i][0]
            if before_or_after == 'before':
                time_last_event = pi_events.time_recording[
                    (pi_events.time_recording < time_prior_reward) & condition].max()
                transient_df.at[i, created_col] = time_prior_reward - time_last_event
            if before_or_after == 'after':
                time_first_event = pi_events.time_recording[
                    (pi_events.time_recording > time_prior_reward) & condition].min()
                transient_df.at[i, created_col] = time_first_event - time_prior_reward


def get_reward_info(pi_events, transient_df):
    transient_df['reward_time'] = np.nan
    transient_df['reward_order'] = np.nan
    transient_df['num_reward_in_halfsec_after'] = np.nan
    transient_df['num_reward_in_halfsec_before'] = np.nan
    transient_df['is_1st_reward'] = np.zeros(len(transient_df.index))
    transient_df['is_1st_reward'] = transient_df['is_1st_reward'].astype(bool)
    transient_df['closest_idx_in_dFF0'] = np.nan
    transient_df['is_from_valid_trial'] = True
    for i in range(len(transient_df.index)):
        if len(transient_df.prior_reward[i]) > 0:
            time_prior_reward = transient_df.prior_reward[i][0]
            transient_df.at[i, 'reward_time'] = time_prior_reward
            idx = find_closest_values(pi_events.time_recording, time_prior_reward)
            transient_df.at[i, 'closest_idx_in_dFF0'] = idx
            transient_df.at[i, 'reward_order'] = pi_events.reward_order_in_trial[idx]
            if pi_events.reward_order_in_trial[idx] == 1:
                transient_df.at[i, 'is_1st_reward'] = True
            if pi_events.is_valid_trial[idx] == False:
                transient_df.at[i, 'is_from_valid_trial'] = False
            events_halfsec_after = pi_events[
                (pi_events.time_recording <= time_prior_reward + 0.5) & (pi_events.time_recording > time_prior_reward)]
            events_halfsec_before = pi_events[
                (pi_events.time_recording <= time_prior_reward) & (pi_events.time_recording > time_prior_reward - 0.5)]
            transient_df.at[i, 'num_reward_in_halfsec_after'] = len(
                events_halfsec_after[(events_halfsec_after.key == 'reward') & (events_halfsec_after.value == 1)])
            transient_df.at[i, 'num_reward_in_halfsec_before'] = len(
                events_halfsec_before[(events_halfsec_before.key == 'reward') & (events_halfsec_before.value == 1)])
    transient_df['is_end_reward'] = np.zeros(len(transient_df.index))
    transient_df['is_end_reward'] = transient_df['is_end_reward'].astype(bool)
    idx_endreward = transient_df[transient_df.port == 1].groupby('trial')['reward_order'].idxmax().values.astype(int)
    idx_endreward = idx_endreward[idx_endreward > 0]
    transient_df['is_end_reward'].iloc[idx_endreward] = True


def extract_transient_info(col_name, dFF0, pi_events, plot_zscore=0, plot=0):
    zscore, peaks, widths, prominences, auc = quantify_da(col_name, dFF0, pi_events, plot=plot_zscore)
    peaktime = dFF0.time_recording[peaks].to_numpy()
    reward_time = pi_events.time_recording[(pi_events.key == 'reward') & (pi_events.value == 1)]
    firstlick_time = pi_events.time_recording[pi_events.is_1st_lick]
    firstencounter_time = pi_events.time_recording[pi_events.is_1st_encounter]
    transient_df = pd.DataFrame(
        columns=['peak_idx', 'peak_time', 'height', 'transient_start', 'transient_end', 'width', 'AUC'])
    # region The properties of the transients themselves
    transient_df['peak_idx'] = peaks
    transient_df['peak_time'] = peaktime
    transient_df['height'] = dFF0[col_name].iloc[peaks].to_numpy()
    transient_df['transient_start'] = dFF0.time_recording[prominences[1]].to_numpy()
    transient_df['transient_end'] = dFF0.time_recording[prominences[2]].to_numpy()
    transient_df['width'] = transient_df['transient_end'] - transient_df['transient_start']
    transient_df['AUC'] = auc
    # endregion
    # region Find events surrounding each transient
    add_event_as_column(transient_df, 'prior_reward', pi_events, search_range=[-1, -0.2],
                        condition=((pi_events.key == 'reward') & (pi_events.value == 1)))
    add_event_as_column(transient_df, 'peri_reward', pi_events, search_range=[-0.4, 0.4],
                        condition=((pi_events.key == 'reward') & (pi_events.value == 1)))
    add_event_as_column(transient_df, 'post_reward', pi_events, search_range=[0.2, 1],
                        condition=((pi_events.key == 'reward') & (pi_events.value == 1)))

    add_event_as_column(transient_df, 'prior_1st_encounter', pi_events, search_range=[-1, -0.1],
                        condition=pi_events.is_1st_encounter)

    add_event_as_column(transient_df, 'prior_1st_lick', pi_events, search_range=[-0.9, -0.1],
                        condition=pi_events.is_1st_lick)
    add_event_as_column(transient_df, 'post_1st_lick', pi_events, search_range=[0.2, 1],
                        condition=pi_events.is_1st_lick)

    add_event_as_column(transient_df, 'prior_entry', pi_events, search_range=[-0.9, -0.1],
                        condition=((pi_events.key == 'head') & (pi_events.value == 1)))
    add_event_as_column(transient_df, 'post_entry', pi_events, search_range=[0.2, 1],
                        condition=((pi_events.key == 'head') & (pi_events.value == 1)))

    add_event_as_column(transient_df, 'prior_exit', pi_events, search_range=[-0.9, -0.1],
                        condition=((pi_events.key == 'head') & (pi_events.value == 0)))
    add_event_as_column(transient_df, 'post_exit', pi_events, search_range=[0.2, 1],
                        condition=((pi_events.key == 'head') & (pi_events.value == 0)))
    # endregion
    # region Calculate the occurrences of behavior events in the vicinity of each transient
    transient_df.reward_prior_given_transient = calculate_occurrence_given_transient(transient_df['prior_reward'])
    transient_df.reward_peri_given_transient = calculate_occurrence_given_transient(transient_df['peri_reward'])
    transient_df.reward_post_given_transient = calculate_occurrence_given_transient(transient_df['post_reward'])
    transient_df.encounter1_prior_given_transient = calculate_occurrence_given_transient(
        transient_df['prior_1st_encounter'])
    transient_df.lick1_prior_given_transient = calculate_occurrence_given_transient(transient_df['prior_1st_lick'])
    transient_df.lick1_post_given_transient = calculate_occurrence_given_transient(transient_df['post_1st_lick'])
    transient_df.entry_prior_given_transient = calculate_occurrence_given_transient(transient_df['prior_entry'])
    transient_df.entry_post_given_transient = calculate_occurrence_given_transient(transient_df['post_entry'])
    transient_df.exit_prior_given_transient = calculate_occurrence_given_transient(transient_df['prior_exit'])
    transient_df.exit_post_given_transient = calculate_occurrence_given_transient(transient_df['post_exit'])
    # endregion
    # region Calculate the time shift of each event relative to each transient
    transient_df['r2p'] = get_event2peak_interval(transient_df, 'prior_reward', direction=1)
    transient_df['p2r'] = get_event2peak_interval(transient_df, 'post_reward', direction=1)
    transient_df['r2p2r'] = get_event2peak_interval(transient_df, 'peri_reward', direction=1)
    transient_df['e2p'] = get_event2peak_interval(transient_df, 'prior_1st_encounter', direction=1)
    transient_df['l2p'] = get_event2peak_interval(transient_df, 'prior_1st_lick', direction=1)
    transient_df['p2l'] = get_event2peak_interval(transient_df, 'post_1st_lick', direction=1)
    transient_df['n2p'] = get_event2peak_interval(transient_df, 'prior_entry', direction=1)
    transient_df['p2n'] = get_event2peak_interval(transient_df, 'post_entry', direction=1)
    transient_df['x2p'] = get_event2peak_interval(transient_df, 'prior_exit', direction=1)
    transient_df['p2x'] = get_event2peak_interval(transient_df, 'post_exit', direction=1)
    transient_df.var_r2p = stats.variance(transient_df.r2p[~transient_df['r2p'].isna()])
    transient_df.var_p2r = stats.variance(transient_df.p2r[~transient_df['p2r'].isna()])
    transient_df.var_r2p2r = stats.variance(transient_df.r2p2r[~transient_df['r2p2r'].isna()])
    transient_df.var_e2p = stats.variance(transient_df.e2p[~transient_df['e2p'].isna()])
    transient_df.var_l2p = stats.variance(transient_df.l2p[~transient_df['l2p'].isna()])
    transient_df.var_p2l = stats.variance(transient_df.p2l[~transient_df['p2l'].isna()])
    transient_df.var_n2p = stats.variance(transient_df.n2p[~transient_df['n2p'].isna()])
    transient_df.var_p2n = stats.variance(transient_df.p2n[~transient_df['p2n'].isna()])
    transient_df.var_x2p = stats.variance(transient_df.x2p[~transient_df['x2p'].isna()])
    transient_df.var_p2x = stats.variance(transient_df.p2x[~transient_df['p2x'].isna()])
    # endregion
    # region Find which block/port/trial the mouse is in at the time of each peak
    transient_df['block'] = ''
    transient_df['port'] = ''
    transient_df['trial'] = ''
    for i in range(len(transient_df.index)):
        closest_idx = find_closest_values(pi_events.time_recording, transient_df.peak_time[i])
        transient_df.at[i, 'block'] = pi_events.phase[closest_idx]
        transient_df.at[i, 'port'] = pi_events.port[closest_idx]
        transient_df.at[i, 'trial'] = pi_events.trial[closest_idx]
    # endregion
    # region Calculate the time relationship between reward and all other behavior events
    get_reward_info(pi_events, transient_df)
    get_rewardtime_related_to_event(pi_events, transient_df, 'ts_reward',
                                    condition=(pi_events.key == 'reward') & (pi_events.value == 1),
                                    before_or_after='before')
    get_rewardtime_related_to_event(pi_events, transient_df, 'ts_entry',
                                    condition=(pi_events.key == 'head') & (pi_events.value == 1),
                                    before_or_after='before')
    get_rewardtime_related_to_event(pi_events, transient_df, 'tt_exit',
                                    condition=(pi_events.key == 'head') & (pi_events.value == 0),
                                    before_or_after='after')
    transient_df['ts_entry_or_reward'] = transient_df['ts_reward']
    transient_df.loc[transient_df.is_1st_reward, ['ts_entry_or_reward']] = transient_df['ts_entry'].loc[
        transient_df.is_1st_reward]

    # endregion
    if plot:
        plt.style.use('ggplot')
        plt.plot(dFF0['time_recording'], dFF0[col_name]*100)
        plt.plot(transient_df['peak_time'], transient_df['height']*100, '*', label='peak')
        plt.plot(dFF0.time_recording[prominences[1]], dFF0[col_name].iloc[prominences[1]]*100, 'x',
                 label='start of transient')
        plt.plot(dFF0.time_recording[prominences[2]], dFF0[col_name].iloc[prominences[2]]*100, 'x',
                 label='end of transient')
        reward_time_exp = pi_events.time_recording[
            (pi_events.key == 'reward') & (pi_events.value == 1) & (pi_events.port == 1)]
        reward_time_bg = pi_events.time_recording[
            (pi_events.key == 'reward') & (pi_events.value == 1) & (pi_events.port == 2)]
        lick_time = pi_events.time_recording[
            (pi_events.key == 'lick') & (pi_events.value == 1)]
        plt.scatter(reward_time_exp, [-2] * len(reward_time_exp), label="exp reward")
        plt.scatter(reward_time_bg, [-2] * len(reward_time_bg), label="bg reward")
        plt.scatter(lick_time, [-2.5] * len(lick_time), marker='|')
        plt.vlines(x=pi_events.time_recording[pi_events.is_1st_encounter], ymin=-3, ymax=6,
                   colors='grey', linestyles='dashdot', alpha=0.8, label='1st reward encounter')
        threshold = np.percentile(dFF0[col_name], 90)
        plt.axhline(y=threshold*100, color='grey', linestyle='dotted', alpha=0.8, label='90th percentile')
        plt.title(col_name)
        plt.ylim([-3, 6])
        plt.xlabel('Time (sec)')
        plt.ylabel('dF/F0 (%)')
        plt.show()

    return transient_df
