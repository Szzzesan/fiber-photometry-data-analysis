import numpy as np
import pandas as pd
from .get_interlocked_arrays import get_interlocked_arrays


def identify_event_order(pi_events, col_name_to_add, condition):
    pi_events[col_name_to_add] = np.NaN
    for tr in range(1, pi_events.total_trial + 1):
        all_condition_bg = (pi_events.trial == tr) & condition & (pi_events.port == 2)
        all_condition_exp = (pi_events.trial == tr) & condition & (pi_events.port == 1)
        pi_events[col_name_to_add].loc[all_condition_bg] = range(1, len(pi_events.index[all_condition_bg]) + 1)
        pi_events[col_name_to_add].loc[all_condition_exp] = range(1, len(pi_events.index[all_condition_exp]) + 1)


def get_valid_entry_exit(pi_events):
    def update_trial_validity(valid_trials):
        nonlocal exp_entries, exp_exits, bg_entries, bg_exits
        exp_entries = exp_entries[exp_entries.trial.isin(valid_trials)]
        exp_exits = exp_exits[exp_exits.trial.isin(valid_trials)]
        bg_entries = bg_entries[bg_entries.trial.isin(valid_trials)]
        bg_exits = bg_exits[bg_exits.trial.isin(valid_trials)]

    pi_events['is_valid_trial'] = True
    reward_trials = pi_events[(pi_events.key == 'reward_initiate')].trial.to_numpy()
    non_reward = ~pi_events.trial.isin(reward_trials)
    bg_end_times = pi_events[(pi_events.key == 'LED') & (pi_events.port == 2) & (pi_events.value == 1)]

    exp_entries = pi_events[(pi_events.key == 'head') & (pi_events.value == 1) & (pi_events.port == 1)]
    exp_exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0) & (pi_events.port == 1)]
    bg_entries = pi_events[(pi_events.key == 'trial') & (pi_events.value == 1) & (pi_events.port == 2)]
    bg_exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0) & (pi_events.port == 2)]
    # todo: give a valid entry and exit to each trial regardless of whether the trial is valid
    # todo: debug get_interlocked_arrays()
    vid_bg_entries, vid_exp_exits = get_interlocked_arrays(bg_entries.index.to_numpy(), exp_exits.index.to_numpy(),
                                                           direction='widest')
    vid_bg_entries, vid_bg_exits = get_interlocked_arrays(bg_entries.index.to_numpy(), bg_exits.index.to_numpy(),
                                                          direction='widest')

    # bgx_bgn = np.subtract.outer(bg_exits.index.to_numpy(), vid_bg_entries)
    # expx_bgx = np.subtract.outer(vid_exp_exits, bg_exits.index.to_numpy())
    # product_for_bg_exits = np.matmul(bgx_bgn > 0, expx_bgx > 0)
    # product_rolled = np.roll(product_for_bg_exits, -1)
    # subtracted = np.subtract(product_for_bg_exits.astype(int), product_rolled.astype(int))
    # vid_bg_exits = bg_exits.index.to_numpy()[list(  (subtracted.diagonal())[0])]

    expn_bgx = np.subtract.outer(exp_entries.index.to_numpy(), vid_bg_exits)
    expx_expn = np.subtract.outer(vid_exp_exits, exp_entries.index.to_numpy())
    product_for_intervals = np.matmul(expx_expn > 0, expn_bgx > 0)
    intervals_usable = list(np.where(product_for_intervals.diagonal())[0])
    expn_bgx = expn_bgx[:, intervals_usable]
    expx_expn = expx_expn[intervals_usable, :]
    product_for_exp_entries = np.matmul(expn_bgx > 0, expx_expn > 0)
    vid_exp_entries = exp_entries.index.to_numpy()[list(np.where(product_for_exp_entries.diagonal() == 1)[0])]
    vid_exp_entries, vid_exp_exits_new = get_interlocked_arrays(vid_exp_entries, vid_exp_exits, direction='widest')
    # todo: examine why the returned arrays skip some elements
    vid_bg_exits, vid_exp_entries = get_interlocked_arrays(vid_bg_exits, exp_entries.index.to_numpy(),
                                                           direction='narrowest')
    vid_exp_entries, vid_exp_exits = get_interlocked_arrays(vid_exp_entries, vid_exp_exits, direction='widest')
    vid_bg_entries, vid_exp_exits = get_interlocked_arrays(vid_bg_entries, vid_exp_exits, direction='widest')
    pi_events['is_valid'] = False
    pi_events.loc[vid_exp_entries, 'is_valid'] = True
    pi_events.loc[vid_exp_exits, 'is_valid'] = True
    pi_events.loc[vid_bg_entries, 'is_valid'] = True
    pi_events.loc[vid_bg_exits, 'is_valid'] = True
    # region only get the trials where there are both an exp_entry and an exp_exit
    complete_trials = set(exp_entries.trial) & set(exp_exits.trial) & set(bg_entries.trial) & set(bg_exits.trial)
    update_trial_validity(complete_trials)
    # endregion
    # region valid condition 1: toss trials with multiple re-entry into the exponential port
    # but skip this step when it's a single-reward task
    if (pi_events['task'].iloc[0] != 'single_reward'):
        is_single_entry = exp_entries.groupby('trial').entry_order_in_trial.max() == 1
        single_entry_trials = is_single_entry[is_single_entry].index
        update_trial_validity(single_entry_trials)
        pi_events.is_valid_trial[~pi_events['trial'].isin(single_entry_trials)] = False
    # endregion
    # region valid condition 2: toss trials with background stay too long
    bg_stay = bg_exits.groupby('trial').trial_time.max()
    trial_phase = bg_exits.groupby('trial').phase.max()
    good_for_long_block = (trial_phase == '0.4') & (bg_stay < 15)
    good_for_short_block = (trial_phase == '0.8') & (bg_stay < 7.5)
    good_bg_stay = good_for_long_block | good_for_short_block
    good_bg_trials = good_bg_stay[good_bg_stay].index
    update_trial_validity(good_bg_trials)
    pi_events.loc[~pi_events['trial'].isin(good_bg_trials), 'is_valid_trial'] = False
    for trial in trial_phase.index.to_list():
        pi_events.loc[pi_events.trial == trial, 'phase'] = trial_phase.loc[trial]
    # endregion
    valid_trials = np.unique(pi_events.trial[pi_events.is_valid_trial])
    bg_entries_idx = [bg_entries.groupby('trial').groups[i].max() for i in valid_trials]
    bg_exits_idx = [bg_exits.groupby('trial').groups[i].max() for i in valid_trials]
    exp_entries_idx = [exp_entries.groupby('trial').groups[i].max() for i in valid_trials]
    exp_exits_idx = [exp_exits.groupby('trial').groups[i].max() for i in valid_trials]
    bg_entries = bg_entries.time_recording.to_numpy()
    bg_exits = bg_exits.groupby('trial').time_recording.max().to_numpy()
    exp_entries = exp_entries.time_recording.to_numpy()
    exp_exits = exp_exits.time_recording.to_numpy()

    return pi_events, valid_trials, exp_entries, exp_exits, bg_entries, bg_exits


def add_2ndry_properties_to_pi_events(pi_events):
    pd.options.mode.chained_assignment = None  # default='warn'
    pi_events.total_trial = int(pi_events.trial.max())

    # region Identify the order of rewards, entries, and exits in each trial
    identify_event_order(pi_events, 'reward_order_in_trial',
                         condition=(pi_events.key == 'reward') & (pi_events.value == 1))
    identify_event_order(pi_events, 'entry_order_in_trial',
                         condition=(pi_events.key == 'head') & (pi_events.value == 1))
    identify_event_order(pi_events, 'exit_order_in_trial',
                         condition=(pi_events.key == 'head') & (pi_events.value == 0))
    # endregion

    # region Identify the animal's first encounters of each reward
    pi_events['is_1st_lick'] = 0
    pi_events['is_1st_encounter'] = 0
    reward_idx = pi_events.index[(pi_events.key == 'reward') & (pi_events.value == 1)]
    for i in range(len(reward_idx)):
        idx = reward_idx[i]
        if i + 1 >= len(reward_idx):
            next_idx = max(pi_events.index)
        else:
            next_idx = reward_idx[i + 1]
        isafter = pi_events.index > idx
        islick = pi_events.key == 'lick'
        if pi_events[isafter & islick & (pi_events.value == 1)].empty:
            print('')
        else:
            first_lick_idx = pi_events[isafter & islick & (pi_events.value == 1)].index[0]
            pi_events.is_1st_lick[first_lick_idx] = 1

        if pi_events[isafter & islick].empty:
            print('')
        elif (pi_events.value[isafter & islick].iloc[0] == 0) & (pi_events.index[isafter & islick].min() < next_idx):
            pi_events.is_1st_encounter[idx] = 1
        elif pi_events.value[isafter & islick].iloc[0] == 1:
            idx_to_change = pi_events[isafter & islick].index[0]
            pi_events.is_1st_encounter[idx_to_change] = 1
    pi_events['is_1st_lick'] = pi_events['is_1st_lick'].astype(bool)
    pi_events['is_1st_encounter'] = pi_events['is_1st_encounter'].astype(bool)
    # endregion
    pi_events, valid_trials, exp_entries, exp_exits, bg_entries, bg_exits = get_valid_entry_exit(pi_events)
    return pi_events
