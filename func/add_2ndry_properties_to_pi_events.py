import numpy as np
import pandas as pd
import func


def identify_event_order(pi_events, col_name_to_add, condition):
    pi_events[col_name_to_add] = np.NaN
    for tr in range(1, pi_events.total_trial + 1):
        all_condition_bg = (pi_events.trial == tr) & condition & (pi_events.port == 2)
        all_condition_exp = (pi_events.trial == tr) & condition & (pi_events.port == 1)
        pi_events[col_name_to_add].loc[all_condition_bg] = range(1, len(pi_events.index[all_condition_bg]) + 1)
        pi_events[col_name_to_add].loc[all_condition_exp] = range(1, len(pi_events.index[all_condition_exp]) + 1)


def get_valid_entry_exit(pi_events):
    reward_trials = pi_events[(pi_events.key == 'reward_initiate')].trial.to_numpy()
    non_reward = ~pi_events.trial.isin(reward_trials)
    bg_end_times = pi_events[(pi_events.key == 'LED') & (pi_events.port == 2) & (pi_events.value == 1) & non_reward]
    exp_entries = pi_events[(pi_events.key == 'head') & (pi_events.value == 1) & (pi_events.port == 1) & non_reward]
    exp_exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0) & (pi_events.port == 1) & non_reward]
    bg_end_times = bg_end_times[bg_end_times.time_recording < exp_entries.time_recording.max()]
    ind, dif = func.min_dif(bg_end_times.time_recording, exp_entries.time_recording, return_index=True)
    exp_entries = exp_entries.iloc[np.unique(ind)]
    grouped_entries = exp_entries.groupby('trial')
    grouped_exits = exp_exits.groupby('trial')
    exp_entries = grouped_entries.time_recording.max()
    exp_exits = grouped_exits.time_recording.max()
    exp_entries_idx = [grouped_entries.groups[i][-1] for i in range(1, len(grouped_entries.groups) + 1)]
    exp_exits_idx = [grouped_exits.groups[i][-1] for i in range(1, len(grouped_exits.groups) + 1)]
    valid_trials = np.intersect1d(exp_exits.index.values, exp_entries.index.values)
    valid_trials = [int(valid_trials[i]) for i in range(len(valid_trials))]
    exp_exits = exp_exits.loc[valid_trials]
    exp_entries = exp_entries.loc[valid_trials]
    exp_entries_idx = [exp_entries_idx[i - 1] for i in valid_trials]
    exp_exits_idx = [exp_exits_idx[i - 1] for i in valid_trials]
    if len(exp_exits.to_numpy()) != len(exp_entries.to_numpy()):
        print()
    exp_entries = exp_entries.reset_index()
    exp_exits = exp_exits.reset_index()

    bg_entries = pi_events[(pi_events.key == 'trial') & (pi_events.value == 1) & (pi_events.port == 2) & non_reward]
    bg_exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0) & (pi_events.port == 2) & non_reward]
    grouped_bg_entries = bg_entries.groupby('trial')
    grouped_bg_exits = bg_exits.groupby('trial')
    bg_entries_idx = [grouped_bg_entries.groups[i][0] for i in range(1, len(grouped_bg_entries.groups) + 1)]
    bg_exits_idx = [grouped_bg_exits.groups[i][-1] for i in range(1, len(grouped_bg_exits.groups) + 1)]
    bg_entries_idx = [bg_entries_idx[i - 1] for i in valid_trials]
    bg_exits_idx = [bg_exits_idx[i - 1] for i in valid_trials]
    bg_entries = grouped_bg_entries.time_recording.max()
    bg_exits = grouped_bg_exits.time_recording.max()
    bg_entries = bg_entries.loc[valid_trials]
    bg_exits = bg_exits.loc[valid_trials]
    bg_entries = bg_entries.reset_index()
    bg_exits = bg_exits.reset_index()
    return exp_entries_idx, exp_exits_idx, bg_entries_idx, bg_exits_idx, exp_entries, exp_exits, bg_entries, bg_exits


def add_2ndry_properties_to_pi_events(pi_events):
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
            print('no lick after this reward hence no first lick')
        else:
            first_lick_idx = pi_events[isafter & islick & (pi_events.value == 1)].index[0]
            pi_events.is_1st_lick[first_lick_idx] = 1
        if (pi_events.value[isafter & islick].iloc[0] == 0) & (pi_events.index[isafter & islick].min() < next_idx):
            pi_events.is_1st_encounter[idx] = 1
        elif pi_events.value[isafter & islick].iloc[0] == 1:
            idx_to_change = pi_events[isafter & islick].index[0]
            pi_events.is_1st_encounter[idx_to_change] = 1
    pi_events['is_1st_lick'] = pi_events['is_1st_lick'].astype(bool)
    pi_events['is_1st_encounter'] = pi_events['is_1st_encounter'].astype(bool)
    # endregion
    # region Identify the valid entries and exits
    exp_entries_idx, exp_exits_idx, bg_entries_idx, bg_exits_idx, exp_entries, exp_exits, bg_entries, bg_exits = get_valid_entry_exit(
        pi_events)
    pi_events['is_valid'] = 0
    pi_events.is_valid[exp_entries_idx] = 1
    pi_events.is_valid[exp_exits_idx] = 1
    pi_events.is_valid[bg_entries_idx] = 1
    pi_events.is_valid[bg_exits_idx] = 1
    # endregion
    # region Get the travel time
    pi_events['travel_b2e'] = 0
    travel_time_b2e = np.empty(len(exp_entries_idx) + 1)
    travel_time_b2e[:] = np.nan
    travel_time_b2e[0:-1] = [(exp_entries.time_recording[i] - bg_exits.time_recording[i]) for i in
                             range(len(exp_entries))]
    pi_events['travel_b2e'] = travel_time_b2e[pd.factorize(pi_events.trial)[0]]
    # endregion
    # region Get the consumption time
    # endregion
    return pi_events
