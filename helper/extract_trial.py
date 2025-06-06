import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper


def get_leave_times(pi_events):
    reward_trials = pi_events[(pi_events.key == 'reward_initiate')].trial.to_numpy()
    non_reward = ~pi_events.trial.isin(reward_trials)
    bg_end_times = pi_events[(pi_events.key == 'LED') & (pi_events.port == 2) & (pi_events.value == 1) & non_reward]
    exp_entries = pi_events[(pi_events.key == 'head') & (pi_events.value == 1) & (pi_events.port == 1) & non_reward]
    exp_exits = pi_events[(pi_events.key == 'head') & (pi_events.value == 0) & (pi_events.port == 1) & non_reward]
    bg_end_times = bg_end_times[bg_end_times.time_recording < exp_entries.time_recording.max()]
    ind, dif = helper.min_dif(bg_end_times.time_recording, exp_entries.time_recording, return_index=True)
    exp_entries = exp_entries.iloc[np.unique(ind)]
    exp_entries = exp_entries.groupby('trial').time_recording.max()
    exp_exits = exp_exits.groupby('trial').time_recording.max()
    valid_trials = np.intersect1d(exp_exits.index.values, exp_entries.index.values)
    exp_exits = exp_exits.loc[valid_trials]
    exp_entries = exp_entries.loc[valid_trials]
    if len(exp_exits.to_numpy()) != len(exp_entries.to_numpy()):
        print()
    exp_entries = exp_entries.reset_index()
    exp_exits = exp_exits.reset_index()
    leave_times = exp_exits.time_recording.to_numpy() - exp_entries.time_recording.to_numpy()
    return exp_entries, exp_exits, leave_times


def extract_trial(pi_events_df):
    num_trial = int(pi_events_df.trial.dropna().max())
    if (pi_events_df.value[pi_events_df.key == 'trial'].tail(1) == 1).values:
        num_trial = num_trial - 1
    if (pi_events_df.value[pi_events_df.key == 'trial'].head(1) == 0).values:
        num_trial = num_trial - 1
    pi_trials = pd.DataFrame(index=range(num_trial),
                             columns=["trial", "bg_entries", "bg_exits", "bg_rewards", "bg_licks", "exp_entries",
                                      "exp_exits", "exp_rewards", "exp_licks", "phase", "trial_start", "trial_end"])
    pi_trials.trial = pi_trials.index + 1
    for trial in pi_trials.trial:
        enex = helper.get_entry_exit(pi_events_df, trial)
        i = trial - 1
        pi_trials.bg_entries[i] = enex[0]
        pi_trials.bg_exits[i] = enex[1]
        pi_trials.exp_entries[i] = enex[2]
        pi_trials.exp_exits[i] = enex[3]
        isreward = pi_events_df.key == 'reward'
        islick = pi_events_df.key == 'lick'
        on = pi_events_df.value == 1
        istrial = pi_events_df.trial == trial
        isbg = pi_events_df.port == 2
        isexp = pi_events_df.port == 1
        pi_trials.bg_rewards[i] = pi_events_df.time_recording[istrial & isbg & isreward & on].to_numpy()
        pi_trials.bg_licks[i] = pi_events_df.time_recording[istrial & isbg & islick & on].to_numpy()
        pi_trials.exp_rewards[i] = pi_events_df.time_recording[istrial & isexp & isreward & on].to_numpy()
        pi_trials.exp_licks[i] = pi_events_df.time_recording[istrial & isexp & islick & on].to_numpy()
        pi_trials.phase[i] = pi_events_df.phase.values[(pi_events_df.key == 'trial') & on & istrial][0]
        pi_trials.trial_start[i] = pi_events_df.time_recording.values[(pi_events_df.key == 'trial') & on & istrial][0]
        pi_trials.trial_end[i] = pi_events_df.time_recording.values[(pi_events_df.key == 'trial') & (~on) & istrial][0]
    # pi_trials = pi_trials[~pd.isna(pi_trials["exit"])]
    # pi_trials.reset_index(drop=True, inplace=True)

    # region Extract secondary features
    num_exp_rewards = np.empty(len(pi_trials.index), dtype='int')
    num_bg_exits = np.empty(len(pi_trials.index), dtype='int')
    num_exp_exits = np.empty(len(pi_trials.index), dtype='int')
    valid_bg_entry = np.empty(len(pi_trials.index), dtype='float')
    valid_bg_exit = np.empty(len(pi_trials.index), dtype='float')
    for i in range(len(pi_trials.index)):
        num_exp_rewards[i] = len(pi_trials.exp_rewards[i])
        num_bg_exits[i] = len(pi_trials.bg_exits[i])
        num_exp_exits[i] = len(pi_trials.exp_exits[i])
        valid_bg_entry[i] = [entry for entry in pi_trials.bg_entries[i] if entry <= pi_trials.bg_rewards[i][0]][-1]
        valid_bg_exit[i] = [exit for exit in pi_trials.bg_exits[i] if exit >= pi_trials.bg_rewards[i][-1]][0]
    pi_trials["num_exp_rewards"] = num_exp_rewards
    pi_trials["num_bg_reentries"] = num_bg_exits - 1
    pi_trials["num_exp_reentries"] = num_exp_exits - 1
    valid_exp_entry, valid_exp_exit, leave_times = get_leave_times(pi_events_df)
    pi_trials["leave_time"] = leave_times[0:num_trial]
    pi_trials['valid_exp_entry'] = valid_exp_entry.time_recording[0:num_trial]
    pi_trials['valid_exp_exit'] = valid_exp_exit.time_recording[0:num_trial]
    pi_trials['valid_bg_entry'] = valid_bg_entry[0:num_trial]
    pi_trials['valid_bg_exit'] = valid_bg_exit[0:num_trial]
    # endregion

    return pi_trials
