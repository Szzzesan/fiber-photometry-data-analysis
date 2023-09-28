import pandas as pd
import numpy as np


def extract_raw_trial(pi_events_df):
    pi_no_cam = pi_events_df[pi_events_df.key != "camera"]
    pi_no_cam.reset_index(drop=True, inplace=True)
    pnc = pi_no_cam
    total_poke = int(len(pnc[pnc.key == "head"]) / 2)
    pi_trials = pd.DataFrame(index=range(total_poke), columns=["port", "entry", "exit", "rewards", "licks", "phase"])
    poke_num = 0
    for i in range(len(pnc)):
        event_type = [int(pnc.iloc[i, 7]), pnc.iloc[i, 8]]
        if event_type == [1, 'head']:
            pi_trials.port[poke_num] = pnc.port[i]
            pi_trials.entry[poke_num] = pnc.session_time[i]
            pi_trials.phase[poke_num] = pnc.phase[i]
            lick_arr = []
            reward_arr = []
        if event_type == [1, 'lick']:
            lick_arr.append(pnc.session_time[i])
        if event_type == [1, 'reward']:
            reward_arr.append(pnc.session_time[i])
        if event_type == [0, 'head']:
            pi_trials.licks[poke_num] = lick_arr
            pi_trials.rewards[poke_num] = reward_arr
            pi_trials.exit[poke_num] = pnc.session_time[i]
            poke_num = poke_num + 1
    pi_trials = pi_trials[~pd.isna(pi_trials["exit"])]
    pi_trials.reset_index(drop=True, inplace=True)

    # region Extract secondary features
    pi_trials["stay"] = pi_trials.exit - pi_trials.entry
    num_reward = np.empty(len(pi_trials), dtype='int')
    num_lick = np.empty(len(pi_trials), dtype='int')
    port_sign = np.empty(len(pi_trials), dtype='int')
    for i in range(len(pi_trials)):
        num_reward[i] = len(pi_trials.rewards[i])
        num_lick[i] = len(pi_trials.licks[i])
        if pi_trials.port[i] == 2:
            port_sign[i] = -1
        else:
            port_sign[i] = 1
    pi_trials["num_reward"] = num_reward
    pi_trials["num_lick"] = num_lick
    pi_trials["port_sign"] = port_sign
    # endregion

    return pi_trials
