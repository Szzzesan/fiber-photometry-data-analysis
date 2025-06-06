import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_intervals_expreward(pi_events, plot_histograms=0, ani_str='animal', ses_str='session'):
    is_exp = (pi_events['port'] == 1)
    is_bg = (pi_events['port'] == 2)
    is_reward = (pi_events['key'] == 'reward') & (pi_events['value'] == 1)
    is_entry = (pi_events['key'] == 'head') & (pi_events['value'] == 1) & (pi_events['is_valid'])
    is_exit = (pi_events['key'] == 'head') & (pi_events['value'] == 0) & (pi_events['is_valid'])
    arr_reward_exp = pi_events.loc[is_exp & is_reward, 'time_recording'].to_numpy()
    arr_reward_bg = pi_events.loc[is_bg & is_reward, 'time_recording'].to_numpy()
    arr_lastreward_exp = np.insert(arr_reward_exp[:-1], 0, np.nan)
    arr_nextreward_exp = np.append(arr_reward_exp[1:], np.nan)
    arr_NRI = pi_events.loc[is_exp & is_reward, 'time_in_port'].to_numpy()
    arr_trial_exp = pi_events.loc[is_reward & is_exp, 'trial']
    arr_block_exp = pi_events.loc[is_reward & is_exp, 'phase']
    df_intervals = pd.DataFrame(
        {'trial': arr_trial_exp, 'block': arr_block_exp, 'last_reward_time': arr_lastreward_exp,
         'reward_time': arr_reward_exp, 'next_reward_time': arr_nextreward_exp, 'time_in_port': arr_NRI})
    first_rewards_idx = df_intervals.groupby('trial').head(1).index
    last_rewards_idx = df_intervals.groupby('trial').tail(1).index
    df_intervals.loc[first_rewards_idx, 'last_reward_time'] = np.nan
    df_intervals.loc[last_rewards_idx, 'next_reward_time'] = np.nan
    first_rewards_times = df_intervals.loc[first_rewards_idx, 'reward_time'].to_numpy()
    last_bg_rewards = np.searchsorted(arr_reward_bg, first_rewards_times, side='right') - 1
    valid_bg_mask = (last_bg_rewards > 0)
    df_intervals.loc[first_rewards_idx[valid_bg_mask], 'last_reward_time'] = arr_reward_bg[
        last_bg_rewards[valid_bg_mask]]
    df_intervals['IRI_prior'] = df_intervals['reward_time'] - df_intervals['last_reward_time']
    df_intervals['IRI_post'] = df_intervals['next_reward_time'] - df_intervals['reward_time']
    df_intervals.reset_index(inplace=True, drop=True)
    arr_recent_reward_rate = np.zeros(df_intervals.shape[0])
    arr_recent_reward_rate[:] = np.nan
    arr_recent_reward_rate_exp = np.zeros(df_intervals.shape[0])
    arr_recent_reward_rate_exp[:] = np.nan
    arr_local_reward_rate_1sec = np.zeros(df_intervals.shape[0])
    arr_local_reward_rate_1sec[:] = np.nan

    arr_exp_exits = np.zeros(df_intervals.shape[0])
    arr_exp_exits[:] = np.nan
    arr_exp_entries = np.full(df_intervals.shape[0], np.nan)

    for i in range(df_intervals.shape[0]):
        search_begin = max(0, df_intervals.loc[i, 'reward_time'] - 30)
        search_end = df_intervals.loc[i, 'reward_time']
        is_in_range = (pi_events['time_recording'] >= search_begin) & (pi_events['time_recording'] < search_end)
        onesec_begin = max(0, df_intervals.loc[i, 'reward_time'] - 1)
        onesec_end = df_intervals.loc[i, 'reward_time']
        is_in_1sec = (pi_events['time_recording'] >= onesec_begin) & (pi_events['time_recording'] < onesec_end)
        arr_recent_reward_rate[i] = pi_events.loc[is_in_range & is_reward].shape[0] / (search_end - search_begin)
        arr_recent_reward_rate_exp[i] = pi_events.loc[is_in_range & is_reward & is_exp].shape[0] / (
                search_end - search_begin)
        arr_local_reward_rate_1sec[i] = pi_events.loc[is_in_1sec & is_reward].shape[0]

        # find the exponential port entries and exits for each trial
        def extract_single_time(time_values, trial, event_type):
            if time_values.shape[0] == 1:
                return time_values[0]
            elif time_values.shape[0] > 1:
                raise ValueError(
                    f"Found {time_values.shape[0]} valid time investment port {event_type}s for trial {trial}!")

        is_trial = (pi_events['trial'] == df_intervals.loc[i, 'trial'])

        time_entry = pi_events.loc[is_trial & is_exp & is_entry, 'time_recording'].to_numpy()
        time_exit = pi_events.loc[is_trial & is_exp & is_exit, 'time_recording'].to_numpy()
        arr_exp_entries[i] = extract_single_time(time_entry, df_intervals.loc[i, 'trial'], "entry")
        arr_exp_exits[i] = extract_single_time(time_exit, df_intervals.loc[i, 'trial'], "exit")

    df_intervals['entry_time'] = arr_exp_entries
    df_intervals['exit_time'] = arr_exp_exits
    df_intervals['recent_reward_rate'] = arr_recent_reward_rate
    df_intervals['recent_reward_rate_exp'] = arr_recent_reward_rate_exp
    df_intervals['local_reward_rate_1sec'] = arr_local_reward_rate_1sec
    df_intervals['num_rewards_prior'] = df_intervals.groupby('trial')['reward_time'].cumcount()
    if plot_histograms:
        fig = plt.figure(figsize=(10, 6))
        x = df_intervals.loc[df_intervals['block'] == '0.8', 'time_in_port'].to_list()
        y = df_intervals.loc[df_intervals['block'] == '0.4', 'time_in_port'].to_list()
        bins = np.linspace(0, 16, 9)
        plt.hist(x, bins, alpha=0.5, label='block 0.8')
        plt.hist(y, bins, alpha=0.5, label='block 0.4')
        plt.legend(loc='upper right')
        plt.title(f"{ani_str}\n{ses_str}\nHistogram of NRIs")
        plt.xlabel('NRI (sec)')
        plt.ylabel('# of intervals')
        fig.show()

        fig = plt.figure(figsize=(10, 6))
        x = df_intervals.loc[df_intervals['block'] == '0.8', 'IRI_post'].to_list()
        y = df_intervals.loc[df_intervals['block'] == '0.4', 'IRI_post'].to_list()
        bins = np.linspace(0, 6, 7)
        plt.hist(x, bins, alpha=0.5, label='block 0.8')
        plt.hist(y, bins, alpha=0.5, label='block 0.4')
        plt.legend(loc='upper right')
        plt.title(f"{ani_str}\n{ses_str}\nHistogram of IRIs")
        plt.xlabel('IRI (sec)')
        plt.ylabel('# of intervals')
        fig.show()
    return df_intervals


def extract_intervals_bg_inport(pi_events, plot_histograms=0, ani_str='animal', ses_str='session'):
    # Find the entry and exit of the background port for each trial
    is_bg_entry = (pi_events['key'] == 'trial') & (pi_events['value'] == 1) & pi_events['is_valid']
    is_bg_exit = (pi_events['port'] == 2) & (pi_events['key'] == 'head') & (pi_events['value'] == 0) & pi_events[
        'is_valid']

    arr_bg_entry = pi_events.loc[is_bg_entry, 'time_recording'].to_numpy()
    arr_bg_exit = pi_events.loc[is_bg_exit, 'time_recording'].to_numpy()

    arr_trial_bg = pi_events.loc[is_bg_entry, 'trial'].to_numpy()
    arr_block_bg = pi_events.loc[is_bg_entry, 'phase'].to_numpy()

    # Find all 4 rewards of the background port in each trial
    is_bg = pi_events['port'] == 2
    is_reward = (pi_events['key'] == 'reward') & (pi_events['value'] == 1)

    # Initialize arrays for rewards efficiently
    num_trials = len(arr_trial_bg)
    arr_rewards = {order: np.full(num_trials, np.nan) for order in range(1, 5)}

    # Helper function to extract reward times for a specific order
    def get_reward_time(trial, order):
        is_trial = pi_events['trial'] == trial
        condition = is_bg & is_reward & is_trial & (pi_events['reward_order_in_trial'] == order)
        result = pi_events.loc[condition, 'time_recording']
        # if the block logged at the time of reward is different from that at the time of entry, use the logging at the time of reward
        if (pi_events.loc[condition, 'phase'].to_numpy() != arr_block_bg[arr_trial_bg == trial]):
            print(f'Block of trial {trial} corrected')
            arr_block_bg[arr_trial_bg == trial] = pi_events.loc[condition, 'phase']
        return result.iloc[0] if not result.empty else np.nan

    # Populate reward arrays
    for i, trial in enumerate(arr_trial_bg):
        for order in arr_rewards.keys():
            arr_rewards[order][i] = get_reward_time(trial, order)

    df_intervals = pd.DataFrame({
        'trial': arr_trial_bg,
        'block': arr_block_bg,
        'entry': arr_bg_entry,
        'exit': arr_bg_exit,
        'reward_1': arr_rewards[1],
        'reward_2': arr_rewards[2],
        'reward_3': arr_rewards[3],
        'reward_4': arr_rewards[4]
    })

    df_intervals['is_transition_trial'] = df_intervals['block'] != df_intervals['block'].shift()
    df_intervals['block_sequence'] = (df_intervals['block'] != df_intervals['block'].shift()).cumsum()

    return df_intervals
