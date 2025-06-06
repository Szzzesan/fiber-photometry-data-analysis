import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_event_history(df, event_key, event_value, timescale_in_sec, new_col_realtime, new_col_rolling):
    df[new_col_realtime] = 0
    df[new_col_rolling] = np.nan
    for i in range(len(df.index)):
        t = df.time_recording[i]
        df.loc[i, new_col_realtime] = (df.key[i] == event_key) & (df.value[i] == event_value)
        if t > timescale_in_sec:
            df.loc[i, new_col_rolling] = df.loc[
                (df.time_recording > t - timescale_in_sec) & (df.time_recording <= t), new_col_realtime].apply(
                int).sum()
    return df


def slow_timescale_analysis(dFF0, pi_events, timescale_in_sec):
    num_frame = int(timescale_in_sec / 0.025)
    rolling_mean_right = dFF0['green_right'].rolling(num_frame).mean()
    rolling_mean_left = dFF0['green_left'].rolling(num_frame).mean()

    pi_events['block_change'] = np.nan
    for i in range(1, len(pi_events.index)):
        pi_events.block_change[i] = pi_events.phase[i] != pi_events.phase[i - 1]
    block_change = pi_events.time_recording[pi_events.block_change == True].to_numpy()

    add_event_history(pi_events, 'reward', 1, timescale_in_sec, 'reward_delivered', 'reward_collected')
    add_event_history(pi_events, 'lick', 1, timescale_in_sec, 'lick_initiated', 'lick_collected')
    add_event_history(pi_events, 'head', 1 or 0, timescale_in_sec, 'head_movement', 'head_movement_collected')

    DA_average_by_block_R = np.ndarray(shape=(len(block_change) + 1, 1), dtype=float)
    DA_average_by_block_R[0] = dFF0.green_right[dFF0.time_recording < block_change[0]].mean()
    DA_average_by_block_R[-1] = dFF0.green_right[dFF0.time_recording > block_change[-1]].mean()
    DA_average_by_block_L = np.ndarray(shape=(len(block_change) + 1, 1), dtype=float)
    DA_average_by_block_L[0] = dFF0.green_left[dFF0.time_recording < block_change[0]].mean()
    DA_average_by_block_L[-1] = dFF0.green_left[dFF0.time_recording > block_change[-1]].mean()
    for i in range(1, len(block_change)):
        DA_average_by_block_R[i] = dFF0.green_right[
            (dFF0.time_recording >= block_change[i - 1]) & (dFF0.time_recording < block_change[i])].mean()
        DA_average_by_block_L[i] = dFF0.green_left[
            (dFF0.time_recording >= block_change[i - 1]) & (dFF0.time_recording < block_change[i])].mean()

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Rolling DA Reponse to Behavior History', fontsize=30)
    axes[0].plot(dFF0['time_recording'] / 60, rolling_mean_right * 100, color='b',
                 label=f'{timescale_in_sec} sec dF/F0 rolling mean right')
    axes[0].plot(dFF0['time_recording'] / 60, rolling_mean_left * 100, color='tab:orange',
                 label=f'{timescale_in_sec} sec dF/F0 rolling mean left')
    axes[0].legend(fontsize=15)
    axes[0].set_ylabel('dF/F0 (%)')
    axes[1].plot(pi_events['time_recording'] / 60, pi_events['reward_collected'] / timescale_in_sec,
                 label=f'{timescale_in_sec} sec rolling reward rate')
    axes[1].legend(fontsize=15)
    axes[1].set_ylabel('Reward rate (/sec)')
    axes[2].plot(pi_events['time_recording'] / 60, pi_events['lick_collected'] / timescale_in_sec,
                 label=f'{timescale_in_sec} sec rolling lick rate')
    axes[2].legend(fontsize=15)
    axes[2].set_ylabel('Lick rate (/sec)')
    axes[3].plot(pi_events['time_recording'] / 60, pi_events['head_movement_collected'] / timescale_in_sec,
                 label=f'{timescale_in_sec} sec head movement')
    axes[3].legend(fontsize=15)
    axes[3].set_ylabel('Head movement (times/sec)')
    for i in range(len(block_change)):
        axes[0].axvline(x=block_change[i] / 60, color='k', linestyle='--', clip_on=False)
        axes[1].axvline(x=block_change[i] / 60, color='k', linestyle='--', clip_on=False)
        axes[2].axvline(x=block_change[i] / 60, color='k', linestyle='--', clip_on=False)
        axes[3].axvline(x=block_change[i] / 60, color='k', linestyle='--', clip_on=False)
    plt.xlabel('Time (min)')
    plt.xlim([0, 18.5])
    plt.subplots_adjust(hspace=0)
    fig.show()

    print('Hello')
