import pandas as pd


def extract_behavior_events(pi_events_df):
    all_lick = pi_events_df[
        (pi_events_df.key == 'lick') & (pi_events_df.value == 1)].time_recording.to_numpy()
    all_reward = pi_events_df[
        (pi_events_df.key == 'reward') & (pi_events_df.value == 1)].time_recording.to_numpy()
    all_entry = pi_events_df[
        (pi_events_df.key == 'head') & (pi_events_df.value == 1)].time_recording.to_numpy()
    all_exit = pi_events_df[
        (pi_events_df.key == 'head') & (pi_events_df.value == 0)].time_recording.to_numpy()

    exp_LED_on = pi_events_df[
        (pi_events_df.key == 'LED') & (pi_events_df.value == 1) & (pi_events_df.port == 1)].time_recording.to_numpy()
    exp_LED_off = pi_events_df[
        (pi_events_df.key == 'LED') & (pi_events_df.value == 0) & (pi_events_df.port == 1)].time_recording.to_numpy()
    exp_lick = pi_events_df[
        (pi_events_df.key == 'lick') & (pi_events_df.value == 1) & (pi_events_df.port == 1)].time_recording.to_numpy()
    exp_reward = pi_events_df[
        (pi_events_df.key == 'reward') & (pi_events_df.value == 1) & (pi_events_df.port == 1)].time_recording.to_numpy()
    exp_entry = pi_events_df[
        (pi_events_df.key == 'head') & (pi_events_df.value == 1) & (pi_events_df.port == 1)].time_recording.to_numpy()
    exp_exit = pi_events_df[
        (pi_events_df.key == 'head') & (pi_events_df.value == 0) & (pi_events_df.port == 1)].time_recording.to_numpy()

    bg_LED_on = pi_events_df[
        (pi_events_df.key == 'LED') & (pi_events_df.value == 1) & (pi_events_df.port == 2)].time_recording.to_numpy()
    bg_LED_off = pi_events_df[
        (pi_events_df.key == 'LED') & (pi_events_df.value == 0) & (pi_events_df.port == 2)].time_recording.to_numpy()
    bg_lick = pi_events_df[
        (pi_events_df.key == 'lick') & (pi_events_df.value == 1) & (pi_events_df.port == 2)].time_recording.to_numpy()
    bg_reward = pi_events_df[
        (pi_events_df.key == 'reward') & (pi_events_df.value == 1) & (pi_events_df.port == 2)].time_recording.to_numpy()
    bg_entry = pi_events_df[
        (pi_events_df.key == 'head') & (pi_events_df.value == 1) & (pi_events_df.port == 2)].time_recording.to_numpy()
    bg_exit = pi_events_df[
        (pi_events_df.key == 'head') & (pi_events_df.value == 0) & (pi_events_df.port == 2)].time_recording.to_numpy()

    event_timestamps = pd.DataFrame(
        data=[all_lick, all_reward, all_entry, all_exit, exp_lick, exp_reward, exp_entry, exp_exit,
              exp_LED_on, exp_LED_off, bg_lick, bg_reward, bg_entry, bg_exit, bg_LED_on, bg_LED_off]).T
    event_timestamps.columns = ['all_lick', 'all_reward', 'all_entry', 'all_exit',
                                'exp_lick', 'exp_reward', 'exp_entry', 'exp_exit', 'exp_LED_on',
                                'exp_LED_off',
                                'bg_lick', 'bg_reward', 'bg_entry', 'bg_exit', 'bg_LED_on', 'bg_LED_off']
    event_timestamps = event_timestamps.div(1000)
    return event_timestamps
