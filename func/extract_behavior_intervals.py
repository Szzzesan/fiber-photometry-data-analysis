import pandas as pd


def extract_behavior_intervals(event_timestamps_df):
    left_port = event_timestamps_df['left_entry', 'left_exit']
    right_port = event_timestamps_df['right_entry', 'right_exit']
    inter_port = pd.DataFrame(
        data=[event_timestamps_df['all_exit'][1:].to_numpy(), event_timestamps_df['all_entry'].to_numpy()])
