import pandas as pd
import helper


def process_block_transition(prev_block, current_block, df_intervals_bg, reward_columns, zscore, branch):
    """Processes transition between two blocks and returns the necessary DataFrames."""
    last_two = df_intervals_bg[df_intervals_bg['block_sequence'] == prev_block].iloc[-2:]
    first_four = df_intervals_bg[df_intervals_bg['block_sequence'] == current_block].iloc[:4]

    transition_df = pd.concat([last_two, first_four], ignore_index=True)

    time_series_df, trial_info_df = helper.construct_matrix_for_average_traces(
        zscore, branch,
        transition_df['entry'].to_numpy(),
        transition_df['exit'].to_numpy(),
        transition_df['trial'].to_numpy(),
        transition_df['block'].to_numpy()
    )

    reward_df = transition_df[reward_columns].subtract(transition_df['entry'], axis=0)

    return transition_df, time_series_df, trial_info_df, reward_df
