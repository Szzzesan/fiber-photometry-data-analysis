import numpy as np
import pandas as pd


def construct_reward_history_matrix(df, binsize):
    trials_back = 4  # Number of trials back (rows)
    bins_per_trial = int(2.5/binsize)  # Number of bins per trial
    shape = (trials_back + 1, bins_per_trial)
    df.attrs['sparse_shape'] = shape
    # Add bin index within each trial
    # df = df.copy()
    df['bin_idx'] = df.groupby('trial').cumcount()
    # Store reward matrices for each bin
    # reward_matrix_list = []
    # df['history_matrix'] = [np.random.rand(5, 25) for _ in range(len(df))]
    df['history_matrix_sparse'] = [None] * len(df)
    for i, row in df.iterrows():

        current_trial = row['trial']
        current_bin_idx = row['bin_idx']

        # Create an empty matrix with zeros
        history_matrix = np.zeros((trials_back + 1, bins_per_trial))

        for trial_offset in range(trials_back + 1):  # 0 (current trial) to 4 trials back
            target_trial = current_trial - trial_offset
            for bin_offset in range(bins_per_trial):
                target_bin_idx = current_bin_idx - bin_offset
                target_bin = df[(df['trial'] == target_trial) & (df['bin_idx'] == target_bin_idx)]
                # Store the reward count if found
                if target_bin.shape[0] > 0:
                    history_matrix[trial_offset, bin_offset] = target_bin['reward_num'].values[0]
        history_matrix = history_matrix.astype(bool)
        sparse_indices = np.flatnonzero(history_matrix)

        # reward_matrix_list.append(history_matrix)
        df.at[i, 'history_matrix_sparse'] = sparse_indices
        print(f'Finished matrix for trial {current_trial} bin {current_bin_idx}')
    return df

    # return np.array(reward_matrix_list)  # Shape: (num_bins, trials_back+1, bins_per_trial)