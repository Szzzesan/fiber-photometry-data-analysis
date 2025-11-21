import numpy as np
from figure_making.data_loader import load_session_dataframe


def match_recording_time_to_trial_time(time_recording_df, trial_df):
    recording_times = time_recording_df['time_recording'].to_numpy()

    exp_entries = trial_df['exp_entry'].to_numpy()
    exp_exits = trial_df['exp_exit'].to_numpy()
    bg_entries = trial_df['bg_entry'].to_numpy()
    bg_exits = trial_df['bg_exit'].to_numpy()
    trial_numbers = trial_df['trial'].to_numpy()  # Get trial IDs for assignment
    block = trial_df['phase'].to_numpy()

    time_recording_df['phase'] = np.nan
    time_recording_df['trial'] = np.nan
    time_recording_df['port'] = np.nan
    time_recording_df['time_in_port'] = np.nan

    # Map time points to trials and calculating relative time ---
    for i in range(len(trial_df)):
        exp_entry_time = exp_entries[i]
        exp_exit_time = exp_exits[i]
        bg_entry_time = bg_entries[i]
        bg_exit_time = bg_exits[i]
        trial_id = trial_numbers[i]
        block_i = block[i]
        is_in_bg = (recording_times >= bg_entry_time) & (recording_times < bg_exit_time)
        is_in_exp = (recording_times >= exp_entry_time) & (recording_times < exp_exit_time)
        is_in_trial = (recording_times >= bg_entry_time) & (recording_times < exp_exit_time)
        bg_indices_to_update = time_recording_df.loc[is_in_bg].index
        exp_indices_to_update = time_recording_df.loc[is_in_exp].index
        trial_indices_to_update = time_recording_df.loc[is_in_trial].index
        if not exp_indices_to_update.empty:
            time_recording_df.loc[trial_indices_to_update, 'phase'] = block_i
            time_recording_df.loc[trial_indices_to_update, 'trial'] = trial_id
            time_recording_df.loc[exp_indices_to_update, 'port'] = 'exp'
            time_recording_df.loc[bg_indices_to_update, 'port'] = 'bg'
            exp_time_relative_to_entry = (
                    time_recording_df.loc[exp_indices_to_update, 'time_recording'] - exp_entry_time
            )
            time_recording_df.loc[exp_indices_to_update, 'time_in_port'] = exp_time_relative_to_entry
            bg_time_relative_to_entry = (
                    time_recording_df.loc[bg_indices_to_update, 'time_recording'] - bg_entry_time
            )
            time_recording_df.loc[bg_indices_to_update, 'time_in_port'] = bg_time_relative_to_entry

    time_recording_df = time_recording_df.dropna(subset=['phase', 'trial', 'time_in_port']).reset_index(
        drop=True)
    return time_recording_df


if __name__ == '__main__':
    animal_str = 'SZ036'
    dFF0 = load_session_dataframe(animal_str, 'dFF0', session_id=15, file_format='parquet')
    trial_df = load_session_dataframe(animal_str, 'trial_df', session_id=15, file_format='parquet')
    dFF0_matched = match_recording_time_to_trial_time(dFF0, trial_df)
