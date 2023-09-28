import pandas as pd


def get_structured_events(trials_df):
    # region Initialize a dataframe that can store timestamps of entry, exit, and rewards
    max_num_reward = trials_df.num_exp_rewards.max()
    exp_reward_df = pd.DataFrame(index=trials_df.index, columns=list(range(1, max_num_reward + 1)))
    exp_reward_df = exp_reward_df.add_prefix("exp_reward_")
    bg_reward_df = pd.DataFrame(index=trials_df.index, columns=list(range(1, 4 + 1)))
    bg_reward_df = bg_reward_df.add_prefix("bg_reward_")
    head_df = pd.DataFrame(index=trials_df.index, columns=['bg_entry', 'bg_exit', 'exp_entry', 'exp_exit'])
    trial_event = pd.concat([head_df, bg_reward_df, exp_reward_df], axis=1)
    # endregion

    # region Assign event timestamps to the corresponding columns/cells
    trial_event['trial'] = trials_df.trial
    trial_event['bg_entry'] = trials_df.valid_bg_entry
    trial_event['bg_exit'] = trials_df.valid_bg_exit
    trial_event['exp_entry'] = trials_df.valid_exp_entry
    trial_event['exp_exit'] = trials_df.valid_exp_exit
    trial_event['exp_leave_time'] = trials_df.leave_time
    trial_event['phase'] = trials_df.phase

    for idx in range(len(trial_event.index)):
        for r in range(max_num_reward):
            try:
                trial_event.iloc[idx, r+8] = trials_df.exp_rewards[idx][r]
            except:
                pass
    for idx in range(len(trial_event.index)):
        for r in range(4):
            try:
                trial_event.iloc[idx, r+4] = trials_df.bg_rewards[idx][r]
            except:
                pass
    # endregion
    return trial_event
