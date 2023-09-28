def remove_unfocused_trials(pi_trials):
    for i in range(len(pi_trials)):
        if int(pi_trials.port[i]) == 2 and (
                (pi_trials.num_reward[i] == 0) or (pi_trials.rewards[i][0] - pi_trials.entry[i] > 5000)):
            pi_trials.drop(i, inplace=True)

    pi_trials.reset_index(drop=True, inplace=True)
