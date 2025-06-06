import matplotlib.pyplot as plt


def plot_behav_events(pi_trials):
    for i in range(len(pi_trials.index)):
        x = pi_trials.entry[i]
        y = pi_trials.stay[i] * pi_trials.port_sign[i]
        reward_time = (pi_trials.rewards[i] - x) * pi_trials.port_sign[i]
        if len(reward_time) > 0:
            plt.scatter([x]*len(reward_time), reward_time, c='rebeccapurple', s=3)
        if pi_trials.port_sign[i] > 0:
            c = 'dodgerblue'
        else:
            c = 'olivedrab'
        plt.plot([x, x], [0, y], c=c)

    plt.xlabel("Time in session (sec)")
    plt.ylabel("Time of occupancy (sec)")
    plt.show()
