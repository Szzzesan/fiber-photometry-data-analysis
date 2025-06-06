import matplotlib.pyplot as plt
import os


def plot_average(slow_average, fast_average, save_path, save_name, left_or_right, save=0):
    time_plot = slow_average['time_aligned'].tolist()
    slow_average_list = slow_average['average'].tolist()
    slow_shade_l = slow_average['std_l'].tolist()
    slow_shade_h = slow_average['std_h'].tolist()
    fast_average_list = fast_average['average'].tolist()
    fast_shade_l = fast_average['std_l'].tolist()
    fast_shade_h = fast_average['std_h'].tolist()
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].plot(time_plot, slow_average_list, label='slow block')
    for _x in [2.5, 5, 7.5, 10]:
        ax[0].axvline(_x, linestyle='--', color='k')
    ax[0].fill_between(time_plot, slow_shade_l, slow_shade_h, alpha=0.2)
    ax[1].plot(time_plot, fast_average_list, 'red', label='fast block')
    ax[1].fill_between(time_plot, fast_shade_l, fast_shade_h, alpha=0.2)
    for _x in [1.25, 2.5, 3.75, 5]:
        ax[1].axvline(_x, linestyle='--', color='k')
    for subplot in [0, 1]:
        ax[subplot].set_ylim(-1.5, 4)
    fig.text(0.04, 0.5, 'dF/F0 (%)', va='center', rotation='vertical')
    fig.legend()
    fig.suptitle(save_name)
    fig.show()

    if save:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path, save_name + '_' + left_or_right + '.png')
        fig.savefig(fig_name)