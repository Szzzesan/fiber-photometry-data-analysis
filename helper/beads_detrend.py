import matplotlib.pyplot as plt
import numpy as np
import pybeads as pbd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def single_channel_detrend(y, channel_name, plot=False):
    y_base = np.mean(y)
    y = y - y_base

    # extend both ends smoothly to zero
    xscale_l, xscale_r = 200, 450
    dx = 1
    y_l = np.mean(y[0]) * sigmoid(1 / xscale_l * np.arange(-5 * xscale_l, 5 * xscale_l, dx))
    y_r = np.mean(y[-1]) * sigmoid(-1 / xscale_r * np.arange(-5 * xscale_r, 5 * xscale_r, dx))
    y_ext = np.hstack([y_l, y, y_r])
    y_ext = y_ext.reshape(-1, 1)
    # len_l, len_o, len_r = len(y_l), len(y), len(y_r)
    # plt.plot(range(len_l, len_l + len_o), y)
    # plt.plot(y_l, 'C1')
    # plt.plot(range(len_l + len_o, len_l + len_o + len_r), y_r, 'C1')
    # plt.show()

    # filter parameters
    fc = 0.002 # 0.0001
    d = 1
    # Positivity bias (peaks are positive)
    r = 10
    # regularization parameters
    amp = np.percentile(y, 99.5)
    lam0 = 0.5 * amp
    lam1 = 5 * amp
    lam2 = 4 * amp
    Nit = 15
    pen = 'L1_v2'
    signal_est, bg_est, cost = pbd.beads(y_ext, d, fc, r, Nit, lam0, lam1, lam2, pen)
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        fig.subplots_adjust(hspace=0)
        fig.suptitle(
            f'{channel_name}, BEADS (fc={fc}, d={d}, r={r}, amp={amp}, xscale_l={xscale_l}, xscale_r={xscale_r})')
        axes[0].plot(y, c='k', label='original data')
        axes[0].plot(bg_est[10 * xscale_l: -10 * xscale_r], c='r', label='slow drift estimated by BEADS')
        axes[0].legend()
        axes[1].plot(y - bg_est[10 * xscale_l: -10 * xscale_r], label='after subtracting slow drift')
        axes[1].legend()
        fig.show()

    return y + y_base - bg_est[10 * xscale_l:-10 * xscale_r], bg_est[10 * xscale_l:-10 * xscale_r]


def beads_detrend(raw_separated, session_label, plot='False'):
    xs = raw_separated['time_recording'].to_numpy() / 60000
    y1 = raw_separated['green_right_470'].to_numpy()
    detrended_r470, bg_r470 = single_channel_detrend(y1, 'green right 470', plot=False)
    y2 = raw_separated['green_right_isos'].to_numpy()
    detrended_r405, bg_r405 = single_channel_detrend(y2, 'green right isos', plot=False)
    y3 = raw_separated['green_left_470'].to_numpy()
    detrended_l470, bg_l470 = single_channel_detrend(y3, 'green left 470', plot=False)
    y4 = raw_separated['green_left_isos'].to_numpy()
    detrended_l405, bg_l405 = single_channel_detrend(y4, 'green left isos', plot=False)

    detrended = raw_separated
    detrended['green_right_actual'] = detrended_r470
    detrended['green_right_isos'] = detrended_r405
    detrended['green_left_actual'] = detrended_l470
    detrended['green_left_isos'] = detrended_l405

    if plot:
        fig, axes = plt.subplots(4, 2, figsize=(24, 10), sharex=True)
        fig.subplots_adjust(hspace=0)
        fig.suptitle('Raw Data Before and After Subtracting Slow Drift', fontsize=30)
        axes[0, 1].plot(xs, y1, c='k', label='raw R470')
        axes[0, 1].plot(xs, bg_r470 + np.mean(y1), c='r', label='estimated slow drift')
        axes[0, 1].legend(fontsize=20)
        axes[1, 1].plot(xs, detrended_r470, label='R470 after subtracting slow drift')
        axes[1, 1].legend(fontsize=20)
        axes[2, 1].plot(xs, y2, c='k', label='raw R405')
        axes[2, 1].plot(xs, bg_r405 + np.mean(y2), c='r', label='estimated slow drift')
        axes[2, 1].legend(fontsize=20)
        axes[3, 1].plot(xs, detrended_r405, label='R405 after subtracting slow drift')
        axes[3, 1].legend(fontsize=20)
        axes[0, 0].plot(xs, y3, c='k', label='raw L470')
        axes[0, 0].plot(xs, bg_l470 + np.mean(y3), c='r', label='estimated slow drift')
        axes[0, 0].legend(fontsize=20)
        axes[1, 0].plot(xs, detrended_l470, label='L470 after subtracting slow drift')
        axes[1, 0].legend(fontsize=20)
        axes[2, 0].plot(xs, y4, c='k', label='raw L405')
        axes[2, 0].plot(xs, bg_l405 + np.mean(y4), c='r', label='estimated slow drift')
        axes[2, 0].legend(fontsize=20)
        axes[3, 0].plot(xs, detrended_l405, label='L405 after subtracting slow drift')
        axes[3, 0].legend(fontsize=20)
        fig.show()

    return detrended
