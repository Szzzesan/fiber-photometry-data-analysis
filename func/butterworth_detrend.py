from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def butterworth_detrend(raw_separated, session_label, fps=80, plot='False'):
    y1 = raw_separated['green_right_actual'].to_numpy(copy=True)
    y2 = raw_separated['green_right_isos'].to_numpy(copy=True)
    y3 = raw_separated['green_left_actual'].to_numpy(copy=True)
    y4 = raw_separated['green_left_isos'].to_numpy(copy=True)
    detrended = raw_separated
    sos = signal.butter(2, 0.1, btype='highpass', fs=fps / 2, output='sos')
    for i in range(len(raw_separated.columns) - 2):
        sig = np.zeros(len(raw_separated) + 400)
        sig[400:] = raw_separated.iloc[:, i + 1].to_numpy()
        sig[0:400] = sig[400:800]
        sig = signal.sosfilt(sos, sig) + np.mean(sig)
        detrended.iloc[:, i + 1] = sig[400:]
        # if plot:
        #     plt.plot(detrended.iloc[:, 0], detrended.iloc[:, i + 1], label=raw_separated.columns.values[i + 1])

    if plot:
        xs = detrended['time_recording']/60
        detrended_r470 = detrended['green_right_actual'].to_numpy()
        detrended_r405 = detrended['green_right_isos'].to_numpy()
        detrended_l470 = detrended['green_left_actual'].to_numpy()
        detrended_l405 = detrended['green_left_isos'].to_numpy()
        fig, axes = plt.subplots(4, 2, figsize=(24, 10), sharex=True)
        fig.subplots_adjust(hspace=0)
        fig.suptitle('Raw Data Before and After Subtracting Slow Drift', fontsize=30)
        axes[0, 1].plot(xs, y1, c='k', label='raw R470')
        axes[0, 1].plot(xs, y1 - detrended_r470 + np.mean(y1), c='r', label='estimated slow drift')
        axes[0, 1].legend(fontsize=20)
        axes[1, 1].plot(xs, detrended_r470, c='tab:blue', label='R470 after subtracting slow drift')
        axes[1, 1].legend(fontsize=20)
        axes[2, 1].plot(xs, y2, c='k', label='raw R405')
        axes[2, 1].plot(xs, y2 - detrended_r405 + np.mean(y2), c='r', label='estimated slow drift')
        axes[2, 1].legend(fontsize=20)
        axes[3, 1].plot(xs, detrended_r405, c='tab:blue', label='R405 after subtracting slow drift')
        axes[3, 1].legend(fontsize=20)
        axes[0, 0].plot(xs, y3, c='k', label='raw L470')
        axes[0, 0].plot(xs, y3 - detrended_l470 + np.mean(y3), c='r', label='estimated slow drift')
        axes[0, 0].legend(fontsize=20)
        axes[1, 0].plot(xs, detrended_l470, c='tab:blue', label='L470 after subtracting slow drift')
        axes[1, 0].legend(fontsize=20)
        axes[2, 0].plot(xs, y4, c='k', label='raw L405')
        axes[2, 0].plot(xs, y4 - detrended_l405 + np.mean(y4), c='r', label='estimated slow drift')
        axes[2, 0].legend(fontsize=20)
        axes[3, 0].plot(xs, detrended_l405, c='tab:blue', label='L405 after subtracting slow drift')
        axes[3, 0].legend(fontsize=20)
        fig.show()


    return detrended
