import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
import os


def calculate_dFF0_Hamilos(raw_separated, session_label, plot_middle_steps='False'):
    num_color_site = int(len(raw_separated.columns) / 2 - 1)

    # region Preprocessing
    detrended = _butterworth_detrend(raw_separated, fps=40, plot=plot_middle_steps, session_label=session_label)
    denoised = _moving_average_denoise(detrended, win_size=8, plot=plot_middle_steps, session_label=session_label)
    fitted = _lin_reg_fit(denoised, plot=plot_middle_steps, session_label=session_label)
    # endregion

    # region Subtraction
    F_0_timewindow = 10 #in sec
    fps = 40
    F_0_framewindow = F_0_timewindow * fps
    dFF0 = pd.DataFrame(
        columns=['time_recording', fitted.columns.values[1][:-4], fitted.columns.values[3][:-4]])
    dFF0.iloc[:, 0] = fitted.iloc[:, 0]
    for i in range(num_color_site):
        F_0 = fitted.iloc[:, 2 * i + 1].rolling(window=F_0_framewindow, center=True).mean()
        dFF0.iloc[:, i + 1] = (fitted.iloc[:, 2 * i + 1] - fitted.iloc[:, 2 * (i + 1)]) / F_0
    # endregion

    dFF0.iloc[:, 0] = dFF0.iloc[:, 0].div(1000)

    return dFF0

def _butterworth_detrend(raw_separated, session_label, fps=40, plot='False'):
    y1 = raw_separated['green_right_470'].to_numpy(copy=True)
    y2 = raw_separated['green_right_isos'].to_numpy(copy=True)
    y3 = raw_separated['green_left_470'].to_numpy(copy=True)
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
        detrended_r470 = detrended['green_right_470'].to_numpy()
        detrended_r405 = detrended['green_right_isos'].to_numpy()
        detrended_l470 = detrended['green_left_470'].to_numpy()
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


def _moving_average_denoise(detrended_df, session_label, plot='False', win_size=8):
    denoised_df = detrended_df
    for i in range(len(detrended_df.columns) - 2):
        denoised_df.iloc[:, i + 1] = detrended_df.iloc[:, i + 1].rolling(win_size).mean()
        denoised_df.iloc[0:win_size, i + 1] = denoised_df.iloc[win_size, i + 1]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for k in range(len(detrended_df.columns) - 2):
            ax.plot(denoised_df.iloc[10000:15000, 0], denoised_df.iloc[10000:15000, k + 1],
                     label=denoised_df.columns.values[k + 1])
        ax.legend()
        ax.set_title(session_label + ' Moving average denoised')
        ax.set_xlabel('Time recording')
        ax.set_ylabel('FP readout')
        fig.show()

    return denoised_df


def _lin_reg_fit(denoised_df, session_label, plot='False'):
    num_color_site = int(len(denoised_df.columns) / 2 - 1)
    fitted_df = denoised_df
    for i in range(num_color_site):
        x = denoised_df.iloc[:, (i + 1) * 2].to_numpy().reshape(-1, 1)  # x - isosbestic
        y = denoised_df.iloc[:, i * 2 + 1].to_numpy().reshape(-1, 1)  # y - actual
        y_90 = np.percentile(y, 90)
        y_without_outliers = y[y < y_90]
        x_without_outliers = x[np.where(y < y_90)[0]]
        reg = LinearRegression().fit(x_without_outliers, y_without_outliers)
        fitted_isos = reg.predict(x)
        fitted_df.iloc[:, (i + 1) * 2] = fitted_isos

    if plot:
        plt.style.use('ggplot')
        for i in range(num_color_site):
            plt.plot(fitted_df.iloc[10000:15000, 0], fitted_df.iloc[10000:15000, i * 2 + 1],
                     label=fitted_df.columns.values[i * 2 + 1])
            plt.plot(fitted_df.iloc[10000:15000, 0], fitted_df.iloc[10000:15000, (i + 1) * 2],
                     label=fitted_df.columns.values[(i + 1) * 2])
        # plt.legend()
        plt.xlabel('Time recording (msec)')
        plt.ylabel('Fluorescence intensity')
        plt.title(session_label + ' After linear regression fitting')
        plt.show()
    return fitted_df