import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper.beads_detrend import beads_detrend
from helper.butterworth_detrend import butterworth_detrend
from helper.moving_average_denoise import moving_average_denoise
from helper.lin_reg_fit import lin_reg_fit
import os


def calculate_dFF0_corrected(raw_separated, session_label, save_path, plot='False',
                             plot_middle_steps='False', save='False'):
    num_color_site = int(len(raw_separated.columns) / 2 - 1)

    # region Preprocessing
    # raw_separated = framedrop_remedy(raw_separated, plot=plot_middle_steps)
    # detrended = butterworth_detrend(raw_separated, fps=40, plot=plot_middle_steps, session_label=session_label)
    detrended = beads_detrend(raw_separated, plot=plot_middle_steps, session_label=session_label)
    # # region temporary: examining a random 150-sec snippet of raw data
    # snippet = detrended[(detrended['time_recording'] > 660000) & (detrended['time_recording'] < 810000)]
    # plt.plot(snippet['time_recording']/1000, snippet['green_left_actual'])
    # plt.xlim(660, 810)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Raw measurement of photons (470 nm)')
    # plt.title(f'{session_label} left')
    # plt.show()
    # # endregion
    denoised = moving_average_denoise(detrended, win_size=8, plot=plot_middle_steps, session_label=session_label)
    fitted = lin_reg_fit(denoised, plot=plot_middle_steps, session_label=session_label)
    # endregion

    # region Subtraction
    F_0_timewindow = 360
    fps = 40
    F_0_framewindow = F_0_timewindow * fps
    dFF0 = pd.DataFrame(
        columns=['time_recording', fitted.columns.values[1][:-4], fitted.columns.values[3][:-4]]) # for the end product
    dFF0.iloc[:, 0] = denoised.iloc[:, 0]
    dFF0_split_channel = denoised.copy()
    F0_split_channel = denoised.copy()
    for i in range(1, 5):
        F0 = denoised.iloc[:, i].rolling(window=F_0_framewindow, center=True).mean()
        start_mean = denoised.iloc[:F_0_framewindow, i].mean()
        end_mean = denoised.iloc[-F_0_framewindow:, i].mean()
        half_window = (F_0_framewindow - 1) // 2
        F0.iloc[:half_window] = F0.iloc[:half_window].fillna(start_mean)
        F0.iloc[-half_window:] = F0.iloc[-half_window:].fillna(end_mean)
        Ft = denoised.iloc[:, i]
        dF = Ft - F0
        dFF0_split_channel.iloc[:, i] = dF/F0
        F0_split_channel.iloc[:, i] = F0

    dFF0['green_right'] = dFF0_split_channel['green_right_470'] - dFF0_split_channel['green_right_isos']
    dFF0['green_left'] = dFF0_split_channel['green_left_470'] - dFF0_split_channel['green_left_isos']
    dFF0['F0_right'] = F0_split_channel['green_right_470'] - F0_split_channel['green_right_isos']
    dFF0['F0_left'] = F0_split_channel['green_left_470'] - F0_split_channel['green_left_isos']
    # endregion

    dFF0.iloc[:, 0] = dFF0.iloc[:, 0].div(1000)

    if plot:
        plt.style.use('ggplot')
        for i in [0, 1]:
            plt.plot(dFF0.iloc[24000:30000, 0], dFF0.iloc[24000:30000, i + 1] * 100, label=dFF0.columns.values[i + 1], alpha=0.7)
        plt.legend()
        plt.xlabel('Time recording (sec)')
        plt.ylabel('dF/F0 (%)')
        plt.title(session_label + " dF/F0")
        fig = plt.gcf()
        plt.show()

        if save:
            isExist = os.path.exists(save_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(save_path)
                print("A new directory is created!")
            fig_name = os.path.join(save_path, 'dFF0' + '.png')
            fig.savefig(fig_name)

    return dFF0
