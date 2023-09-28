from scipy.signal import find_peaks, peak_widths, peak_prominences
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def quantify_da(col_name, dFF0, pi_events, plot=False):
    dff0 = dFF0[col_name]
    zscore = (dff0.to_numpy() - dff0.mean()) / dff0.std()
    threshold = np.percentile(zscore, 90)
    peaks, prop = find_peaks(zscore, rel_height=0.25, width=[4, 40], height=threshold, wlen=60, prominence=1)
    widths = peak_widths(zscore, peaks, rel_height=0.25)
    prominences = peak_prominences(zscore, peaks, wlen=60)
    for i in range(len(widths[3])):
        widths[2][i] = int(widths[2][i])
        widths[3][i] = int(widths[3][i])
    reward_time_exp = pi_events.time_recording[
        (pi_events.key == 'reward') & (pi_events.value == 1) & (pi_events.port == 1)]
    reward_time_bg = pi_events.time_recording[
        (pi_events.key == 'reward') & (pi_events.value == 1) & (pi_events.port == 2)]
    lick_time = pi_events.time_recording[
        (pi_events.key == 'lick') & (pi_events.value == 1)]
    auc = np.empty(len(peaks))
    for i in range(len(peaks)):
        peak_start = prominences[1][i]
        peak_end = prominences[2][i]
        auc[i] = integrate.trapezoid(y=dff0[peak_start:peak_end],
                                     x=dFF0.time_recording[peak_start:peak_end])
    if plot:
        plt.style.use('ggplot')
        plt.plot(dFF0["time_recording"], zscore)
        plt.plot(dFF0.time_recording[peaks], zscore[peaks], '*')
        plt.plot(dFF0.time_recording[prominences[1]], zscore[prominences[1]], 'x')
        plt.plot(dFF0.time_recording[prominences[2]], zscore[prominences[2]], 'x')
        plt.scatter(reward_time_exp, [-3] * len(reward_time_exp), label="exp")
        plt.scatter(reward_time_bg, [-3] * len(reward_time_bg), label="bg")
        plt.scatter(lick_time, [-3.2] * len(lick_time), marker='|')
        plt.hlines(y=widths[1], xmin=dFF0.time_recording[widths[2]], xmax=dFF0.time_recording[widths[3]],
                   colors='C2')
        plt.vlines(x=pi_events.time_recording[pi_events.is_1st_encounter], ymin=-2, ymax=8,
                   colors='grey', linestyles='dashdot', alpha=0.8, label='1st reward encounter')
        plt.axhline(y=threshold, color='grey', linestyle='-.', alpha=0.8)
        plt.title(col_name)
        plt.ylim([-4, 8])
        plt.xlabel('Time (sec)')
        plt.ylabel('zscore')
        plt.show()
    return zscore, peaks, widths, prominences, auc
