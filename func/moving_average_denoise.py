import matplotlib.pyplot as plt


def moving_average_denoise(detrended_df, session_label, plot='False', win_size=8):
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
