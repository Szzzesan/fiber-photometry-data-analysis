import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def lin_reg_fit(denoised_df, session_label, plot='False'):
    # num_color_site = int(len(denoised_df.columns) / 2 - 1)
    fitted_df = denoised_df
    for i in [0, 1]:
        x = denoised_df.iloc[:, (i + 1) * 2].to_numpy() # x - isosbestic
        y = denoised_df.iloc[:, i * 2 + 1].to_numpy()  # y - 470 nm
        y_90 = np.nanpercentile(y, 90)
        y_without_outliers = y[y < y_90]
        x_without_outliers = x[np.where(y < y_90)[0]]
        mask = ~np.isnan(x) & ~np.isnan(y)
        y_without_outliers = y[mask]
        x_without_outliers = x[mask].reshape(-1, 1)
        reg = LinearRegression().fit(x_without_outliers, y_without_outliers)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        fitted_isos = slope * x + intercept
        fitted_df.iloc[:, (i + 1) * 2] = fitted_isos

    if plot:
        plt.style.use('ggplot')
        for i in [0, 1]:
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
