import numpy as np

import data_loader
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 300


# --- First plotting method starts here ---
def figc_example_1d_traces(zscore, pi_events, ax=None):
    # example_snippet_width = 15  # in sec
    # snippet_begin = 430
    # snippet_end = snippet_begin + example_snippet_width
    snippet_begin = 430.61
    snippet_end = 444.85
    timestamps_to_plot = zscore.loc[
        (zscore['time_recording'] > snippet_begin) & (zscore['time_recording'] < snippet_end), 'time_recording'].values
    DA_trace_to_plot = zscore.loc[
        (zscore['time_recording'] > snippet_begin) & (zscore['time_recording'] < snippet_end), 'green_left'].values
    events_within = pi_events[
        (pi_events['time_recording'] > snippet_begin) & (pi_events['time_recording'] < snippet_end)]

    is_entry = (pi_events['key'] == 'head') & (pi_events['value'] == 1)
    is_exit = (pi_events['key'] == 'head') & (pi_events['value'] == 0)
    is_reward = (pi_events['key'] == 'reward') & (pi_events['value'] == 1)
    is_lick = (pi_events['key'] == 'lick') & (pi_events['value'] == 1)
    is_exp = pi_events['port'] == 1
    is_bg = pi_events['port'] == 2
    entries = events_within.loc[is_entry, 'time_recording'].values
    exits = events_within.loc[is_exit, 'time_recording'].values
    rewards = events_within.loc[is_reward, 'time_recording'].values
    licks = events_within.loc[is_lick, 'time_recording'].values
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 1))
        return_handle = True
    else:
        fig = None
        return_handle = False
    draw_vertical_lines(ax, licks, ymin=0.9, ymax=1, color='grey', alpha=0.1)
    draw_vertical_lines(ax, entries, color='b', linestyle='--')
    draw_vertical_lines(ax, exits, color='g', linestyle='--')
    draw_vertical_lines(ax, rewards, color='r')
    ax.plot(timestamps_to_plot, DA_trace_to_plot, color='black')
    ax.axis('off')
    if return_handle:
        fig.show()
        return fig, ax


# --- helper functions ---
def draw_vertical_lines(ax, x_npy, ymin=0, ymax=1, color='r', alpha=1, linestyle='-', linewidth=1):
    for x_value in x_npy:
        ax.axvline(x_value, ymin=ymin, ymax=ymax, color=color, linestyle=linestyle, linewidth=linewidth)


# --- end of helper function
def main():
    # --- data preparation ---
    animal_str = 'SZ036'
    session_id = 11
    zscore = data_loader.load_session_dataframe(animal_str, session_id, 'zscore', file_format='parquet')
    pi_events = data_loader.load_session_dataframe(animal_str, session_id, 'pi_events_processed', file_format='parquet')
    # --- end of data preparation ---
    figc_example_1d_traces(zscore, pi_events)


if __name__ == '__main__':
    main()
