import matplotlib.pyplot as plt
import pandas as pd
import os


def de_interleave(neural_events, session_label, save_path, plot='False', save='False'):
    green_right_actual = neural_events[neural_events.signal_type == 'actual'].green_right.to_numpy()
    green_right_isos = neural_events[neural_events.signal_type == 'isosbestic'].green_right.to_numpy()
    green_left_actual = neural_events[neural_events.signal_type == 'actual'].green_left.to_numpy()
    green_left_isos = neural_events[neural_events.signal_type == 'isosbestic'].green_left.to_numpy()
    time_raw = neural_events[neural_events.signal_type == 'actual'].timestamps
    time_recording = time_raw - neural_events.timestamps[0]
    time_recording = time_recording.to_numpy()
    raw_neural_deinterleaved = pd.DataFrame(
        data=[time_recording, green_right_actual, green_right_isos, green_left_actual,
              green_left_isos, time_raw.to_numpy()]).T
    raw_neural_deinterleaved.columns = ['time_recording', 'green_right_actual', 'green_right_isos', 'green_left_actual',
                                        'green_left_isos', 'time_raw']

    if plot:
        fig, ax = plt.subplots(1, figsize=(15, 10))
        num_color_site = int(len(raw_neural_deinterleaved.columns) / 2 - 1)
        for i in range(num_color_site):
            ax.plot(raw_neural_deinterleaved.iloc[:, 0], raw_neural_deinterleaved.iloc[:, 2 * i + 1],
                    label=raw_neural_deinterleaved.columns.values[2 * i + 1])
            ax.plot(raw_neural_deinterleaved.iloc[:, 0], raw_neural_deinterleaved.iloc[:, 2 * (i + 1)],
                    label=raw_neural_deinterleaved.columns.values[2 * (i + 1)])
        plt.legend()
        plt.title(session_label + ' raw deinterleaved')
        fig.show()

        if save:
            isExist = os.path.exists(save_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(save_path)
                print("A new directory is created!")
            fig_name = os.path.join(save_path, 'raw_deinterleaved' + '.png')
            fig.savefig(fig_name)

        # fig, ax = plt.subplots(3, figsize=(12, 9))
        # ax[0].plot(raw_neural_deinterleaved.time_recording / 60000, raw_neural_deinterleaved.green_right_actual,
        #            label='raw 470')
        # ax[1].plot(raw_neural_deinterleaved.time_recording / 60000, raw_neural_deinterleaved.green_right_isos,
        #            label='raw isos')
        # ax[0].plot(raw_neural_deinterleaved.time_recording / 60000,
        #            raw_neural_deinterleaved.green_right_actual.rolling(window=1000).mean(), label='470 rolling mean')
        # ax[1].plot(raw_neural_deinterleaved.time_recording / 60000,
        #            raw_neural_deinterleaved.green_right_isos.rolling(window=1000).mean(), label='isos rolling mean')
        # ax[2].plot(raw_neural_deinterleaved.time_recording / 60000,
        #            raw_neural_deinterleaved.green_right_actual.rolling(window=1000).std(), label='470 rolling std')
        # ax[2].plot(raw_neural_deinterleaved.time_recording / 60000,
        #            raw_neural_deinterleaved.green_right_isos.rolling(window=1000).std(), label='isos rolling std')
        # ax[2].set_xlabel('Time (min)')
        # for i in [0, 1, 2]:
        #     ax[i].legend()
        # fig.show()

    return raw_neural_deinterleaved
