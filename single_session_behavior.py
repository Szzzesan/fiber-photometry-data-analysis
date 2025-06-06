import pandas as pd
import helper
import os


def single_session_behavior(pi_dir):
    pi_events = pd.read_csv(pi_dir, na_values=['None'], skiprows=3)
    pi_events["time_recording"] = pi_events['session_time'] * 1000
    # region Extract behavior events without trial structures
    non_trial_events = helper.extract_behavior_events(pi_events)
    # endregion

    # region Extract behavior events in regard to trial structures
    pi_trials = helper.extract_trial(pi_events, session_label=behav_dir[-23:-7], plot=0)
    helper.plot_behav_events(pi_events)
    print('Hello')
    # endregion

if __name__ == '__main__':
    animal_name = 'SZ033'
    session_filename = 'data_2023-07-25_17-09-44.txt'
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    behav_dir = os.path.join(lab_dir, 'behavior_code', 'data', animal_name, session_filename)
    single_session_behavior(behav_dir)