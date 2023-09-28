import os
import func
from matplotlib import pyplot as plt
import pandas as pd
import sys


def single_session_analysis(animal_dir, signal_filename, arduino_filename, behav_filename):
    behav_dir = os.path.join(animal_dir, 'raw_data', behav_filename)
    signal_dir = os.path.join(animal_dir, 'raw_data', signal_filename)
    arduino_dir = os.path.join(animal_dir, 'raw_data', arduino_filename)
    fp_export_dir = os.path.join(animal_dir, 'processed_data', signal_dir[-23:-4])
    event_export_dir = os.path.join(animal_dir, 'processed_data', behav_dir[-23:-4])
    # print(os.path.exists(behav_dir))
    # print(os.path.exists(signal_dir))
    # print(os.path.exists(arduino_dir))
    print("Start analyzing" + " session " + signal_filename[-23:-7])
    pi_events, neural_events = func.data_read_sync(behav_dir, signal_dir, arduino_dir)

    # region Process neural data and export them
    # neural_events[neural_events.signal_type == 'actual'].green_right.to_numpy()
    func.check_framedrop(neural_events)
    raw_separated = func.de_interleave(neural_events, session_label=signal_dir[-23:-7], plot=0)
    dFF0 = func.calculate_dFF0(raw_separated, session_label=signal_dir[-23:-7], plot=0, plot_middle_steps=0)
    dFF0.name = 'dFF0'
    func.export_df_to_csv(dFF0, fp_export_dir)
    # endregion

    # region Process behavior data and export them
    pi_events["time_recording"] = pi_events['time'] - neural_events.timestamps[0]
    pi_events = func.data_reduction(pi_events, lick_tol=.01, head_tol=.2)
    # region Extract behavior events without trial structures
    non_trial_events = func.extract_behavior_events(pi_events)
    non_trial_events.name = "nontrial_event_sec"
    func.export_df_to_csv(non_trial_events, event_export_dir)
    # endregion

    # region Extract behavior events in regard to trial structures
    pi_trials = func.extract_trial(pi_events)
    # func.remove_unfocused_trials(pi_trials)
    exp_events = func.get_reward_by_trial(pi_trials, port=1)
    bg_events = func.get_reward_by_trial(pi_trials, port=2)
    exp_events = exp_events.add_prefix('exp_')
    bg_events = bg_events.add_prefix('bg_')
    all_trial_events = pd.concat([bg_events, exp_events], axis=1)
    exp_events.name = "exp_events"
    bg_events.name = "bg_events"
    all_trial_events.name = "trial_events_sec"
    func.export_df_to_csv(all_trial_events, event_export_dir)
    # endregion

    # endregion

    # # region Visualize with nex
    # func.visualize_with_nex(animal_dir, non_trial_events, all_trial_events, dFF0, fps=80, session_label=signal_filename[-23:-7])
    # # endregion
    print("Finish analyzing" + " session " + signal_filename[-23:-4])


if __name__ == '__main__':
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    animal_str = 'SZ033'
    animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
    raw_dir = os.path.join(animal_dir, 'raw_data')
    FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    single_session_analysis(animal_dir, FP_file_list[8], TTL_file_list[8], behav_file_list[8])
