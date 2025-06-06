import nex
import os
from helper.NexFileData import *
import helper.NexFileWriters
import helper.NexFileHeaders


def visualize_with_nex(animal_dir, all_events_df, behav_trial_df, dFF0_df, session_label, fps=80):
    nex_name = session_label + '.nex'
    fd = FileData()
    fd.Comment = "written by Python NexFileWriter"
    fd.TimestampFrequency = 1000
    # region Import trial_based events
    for i in range(len(behav_trial_df.columns)):
        fd.Events.append(Event(behav_trial_df.columns[i], behav_trial_df.iloc[:, i]))
    # endregion
    # region Import events without trial structure
    fd.Events.append(Event("complete_bg_entry", all_events_df.bg_entry))
    fd.Events.append(Event("complete_bg_exit", all_events_df.bg_exit))
    fd.Events.append(Event("complete_exp_entry", all_events_df.exp_entry))
    fd.Events.append(Event("complete_exp_exit", all_events_df.exp_exit))
    # endregion
    # region Import intervals
    fd.Intervals.append(Interval('bg_inport', behav_trial_df.bg_entry, behav_trial_df.bg_exit))
    fd.Intervals.append(Interval('bg_entry-2ndreward', behav_trial_df.bg_entry, behav_trial_df.bg_reward_2))
    fd.Intervals.append(Interval('bg_1st-3rdreward', behav_trial_df.bg_reward_1, behav_trial_df.bg_reward_3))
    fd.Intervals.append(Interval('bg_2nd-4threward', behav_trial_df.bg_reward_2, behav_trial_df.bg_reward_4))
    fd.Intervals.append(Interval('bg_3rd-exit', behav_trial_df.bg_reward_3, behav_trial_df.bg_exit))
    fd.Intervals.append(Interval('exp_inport', behav_trial_df.exp_entry, behav_trial_df.exp_exit))
    fd.Intervals.append(Interval('exp_entry-2ndreward', behav_trial_df.exp_entry, behav_trial_df.exp_reward_2))
    fd.Intervals.append(Interval('exp_1st-3rdreward', behav_trial_df.exp_reward_1, behav_trial_df.exp_reward_3))
    fd.Intervals.append(Interval('exp_2nd-4threward', behav_trial_df.exp_reward_2, behav_trial_df.exp_reward_4))
    fd.Intervals.append(Interval('exp_3rd-exit', behav_trial_df.exp_reward_3, behav_trial_df.exp_exit))
    # endregion
    # fd.Intervals.append(Interval('interval1', [40, 52], [43, 54]))
    # fd.Markers.append(Marker('marker1', [10, 12], ['f1', 'f2'], [['7', '8'], ['c', 'd']]))

    # region Import neural signals
    # for i in range(len(dFF0_df.columns) - 1):
    fd.Continuous.append(
        Continuous(dFF0_df.columns[1], fps, dFF0_df.iloc[:, 0], range(len(dFF0_df)), dFF0_df.iloc[:, 1]))
    fd.Continuous.append(
        Continuous(dFF0_df.columns[2], fps, dFF0_df.iloc[:, 0], range(len(dFF0_df)), dFF0_df.iloc[:, 2]))
    # endregion

        # fd.Continuous.append(Continuous('cont2', 10000, [5.1, 42], [0, 3], [127, 129, 22, 23]))
    # fd.Waveforms.append(Waveform('waveform1', 10000, [5.1, 42], 3, [127, 129, 22, 23, 99, 200]))

    writer = helper.NexFileWriters.NexFileWriter()
    path = os.path.join(animal_dir, 'nex_data', nex_name)
    writer.WriteDataToNexFile(fd, path)
