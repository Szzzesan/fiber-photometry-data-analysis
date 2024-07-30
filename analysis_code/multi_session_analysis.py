import os
import func
from OneSession import OneSession
import matplotlib.pyplot as plt
import numpy as np
import statistics
from L import L


def visualize_transient_consistency(session_obj_list, save=0, save_path=None):
    transient_occur_xsession_l = np.empty(len(session_obj_list))
    transient_occur_xsession_r = np.empty(len(session_obj_list))
    transient_midmag_xsession_l = np.empty(len(session_obj_list))
    transient_midmag_xsession_r = np.empty(len(session_obj_list))
    for i in range(len(session_obj_list)):
        transient_occur_xsession_l[i] = session_obj_list[i].transient_occur_l
        transient_occur_xsession_r[i] = session_obj_list[i].transient_occur_r
        if session_obj_list[i].transient_midmag_l is not None:
            transient_midmag_xsession_l[i] = session_obj_list[i].transient_midmag_l * 100
        if session_obj_list[i].transient_midmag_r is not None:
            transient_midmag_xsession_r[i] = session_obj_list[i].transient_midmag_r * 100
    session_selected_l = np.where(
        (transient_midmag_xsession_l >= np.nanpercentile(transient_midmag_xsession_l, 25) - 0.5) & (
                transient_midmag_xsession_l <= np.nanpercentile(transient_midmag_xsession_l,
                                                                75) + 0.8))
    session_selected_r = np.where(
        (transient_midmag_xsession_r >= np.nanpercentile(transient_midmag_xsession_r, 25) - 0.5) & (
                transient_midmag_xsession_r <= np.nanpercentile(transient_midmag_xsession_r,
                                                                75) + 0.8))
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(hspace=0)
    if sum(transient_occur_xsession_l == None) != len(transient_occur_xsession_l):
        ax[0].plot(transient_occur_xsession_l, label='left hemisphere')
        ax[1].plot(transient_midmag_xsession_l, label='left hemisphere')
    if sum(transient_occur_xsession_r == None) != len(transient_occur_xsession_r):
        ax[0].plot(transient_occur_xsession_r, label='right hemisphere')
        ax[1].plot(transient_midmag_xsession_r, label='right hemisphere')
    ax[1].axhspan(np.nanpercentile(transient_midmag_xsession_l, 25) - 0.5,
                  np.nanpercentile(transient_midmag_xsession_l, 75) + 0.8, color='b', alpha=0.2, zorder=-10)
    ax[1].axhspan(np.nanpercentile(transient_midmag_xsession_r, 25) - 0.5,
                  np.nanpercentile(transient_midmag_xsession_r, 75) + 0.8, color='orange', alpha=0.2, zorder=-10)
    ax[0].scatter(session_selected_l, transient_occur_xsession_l[session_selected_l], color='b', marker='*', zorder=10)
    ax[0].scatter(session_selected_r, transient_occur_xsession_r[session_selected_r], color='orange', marker='*',
                  zorder=10)
    ax[1].scatter(session_selected_l, transient_midmag_xsession_l[session_selected_l], color='b', marker='*', zorder=10)
    ax[1].scatter(session_selected_r, transient_midmag_xsession_r[session_selected_r], color='orange', marker='*',
                  zorder=10)
    ax[0].legend()
    ax[0].set_ylabel('Transient Occurance Rate (/min)', fontsize=15)
    ax[0].set_ylim([0, 25])
    ax[1].set_ylabel('Median Transient Magnitude (%)', fontsize=15)
    ax[1].set_ylim([0, 15])
    ax[1].set_xlabel('Session', fontsize=20)
    fig.suptitle(f'{session_obj_list.animal}: Transient Consistency Analysis', fontsize=30)
    fig.show()
    if save:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print("A new directory is created!")
        fig_name = os.path.join(save_path, 'TransientConsistencyXSessions' + '.png')
        fig.savefig(fig_name)
    return session_selected_l, session_selected_r


def multi_session_analysis(animal_str, session_list, include_branch='both'):
    lab_dir = os.path.join('C:\\', 'Users', 'Shichen', 'OneDrive - Johns Hopkins', 'ShulerLab')
    animal_dir = os.path.join(lab_dir, 'TemporalDecisionMaking', 'imaging_during_task', animal_str)
    raw_dir = os.path.join(animal_dir, 'raw_data')
    FP_file_list = func.list_files_by_time(raw_dir, file_type='FP', print_names=0)
    behav_file_list = func.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    TTL_file_list = func.list_files_by_time(raw_dir, file_type='arduino', print_names=0)
    xsession_figure_export_dir = os.path.join(animal_dir, 'figures')
    # check if the neural data files, the behavior data files, and the sync data files are of the same numbers
    if (len(FP_file_list) == len(behav_file_list)) & (len(behav_file_list) == len(TTL_file_list)):
        # if so, make a list of objects. Each object is one session of data defined by class OneSession
        session_obj_list = L([None] * len(FP_file_list), animal=animal_str)
    else:
        print("Error: the numbers of different data files should be equal!!")
    for i in session_list:
        # try:
        session_obj_list[i] = OneSession(animal_str, i, include_branch=include_branch)
        # session_obj_list[i].examine_raw(save=1)
        session_obj_list[i].calculate_dFF0(plot=0, plot_middle_step=0, save=0)
        session_obj_list[i].process_behavior_data()
        session_obj_list[i].plot_heatmaps(save=1)
        # session_obj_list[i].actual_leave_vs_adjusted_optimal(save=1)
        # session_obj_list[i].extract_transient(plot_zscore=0)
         # session_obj_list[i].visualize_correlation_scatter(save=0)
        # except:
        #     print(f"skipped session {i} because of error!!!")
        #     print("----------------------------------")
        #     continue
    # ses_selected_l, ses_selected_r = visualize_transient_consistency(session_obj_list, save=1, save_path=xsession_figure_export_dir)
    return session_obj_list


if __name__ == '__main__':
    session_list = list(range(23))
    multi_session_analysis('SZ047', session_list, include_branch='only_right')
