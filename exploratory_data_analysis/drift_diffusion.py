import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config
from figure_making.data_loader import load_session_dataframe


def calculate_dopamine_amplitudes(expreward_df, zscore_df, window=0.5):
    """
    Calculates the DA peak-to-trough amplitude (0-0.5s) for each reward.
    """
    amplitudes = []

    # Iterate through each reward event
    for _, row in expreward_df.iterrows():
        reward_time = row['reward_time']

        # Define the window: [reward_time, reward_time + 0.5s]
        # We look for the zscore data in that time range
        mask = (zscore_df['time_recording'] >= reward_time) & \
               (zscore_df['time_recording'] <= (reward_time + window))

        subset = zscore_df.loc[mask]

        if not subset.empty:
            # Assuming 'green_right'/'green_left' depending on side.
            # If side_relative is available, use it; otherwise, default to a column
            # Note: You may need logic here to switch between right/left green based on port
            da_signal = subset['green_right']  # Example: Change to match your specific hemisphere logic

            amp = da_signal.max() - da_signal.min()
            amplitudes.append(amp)
        else:
            amplitudes.append(np.nan)

    return amplitudes


def exponential_decay(t, amplitude, tau=2.0):
    """Simple exponential decay function."""
    return amplitude * np.exp(-t / tau)


def compute_weighted_cumulative_dopamine(trial_rewards, tau=2.0, time_res=0.01):
    """
    Computes the cumulative dopamine signal where each reward's contribution
    is weighted by its DA_amplitude.
    """
    # Define time window
    start = trial_rewards['entry_time'].min()
    end = trial_rewards['exit_time'].max()
    t = np.arange(start, end, time_res)

    cumulative_signal = np.zeros_like(t)

    for _, reward in trial_rewards.iterrows():
        # Skip if amplitude is NaN
        if pd.isna(reward['DA']):
            continue

        # Mask for times after this specific reward
        mask = t >= reward['reward_time']
        # Add weighted decay: amplitude * exp(-dt/tau)
        cumulative_signal[mask] += exponential_decay(
            t[mask] - reward['reward_time'],
            reward['DA'],
            tau
        )

    return t, cumulative_signal


def plot_session_trials(session_df, ax, trial_id):
    """Plots the cumulative curve for a single trial, colored by block."""
    trial_data = session_df[session_df['trial'] == trial_id]
    if trial_data.empty:
        return

    # Define colors based on block
    # 0.4 -> Set2[0], 0.8 -> Set2[1]
    block_val = trial_data['block'].iloc[0]
    color = sns.color_palette('Set2')[0] if block_val == '0.4' else sns.color_palette('Set2')[1]

    # Compute the cumulative curve
    t, weighted_cum = compute_weighted_cumulative_dopamine(trial_data)

    # Plot curve
    ax.plot(t - trial_data['entry_time'].min(), weighted_cum, color=color, linewidth=2)

    # Identify exit y-value
    exit_y = weighted_cum[np.argmin(np.abs(t - trial_data['exit_time'].max()))]

    # Add horizontal threshold line
    ax.axhline(exit_y, color=color, linestyle='--', alpha=0.6)

    ax.set_title(f"Trial {int(trial_id)} ({block_val} Blk)", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

def main():
    # 1. Load data
    animal = "SZ036"  # Example animal
    session_id = 11 # example session
    # Assuming expreward_df is stored under 'expreward' name
    df = load_session_dataframe(animal, 'expreward_df', session_id=session_id)

    # 2. Setup Plotting
    sns.set_context('paper', font_scale=1.5)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 3. Process a representative trial (e.g., Trial 5)
    sample_trial = df[(df['trial'] == 5)]
    t, cum_sig = compute_weighted_cumulative_dopamine(sample_trial)

    # 4. Visualization
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(t, cum_sig, color='purple', linewidth=2, label='Weighted Cumulative')
    plt.title(f"Weighted Cumulative Dopamine Response (Trial {sample_trial['trial'].iloc[0]})")
    plt.xlabel("Time in Session (s)")
    plt.ylabel("Cumulative Dopamine Signal")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    animal = "SZ036"
    session_id = 5
    zscore_df = load_session_dataframe(animal, 'zscore', session_id=session_id)
    expreward_df = load_session_dataframe(animal, 'expreward_df', session_id=session_id)
    expreward_df['DA'] = calculate_dopamine_amplitudes(expreward_df, zscore_df)
    # 2. Setup 20 Subplots
    fig, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    # 3. Iterate through first 20 trials
    trial_list = expreward_df['trial'].unique()[:20]

    for i, trial_id in enumerate(trial_list):
        plot_session_trials(expreward_df, axes[i], trial_id)

    plt.suptitle(f"Cumulative Dopamine Decay Models: {animal} Session {session_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # main()
    print("hello")
