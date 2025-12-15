import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figure_making import data_loader


def preprocess_data(df):
    df = df.copy()

    # 1. Sort by time to ensure the sequence is correct
    df = df.sort_values('time_recording')

    # 2. Forward Fill Trial and Phase
    # This takes the value from 'exp' (or 'bg') and carries it forward
    # into the NaN rows (the traveling periods) immediately following them.
    df['trial'] = df['trial'].ffill()
    df['phase'] = df['phase'].ffill()

    return df


def draw_break_marks(ax1, d=0.05, x_gap=0.1, y_top=0.9, y_gap=0.05):
    """
    Draws diagonal lines (break marks) with precise control over position.

    Parameters:
    - d: Half-length of the diagonal marker (size).
    - x_gap: How far into the gap to move the markers (closer together).
    - y_top: Vertical position of the top marker (1.0 is top spine, <1.0 is lower).
    - y_gap: How far the bottom marker is from the top marker
    """
    center_x = 1.0 + x_gap
    kwargs = dict(color='grey', clip_on=False, lw=1.5, zorder=10)
    kwargs['transform'] = ax1.transAxes
    ax1.plot((center_x - d, center_x + d), (y_top - d, y_top + d), **kwargs)  # Top-right diagonal
    ax1.plot((center_x - d, center_x + d), (y_top - y_gap - d, y_top - y_gap + d), **kwargs)  # Top-left diagonal


def create_broken_axis_plots(df, alignment_mode='bg_to_exp'):
    signals = ['green_left', 'green_right']

    # data_store[signal][0] = Exit Aligned Data
    # data_store[signal][1] = Enter Aligned Data
    data_store = {sig: [[], []] for sig in signals}

    # Define Labels and Logic based on mode
    if alignment_mode == 'bg_to_exp':
        line_labels = ["Exit\nContext", "Enter\nTime Investment"]
    else:  # exp_to_bg
        line_labels = ["Exit\nTime Investment", "Enter\nContext"]

    trials = df['trial'].dropna().unique()

    for trial_id in trials:
        trial_df = df[df['trial'] == trial_id]
        if trial_df.empty: continue

        # --- LOGIC 1: BG -> EXP (Same Trial) ---
        if alignment_mode == 'bg_to_exp':
            bg_rows = trial_df[trial_df['port'] == 'bg']
            exp_rows = trial_df[trial_df['port'] == 'exp']

            if bg_rows.empty or exp_rows.empty: continue

            t_exit = bg_rows['time_recording'].max()
            t_enter = exp_rows['time_recording'].min()

            df_exit_source = trial_df
            df_enter_source = trial_df

        # --- LOGIC 2: EXP -> BG (Cross Trial) ---
        elif alignment_mode == 'exp_to_bg':
            exp_rows = trial_df[trial_df['port'] == 'exp']
            next_trial_df = df[df['trial'] == trial_id + 1]
            if exp_rows.empty or next_trial_df.empty: continue
            bg_rows = next_trial_df[next_trial_df['port'] == 'bg']
            if bg_rows.empty: continue

            t_exit = exp_rows['time_recording'].max()
            t_enter = bg_rows['time_recording'].min()

            df_exit_source = trial_df
            df_enter_source = pd.concat([trial_df, next_trial_df])

        # --- DATA EXTRACTION (Common for both modes) ---

        # PIECE 1: Aligned to Exit [-1s, +1s]
        # (Last 1s of Port + First 1s of Travel)
        mask1 = (df_exit_source['time_recording'] >= t_exit - 1) & \
                (df_exit_source['time_recording'] <= t_exit + 1)
        df1 = df_exit_source[mask1].copy()
        df1['aligned'] = df1['time_recording'] - t_exit

        # PIECE 2: Aligned to Entry [-1s, +1s]
        # (Last 1s of Travel + First 1s of Next Port)
        mask2 = (df_enter_source['time_recording'] >= t_enter - 1) & \
                (df_enter_source['time_recording'] <= t_enter + 1)
        df2 = df_enter_source[mask2].copy()
        df2['aligned'] = df2['time_recording'] - t_enter

        # Store Data
        for sig in signals:
            cols = ['aligned', sig, 'phase']
            if not df1.empty: data_store[sig][0].append(df1[cols])
            if not df2.empty: data_store[sig][1].append(df2[cols])

    # --- PLOTTING ---
    colors = {'0.4': sns.color_palette('Set2')[0], '0.8': sns.color_palette('Set2')[1]}
    wspace_inbetween = 0.2

    # --- CALCULATE GLOBAL Y-LIMITS ---
    # Concatenate ALL data to find the absolute min and max
    all_values = []
    for sig in signals:
        for piece in [0, 1]:
            if data_store[sig][piece]:
                chunk = pd.concat(data_store[sig][piece])
                all_values.append(chunk[sig])

    if all_values:
        full_series = pd.concat(all_values)
        y_min, y_max = full_series.quantile(0.05), full_series.quantile(0.95)
        y_range = y_max - y_min
        global_ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    else:
        global_ylim = (-1, 1)

    # 2 Rows (Signals), 2 Columns (Exit vs Enter)
    # sharey='row': Scales match vertically
    # width_ratios=[1, 1]: Equal width since both are 2 seconds long
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(8, 8),
        gridspec_kw={'wspace': wspace_inbetween, 'width_ratios': [1, 1]}
    )

    for row_idx, sig in enumerate(signals):
        # Concatenate Data
        df_p1 = pd.concat(data_store[sig][0]) if data_store[sig][0] else pd.DataFrame()
        df_p2 = pd.concat(data_store[sig][1]) if data_store[sig][1] else pd.DataFrame()

        for col_idx, df_plot in enumerate([df_p1, df_p2]):
            ax = axes[row_idx, col_idx]
            ax.set_ylim(global_ylim)
            ax.set_facecolor('none')
            if not df_plot.empty:
                df_plot['time_bin'] = df_plot['aligned'].round(1)
                for phase in ['0.4', '0.8']:
                    subset = df_plot[df_plot['phase'].astype(str) == phase]
                    if not subset.empty:
                        sns.lineplot(
                            data=subset, x='time_bin', y=sig,
                            color=colors[phase], ax=ax, label=phase, errorbar='se'
                        )

            # --- STYLING ---
            # X-Axis limits: Strict -1 to +1
            ax.set_xlim(-1, 1)
            # ax.set_xlabel("Time (s)")

            # Y-Axis Label (Leftmost only)
            if col_idx == 0:
                ax.set_ylabel(f"{sig}\n(Z-Score)")
                ax.tick_params(axis='y', which='both', left=True, labelleft=True, right=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(True)
                if row_idx == 0:
                    ax.legend(loc='upper left')
                    ax.text(0, global_ylim[1], line_labels[0], color='red',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:
                    if ax.get_legend(): ax.get_legend().remove()
                ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

            else:
                ax.set_ylabel("")
                # Remove Y-axis ticks/labels for the right plot to make it look continuous
                ax.tick_params(axis='y', which='both', left=False, labelleft=False, right=False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(True)
                if ax.get_legend():
                    ax.get_legend().remove()
                # Green Dashed Line
                ax.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                if row_idx == 0:
                    ax.text(0, global_ylim[1], line_labels[1], color='green',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

            # X-Axis Label
            if row_idx == 1:
                ax.set_xlabel("Time (s)")
            elif row_idx == 0: ax.set_xlabel("")

        # --- DRAW BROKEN AXIS MARKERS ---
        draw_break_marks(axes[row_idx, 0], d=0.12, x_gap=wspace_inbetween/2, y_top=0.9, y_gap=0.05)

    plt.tight_layout()
    plt.show()


def create_aligned_plots(df, alignment_mode='bg_to_exp'):
    """
    alignment_mode: 'bg_to_exp' (Plot 1 instructions) or 'exp_to_bg' (Plot 2 instructions)
    """
    # --- STEP 1: GATHER DATA PIECES ---
    # We need to collect all data first to calculate proper X-axis durations
    # Structure: data_pieces[piece_index] = list of dataframes
    signals = ['green_left', 'green_right']
    labels = []

    # Dictionary to store all processed frames: data_store[signal][piece_idx]
    data_store = {sig: [[], [], [], []] for sig in signals}

    trials = df['trial'].dropna().unique()

    for trial_id in trials:
        trial_df = df[df['trial'] == trial_id]
        if trial_df.empty: continue

        # --- Logic for 'BG_to_EXP' ---
        if alignment_mode == 'bg_to_exp':
            bg_rows = trial_df[trial_df['port'] == 'bg']
            exp_rows = trial_df[trial_df['port'] == 'exp']

            if bg_rows.empty or exp_rows.empty: continue

            t_exit_bg = bg_rows['time_recording'].max()
            t_enter_exp = exp_rows['time_recording'].min()

            # Piece 1: Last 5s BG
            mask1 = (trial_df['port'] == 'bg') & \
                    (trial_df['time_recording'] >= t_exit_bg - 1)
            df1 = trial_df[mask1].copy()
            df1['aligned'] = df1['time_recording'] - t_exit_bg

            # Piece 2: Fist 1 sec BG -> Exp (Align: Exit BG)
            mask2 = (trial_df['port'].isnull()) & \
                    (trial_df['time_recording'] > t_exit_bg) & \
                    (trial_df['time_recording'] <= t_exit_bg + 1) & \
                    (trial_df['time_recording'] < t_enter_exp)
            df2 = trial_df[mask2].copy()
            df2['aligned'] = df2['time_recording'] - t_exit_bg
            # df2 = df2[df2['aligned'] >= -travel_limit]

            # Piece 3: Last 1 sec BG -> EXP (Align: Enter Exp)
            mask3 = (trial_df['port'].isnull()) & \
                    (trial_df['time_recording'] >= t_enter_exp - 1) & \
                    (trial_df['time_recording'] < t_enter_exp) & \
                    (trial_df['time_recording'] > t_exit_bg)
            df3 = trial_df[mask3].copy()
            df3['aligned'] = df3['time_recording'] - t_enter_exp

            # Piece 4: First 1 sec of EXP
            # It captures the None rows after exp exit
            mask4 = (trial_df['port'] == 'exp') & \
                    (trial_df['time_recording'] <= t_enter_exp + 1)
            df4 = trial_df[mask4].copy()
            df4['aligned'] = df4['time_recording'] - t_enter_exp
            # df4 = df4[df4['aligned'] <= travel_limit]

            labels = ["Last 1s Context", "First 1s Travel", "Last 1s Travel", "First 1s Time Investment"]

        elif alignment_mode == 'exp_to_bg':
            next_trial_df = df[df['trial'] == trial_id + 1]
            if trial_df.empty: continue
            exp_rows = trial_df[trial_df['port'] == 'exp']
            bg_rows = next_trial_df[next_trial_df['port'] == 'bg']
            if exp_rows.empty or bg_rows.empty: continue

            t_exit_exp = exp_rows['time_recording'].max()
            t_enter_bg = bg_rows['time_recording'].min()

            # last 1s of EXP
            mask1 = (trial_df['port'] == 'exp') & \
                    (trial_df['time_recording'] >= t_exit_exp - 1)
            df1 = trial_df[mask1].copy()
            df1['aligned'] = df1['time_recording'] - t_exit_exp

            # first 1s of Traveling from EXP -> BG
            mask2 = (trial_df['port'].isnull()) & \
                    (trial_df['time_recording'] > t_exit_exp) & \
                    (trial_df['time_recording'] <= t_exit_exp + 1) &\
                    (trial_df['time_recording'] < t_enter_bg)
            df2 = trial_df[mask2].copy()
            df2['aligned'] = df2['time_recording'] - t_exit_exp

            # last 1s of Traveling from EXP -> BG
            mask3 = (trial_df['port'].isnull()) & \
                    (trial_df['time_recording'] >= t_enter_bg - 1) & \
                    (trial_df['time_recording'] < t_enter_bg) & \
                    (trial_df['time_recording'] > t_exit_exp)
            df3 = trial_df[mask3].copy()
            df3['aligned'] = df3['time_recording'] - t_enter_bg

            # first 1s of BG in the next trial
            mask4 = (next_trial_df['port'] == 'bg') & \
                    (next_trial_df['time_recording'] <= t_enter_bg + 1)
            df4 = next_trial_df[mask4].copy()
            df4['aligned'] = df4['time_recording'] - t_enter_bg

            labels = ["Last 1s Time Investment", "First 1s Travel", "Last 1s Travel", "First 1s Context"]

        # Store chunks for both signals (left/right)
        for sig_idx, sig in enumerate(signals):
            cols = ['aligned', sig, 'phase']
            if not df1.empty: data_store[sig][0].append(df1[cols])
            if not df2.empty: data_store[sig][1].append(df2[cols])
            if not df3.empty: data_store[sig][2].append(df3[cols])
            if not df4.empty: data_store[sig][3].append(df4[cols])

    # --- STEP 2: PLOTTING ---
    colors = {'0.4': sns.color_palette('Set2')[0], '0.8': sns.color_palette('Set2')[1]}
    ratios = [1, 1, 1, 1]
    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(12, 6),  # Dynamic total width based on time
        sharey='row',
        gridspec_kw={'width_ratios': ratios}  # <--- THIS ENFORCES SAME SCALE
    )

    for row_idx, sig in enumerate(signals):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            chunks = data_store[sig][col_idx]

            if chunks:
                # Binning for smoother error bands
                df_plot = pd.concat(chunks)
                df_plot['time_bin'] = df_plot['aligned'].round(1)

                for phase in ['0.4', '0.8']:
                    subset = df_plot[df_plot['phase'].astype(str) == phase]
                    if not subset.empty:
                        sns.lineplot(
                            data=subset, x='time_bin', y=sig,
                            color=colors[phase], ax=ax, label=phase, errorbar='se'
                        )

            # --- FORMATTING (No Top/Right Spines) ---
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Titles and Labels
            ax.set_xlabel("Time (s)")
            if col_idx == 0:
                ax.set_ylabel(f"{sig}\n(Z-Score)")
                if row_idx == 0: ax.legend()
            else:
                ax.set_ylabel("")
                if ax.get_legend(): ax.get_legend().remove()

            if row_idx == 0:
                ax.set_title(labels[col_idx])

            ax.axvline(0, color='black', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- EXECUTION ---
    animal_str = 'SZ036'
    session_id = 5
    # session_long_name = '2024-01-11T16_25'
    df = data_loader.load_session_dataframe(animal_str, 'zscore', session_id=session_id, file_format='parquet')
    df = preprocess_data(df)
    print("Generating Plot 1: BG -> Exp Sequence")
    create_broken_axis_plots(df, alignment_mode='bg_to_exp')
    # create_aligned_plots(df, alignment_mode='bg_to_exp')

    print("\nGenerating Plot 2: Exp -> BG Sequence")
    create_broken_axis_plots(df, alignment_mode='exp_to_bg')
    # create_aligned_plots(df, alignment_mode='exp_to_bg')
