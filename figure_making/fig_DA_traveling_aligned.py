import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import data_loader
mpl.rcParams['figure.dpi'] = 300


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


def create_broken_axis_plots(df):
    groups = df[['animal', 'hemisphere']].drop_duplicates().values
    if len(groups) == 0:
        print("No valid animal/hemisphere combinations found.")
        return

    # --- PLOTTING config ---
    colors = {'0.4': sns.color_palette('Set2')[0], '0.8': sns.color_palette('Set2')[1]}
    wspace_inbetween = 0.2
    panel_titles = [
        "Exit\nContext",
        "Enter\nInvestment",
        "Exit\nInvestment",
        "Enter\nContext"
    ]

    for (animal, hemi) in groups:
        print(f"Generating plot for: {animal} - {hemi} ...")
        subset = df[(df['animal'] == animal) & (df['hemisphere'] == hemi)]
        if subset.empty: continue
        data_pieces = [[], [], [], []]
        trials = subset['trial'].dropna().unique()

        for trial_id in trials:
            trial_df = subset[subset['trial'] == trial_id]
            next_trial_df = subset[subset['trial'] == trial_id + 1]
            if trial_df.empty: continue

            # --- TIMESTAMPS ---

            bg_rows = trial_df[trial_df['port'] == 'bg']
            exp_rows = trial_df[trial_df['port'] == 'exp']

            if bg_rows.empty or exp_rows.empty: continue

            t_exit_bg = bg_rows['time_recording'].max()
            t_enter_exp = exp_rows['time_recording'].min()
            t_exit_exp = exp_rows['time_recording'].max()

            if next_trial_df.empty:
                t_enter_bg_next = None
            else:
                next_bg_rows = next_trial_df[next_trial_df['port'] == 'bg']
                if not next_bg_rows.empty:
                    t_enter_bg_next = next_bg_rows['time_recording'].min()
                else:
                    t_enter_bg_next = None

            # --- DATA EXTRACTION (Common for both modes) ---

            # PIECE 1: Aligned to Exit [-1s, +1s]
            # (Last 1s of Port + First 1s of Travel)
            mask1 = (trial_df['time_recording'] >= t_exit_bg - 1) & \
                    (trial_df['time_recording'] <= t_exit_bg + 1)
            df1 = trial_df[mask1].copy()
            if not df1.empty:
                df1['aligned'] = df1['time_recording'] - t_exit_bg
                cond_left = (df1['aligned'] <= 0) & (df1['port'] == 'bg')
                cond_right = (df1['aligned'] >= 0) & (df1['port'].isnull())
                df1 = df1[cond_left | cond_right]
                if not df1.empty: data_pieces[0].append(df1[['aligned', 'DA', 'phase']])

            # PIECE 2: Aligned to Entry [-1s, +1s]
            # (Last 1s of Travel + First 1s of Next Port)
            mask2 = (trial_df['time_recording'] >= t_enter_exp - 1) & \
                    (trial_df['time_recording'] <= t_enter_exp + 1)
            df2 = trial_df[mask2].copy()
            if not df2.empty:
                df2['aligned'] = df2['time_recording'] - t_enter_exp
                cond_left = (df2['aligned'] <= 0) & (df2['port'].isnull())
                cond_right = (df2['aligned'] >= 0) & (df2['port'] == 'exp')
                df2 = df2[cond_left | cond_right]
                if not df2.empty: data_pieces[1].append(df2[['aligned', 'DA', 'phase']])

            # Piece 3: Exit Exp [-1s, +1s]
            mask3 = (trial_df['time_recording'] >= t_exit_exp - 1) & \
                    (trial_df['time_recording'] <= t_exit_exp + 1)
            df3 = trial_df[mask3].copy()
            if not df3.empty:
                df3['aligned'] = df3['time_recording'] - t_exit_exp
                cond_left = (df3['aligned'] <= 0) & (df3['port'] == 'exp')
                cond_right = (df3['aligned'] >= 0) & (df3['port'].isnull())
                df3 = df3[cond_left | cond_right]
                if not df3.empty: data_pieces[2].append(df3[['aligned', 'DA', 'phase']])

            # Piece 4: Enter BG (Next Trial) [-1s, +1s]
            if t_enter_bg_next is not None:
                combined = pd.concat([trial_df, next_trial_df])
                mask4 = (combined['time_recording'] >= t_enter_bg_next - 1) & \
                        (combined['time_recording'] <= t_enter_bg_next + 1)
                df4 = combined[mask4].copy()
                if not df4.empty:
                    df4['aligned'] = df4['time_recording'] - t_enter_bg_next
                    cond_left = (df4['aligned'] <= 0) & (df4['port'].isnull())
                    cond_right = (df4['aligned'] >= 0) & (df4['port'] == 'bg')
                    df4 = df4[cond_left | cond_right]
                    if not df4.empty: data_pieces[3].append(df4[['aligned', 'DA', 'phase']])

        # --- CALCULATE GLOBAL Y-LIMITS ---
        all_chunks = data_pieces[0] + data_pieces[1] + data_pieces[2] + data_pieces[3]
        if all_chunks:
            full_series = pd.concat(all_chunks)['DA']
            y_min, y_max = full_series.quantile(0.05), full_series.quantile(0.95)
            y_range = y_max - y_min
            if y_range == 0: y_range = 1
            ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        else:
            ylim = (-1, 1)

        # 2 Rows (Signals), 2 Columns (Exit vs Enter)
        # sharey='row': Scales match vertically
        # width_ratios=[1, 1]: Equal width since both are 2 seconds long
        fig, axes = plt.subplots(
            nrows=1, ncols=4,
            figsize=(16, 4),
            gridspec_kw={'wspace': wspace_inbetween, 'width_ratios': [1, 1, 1, 1]}
        )

        for col_idx in range(4):
            ax = axes[col_idx]
            ax.set_ylim(ylim)
            ax.set_facecolor('none')  # Transparent base so we can add colored patches

            # Get Data
            if data_pieces[col_idx]:
                df_plot = pd.concat(data_pieces[col_idx])
                df_plot['time_bin'] = df_plot['aligned'].round(1)

                for phase in ['0.4', '0.8']:
                    phase_sub = df_plot[df_plot['phase'].astype(str) == phase]
                    if not phase_sub.empty:
                        sns.lineplot(
                            data=phase_sub, x='time_bin', y='DA',
                            color=colors[phase], ax=ax, label=phase, errorbar='se'
                        )

            # --- STYLING & PATCHES ---
            # Handle X-Axis limits
            if col_idx in [0,2]:
                ax.set_xlim([-1, 0.5])
            elif col_idx in [1,3]:
                ax.set_xlim([-0.5, 1])
            ax.set_xlabel("Time (s)")

            # Clean Spines (Top is always gone)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(True)

            # Handle Y-Axis Labels (Only on far left)
            if col_idx == 0:
                ax.set_ylabel(f"{animal} {hemi}\nDA (Z-Score)")
                ax.tick_params(axis='y', left=True, labelleft=True)
                ax.legend(loc='upper left', frameon=False, fontsize=9)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', left=False, labelleft=False)
                if ax.get_legend(): ax.get_legend().remove()

            # Right Spines and Break Marks
            if col_idx == 3:
                ax.spines['right'].set_visible(False)
            else:
                ax.spines['right'].set_visible(True)
                draw_break_marks(ax, d=wspace_inbetween/2 + 0.02, x_gap=wspace_inbetween/2)

            # --- COLORED BACKGROUND PATCHES ---
            # Panel 0: Exit BG (Left half is BG) -> Skyblue on x < 0
            if col_idx == 0:
                ax.axvspan(-1, 0, color='skyblue', alpha=0.2, lw=0)
                ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(0, ylim[1], panel_titles[col_idx], color='red',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Panel 1: Enter Exp (Right half is Exp) -> Coral on x > 0
            elif col_idx == 1:
                ax.axvspan(0, 1, color='coral', alpha=0.2, lw=0)
                ax.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(0, ylim[1], panel_titles[col_idx], color='green',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Panel 2: Exit Exp (Left half is Exp) -> Coral on x < 0
            elif col_idx == 2:
                ax.axvspan(-1, 0, color='coral', alpha=0.2, lw=0)
                ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(0, ylim[1], panel_titles[col_idx], color='red',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Panel 3: Enter BG (Right half is BG) -> Skyblue on x > 0
            elif col_idx == 3:
                ax.axvspan(0, 1, color='skyblue', alpha=0.2, lw=0)
                ax.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(0, ylim[1], panel_titles[col_idx], color='green',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle(f"{animal} - {hemi}", y=0.98)
        plt.tight_layout()
        plt.show()

def main():
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    # animal_ids=["SZ037"]
    master_df1 = data_loader.load_dataframes_for_animal_summary(animal_ids, 'zscore',
                                                                day_0='2023-11-30',
                                                                melt=1,
                                                                hemisphere_qc=1,
                                                                file_format='parquet')

    animal_ids = ["RK007", "RK008"]
    master_df2 = data_loader.load_dataframes_for_animal_summary(animal_ids, 'zscore',
                                                                day_0='2025-06-17',
                                                                melt=1,
                                                                hemisphere_qc=1,
                                                                file_format='parquet')
    master_df = pd.concat([master_df1, master_df2], ignore_index=True)
    create_broken_axis_plots(master_df)

if __name__ == '__main__':
    main()