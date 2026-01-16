import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Project modules
import config
import data_loader
import fig_dopamine
import quality_control as qc


def setup_thesis_fig_4_2_layout():
    """
    Sets up the grid for the second thesis figure.
    Top row: Heatmap group (2/3 width)
    Bottom row: Population box plots (Full width)
    """
    # Width fixed at 12 as requested.
    # Height 10 provides enough vertical space for the two distinct rows.
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.3)

    # 1. Top Left: Heatmap Group (Bars, Heatmap, Colorbar)
    # We use columns 0:1 of the parent gs for the first group
    gs_heatmap_group = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0, 0],
        width_ratios=[0.5, 20, 1],
        wspace=0.05  # Keeps bars, heatmap, and cbar tight
    )

    axes_heatmap = [
        fig.add_subplot(gs_heatmap_group[0, 0]),  # Categorical bars
        fig.add_subplot(gs_heatmap_group[0, 1]),  # Heatmap
        fig.add_subplot(gs_heatmap_group[0, 2]),  # Colorbar
    ]

    # 2. Top Right: Mean Trace
    # We place this in column 1 of the parent gs.
    # The gap is now controlled by the parent gs wspace.
    ax_mean = fig.add_subplot(gs[0, 1])
    axes_heatmap.append(ax_mean)

    # 3. Bottom Row: Population Box Plot (Full width)
    ax_pop = fig.add_subplot(gs[1, :])

    return fig, axes_heatmap, ax_pop


def main():
    # --- Step 1: Data Preparation ---
    # Load session data for heatmap (Example: SZ036)
    animal_ex = 'SZ036'
    session_ex = '2023-12-30T19_57'
    hemi_ex = 'left'
    branch_ex = f'green_{hemi_ex}'

    zscore_heat = data_loader.load_session_dataframe(animal_ex, 'zscore', session_long_name=session_ex)
    reward_df_heat = data_loader.load_session_dataframe(animal_ex, 'expreward_df', session_long_name=session_ex)

    # Load population data for population box plots
    animal_ids_s = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    df_sz = data_loader.load_dataframes_for_animal_summary(animal_ids_s, 'DA_vs_features', day_0='2023-11-30',
                                                          hemisphere_qc=1)

    animal_ids_rk = ["RK007", "RK008"]
    df_rk = data_loader.load_dataframes_for_animal_summary(animal_ids_rk, 'DA_vs_features', day_0='2025-06-17',
                                                           hemisphere_qc=1)

    master_DA_features_df = pd.concat([df_sz, df_rk], ignore_index=True)


    # --- Step 2: Plotting ---
    fig, axes_heatmap, ax_pop = setup_thesis_fig_4_2_layout()

    # Top Row: Heatmap and mean traces split by Entry-Reward Interval (NRI)
    fig_dopamine.figc_example_session_heatmap_split_by_NRI(
        zscore_heat, branch_ex, reward_df_heat,
        animal=animal_ex, hemi=hemi_ex, session=session_ex,
        axes=axes_heatmap
    )

    # Bottom Row: Population summary of peak amplitudes binned by NRI
    fig_dopamine.fige_DA_vs_NRI_v2(master_DA_features_df, dodge=True, axes=ax_pop)

    # Adding Lettering for thesis formatting
    # lettering = 'abc'
    # for i, ax in enumerate([axes_heatmap[0], ax_pop]):
    #     if i == 0: scaling_factor = 100
    #     else: scaling_factor = 1
    #     ax.text(-0.05*scaling_factor, 1, lettering[i], transform=ax.transAxes, fontsize=16, weight='bold', va='bottom')
    fig.text(0.1, 0.88, 'a', fontsize=16, weight='bold', va='bottom')
    fig.text(0.1, 0.45, 'b', fontsize=16, weight='bold', va='bottom')
    # --- Step 3: Save and Export ---
    thesis_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)

    save_path = os.path.join(thesis_dir, 'fig_4-2_DA_vs_NRI.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close(fig)
    # print(f"Thesis figure 2 successfully saved to: {save_path}")


if __name__ == '__main__':
    main()
