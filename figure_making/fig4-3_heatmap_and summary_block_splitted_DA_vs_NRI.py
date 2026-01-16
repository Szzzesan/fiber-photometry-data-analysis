import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Project modules
import config
import data_loader
import fig_dopamine


def setup_thesis_fig_4_3_layout():
    """
    Sets up the grid for the third thesis figure.
    Top row: Heatmap group split by block (context).
    Bottom row: Population box plots comparing blocks across NRI bins.
    """
    # Maintain width at 12 and height at 10 as per previous figure
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
        fig.add_subplot(gs_heatmap_group[0, 0]),  # Categorical bars (Low vs High block)
        fig.add_subplot(gs_heatmap_group[0, 1]),  # Heatmap
        fig.add_subplot(gs_heatmap_group[0, 2]),  # Colorbar
    ]

    # 2. Top Right: Mean Trace
    # Separated from the heatmap via the parent grid's wspace
    ax_mean = fig.add_subplot(gs[0, 1])
    axes_heatmap.append(ax_mean)

    # 3. Bottom Row: Population Box Plot (Full width)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, :],
        width_ratios=[5, 1],
        wspace=0.1
    )

    ax_pop = fig.add_subplot(gs_bottom[0, 0])
    ax_summary = fig.add_subplot(gs_bottom[0, 1])

    return fig, axes_heatmap, ax_pop, ax_summary


def main():
    # --- Step 1: Data Preparation ---
    # Example session for block-split heatmap (SZ042 provides a good contrast)
    animal_ex = 'SZ042'
    session_ex = '2023-12-11T21_06'
    hemi_ex = 'left'
    branch_ex = f'green_{hemi_ex}'

    zscore_heat = data_loader.load_session_dataframe(animal_ex, 'zscore', session_long_name=session_ex)
    reward_df_heat = data_loader.load_session_dataframe(animal_ex, 'expreward_df', session_long_name=session_ex)

    # Load concatenated population data
    animal_ids_sz = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    df_sz = data_loader.load_dataframes_for_animal_summary(animal_ids_sz, 'DA_vs_features',
                                                          day_0='2023-11-30', hemisphere_qc=1)

    animal_ids_rk = ["RK007", "RK008"]
    df_rk = data_loader.load_dataframes_for_animal_summary(animal_ids_rk, 'DA_vs_features',
                                                          day_0='2025-06-17', hemisphere_qc=1)

    master_DA_features_df = pd.concat([df_sz, df_rk], ignore_index=True)

    # --- Step 2: Plotting ---
    fig, axes_heatmap, ax_pop, ax_summary = setup_thesis_fig_4_3_layout()

    # Top Row: Example session heatmap split by Context Reward Rate (Block)
    fig_dopamine.figd_example_session_heatmap_split_by_block(
        zscore_heat, branch_ex, reward_df_heat,
        animal=animal_ex, hemi=hemi_ex, session=session_ex,
        axes=axes_heatmap
    )

    # Bottom Row: Population summary of peak amplitudes split by block across NRI bins
    fig_dopamine.figf_DA_vs_NRI_block_split_v2(master_DA_features_df, axes=ax_pop)
    fig_dopamine.figf_summary_block_split(master_DA_features_df, axes=ax_summary)

    # Standardized lettering for thesis figures
    fig.text(0.1, 0.88, 'a', fontsize=16, weight='bold', va='bottom')
    fig.text(0.1, 0.45, 'b', fontsize=16, weight='bold', va='bottom')


    # --- Step 3: Save and Export ---
    thesis_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)

    save_path = os.path.join(thesis_dir, 'fig_4-3_DA_vs_NRI_block_split.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Thesis figure 4-3 successfully saved to: {save_path}")
    plt.show()



if __name__ == '__main__':
    main()