import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import data_loader


def plot_nri_iri_heatmap(animal_ids, day_0='2023-11-30'):
    """
    Creates a 2D heatmap of Mean DA Peak Amplitude with NRI (x-axis) and IRI (y-axis).
    Ticks are placed at the boundaries of 1-second bins.

    Args:
        animal_ids (list): List of animal ID strings (e.g., ['SZ036'] or multiple).
        day_0 (str): The reference start date.
    """
    # 1. Load data using the project's data loader
    df = data_loader.load_dataframes_for_animal_summary(
        animal_ids=animal_ids,
        df_name='DA_vs_features',
        day_0=day_0,
        file_format='parquet',
        hemisphere_qc=1
    )

    if df.empty:
        print("Error: No data loaded. Check animal IDs or data paths.")
        return



    # 2. Discretize NRI and IRI into 1-second bins (e.g., 2.5 becomes 2.0)
    df['NRI_bin'] = np.floor(df['NRI'])
    df['IRI_bin'] = np.floor(df['IRI'])

    # 3. Filter data limits: NRI (0 to 15s) and IRI (0 to 7s)
    # 15 bins for NRI (0-14), 7 bins for IRI (0-6)
    max_nri = 14
    max_iri = 7

    df_filtered = df[(df['NRI_bin'] >= 0) & (df['NRI_bin'] < max_nri) &
                     (df['IRI_bin'] >= 0) & (df['IRI_bin'] < max_iri)]

    # Normalization
    df_filtered['DA_normalized'] = df_filtered.groupby(['animal', 'session'])['DA'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )

    # 4. Create Pivot Table for the Grid
    heatmap_data = df_filtered.pivot_table(
        values='DA_normalized',
        index='IRI_bin',
        columns='NRI_bin',
        aggfunc='mean'
    )

    # Reindex to ensure all bins are present in the matrix, even if some have no data
    # This prevents the grid from skipping numbers on the axes
    heatmap_data = heatmap_data.reindex(
        index=np.arange(max_iri),
        columns=np.arange(max_nri)
    )

    # Count table to determine where to place asterisks
    count_data = df_filtered.pivot_table(
        values='DA',
        index='IRI_bin',
        columns='NRI_bin',
        aggfunc='count'
    )

    # 5. Plotting using pcolormesh (ideal for edge-aligned boundaries)
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the coordinate boundaries for the grid
    X, Y = np.meshgrid(np.arange(max_nri + 1), np.arange(max_iri + 1))

    # Plot the meshgrid (heatmap_data.values has shape [7, 15])
    mesh = ax.pcolormesh(X, Y, heatmap_data.values, cmap="viridis", shading='flat')

    # Add Colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Mean Normalized DA Amplitude')

    for iri_idx in range(max_iri):
        for nri_idx in range(max_nri):
            count = count_data.iloc[iri_idx, nri_idx]
            marker = ''
            if count < 10:
                marker = '*'
            elif 10 <= count < 30:
                marker = '**'

            if marker:
                # Add text to the center of the corresponding cell
                # We use white text so it stands out against most heatmap backgrounds
                ax.text(nri_idx + 0.5, iri_idx + 0.5, marker,
                        color='white', ha='center', va='center', fontsize=16, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Explicitly set ticks at the integer boundaries defined by max_nri and max_iri
    ax.set_xticks(np.arange(max_nri + 1))
    ax.set_yticks(np.arange(max_iri + 1))

    # Labels and Titles
    animal_label = animal_ids[0] if len(animal_ids) == 1 else f"All Animals (n={len(animal_ids)})"
    # plt.title(f"Dopamine Peak Amplitude by NRI and IRI\n{animal_label}")
    plt.xlabel("Reward Time from Entry (second)")
    plt.ylabel("IRI (second)")
    plt.tight_layout()


    # 6. Dynamic Saving Logic
    if len(animal_ids) == 1:
        # Save to single animal directory
        animal = animal_ids[0]
        save_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.FIGURE_SUBDIR)
        filename = f"{animal}_NRI_vs_IRI_heatmap.png"
    else:
        # Save to thesis summary directory
        save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
        filename = "Summary_NRI_vs_IRI_heatmap.png"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #
    # print(f"Heatmap successfully saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    # ==========================================
    # OPTION 1: Plot for a SINGLE animal
    # ==========================================
    # plot_nri_iri_heatmap(['RK008'])

    # ==========================================
    # OPTION 2: Plot for ALL animals
    # ==========================================
    all_animals = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    plot_nri_iri_heatmap(all_animals)