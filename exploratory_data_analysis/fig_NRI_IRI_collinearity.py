import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config
from figure_making.data_loader import load_dataframes_for_animal_summary
import quality_control as qc


def plot_nri_by_iri_bins_violin():
    # 1. Define the animals based on your quality control dictionary
    # animal_ids = list(qc.qc_selections.keys())
    animal_ids=["RK008"]

    # 2. Load the concatenated data
    df = load_dataframes_for_animal_summary(
        animal_ids=animal_ids,
        df_name='DA_vs_features',
        day_0='2023-11-30',
        hemisphere_qc=1,
        file_format='parquet'
    )

    if df.empty:
        print("No data loaded. Check paths and data existence.")
        return


    # 3. Define the custom IRI bins and labels
    iri_bins = [1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.7, 3.6, np.inf]
    # Create clean string labels for the bins (e.g., '1-1.2', '1.2-1.4', ..., '>3.6')
    bin_labels = [f"{iri_bins[i]}-{iri_bins[i + 1]}" if iri_bins[i + 1] != np.inf else f">{iri_bins[i]}" for i in
                  range(len(iri_bins) - 1)]

    # 4. Apply the bins to the dataframe
    df['IRI_bin'] = pd.cut(df['IRI'], bins=iri_bins, labels=bin_labels, right=False)

    # Drop rows where IRI didn't fall into our bins (e.g., NaNs or values < 1)
    plot_df = df.dropna(subset=['IRI_bin', 'NRI'])

    # 5. Generate the Violin Plot
    sns.set_context("talk")
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot to cleanly show the density distribution
    sns.violinplot(
        data=plot_df,
        x='IRI_bin',
        y='NRI',
        color='lightgray',  # Neutral color since we aren't splitting by context yet
        inner='quartile',  # Displays the median and quartiles as dashed lines inside the violin
        linewidth=1.5,
        ax=ax
    )

    # 6. Formatting
    ax.set_xlabel('Inter-Reward Interval (s)', fontweight='bold')
    ax.set_ylabel('Reward Time from Entry / NRI (s)', fontweight='bold')
    ax.set_title('Distribution of NRI across IRI Bins', pad=15)
    sns.despine()
    plt.tight_layout()

    # 7. Save the figure to the multi-animal summary thesis folder
    # save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    # os.makedirs(save_dir, exist_ok=True)
    #
    # save_path = os.path.join(save_dir, 'NRI_distribution_by_IRI_bins_violin.png')
    # plt.savefig(save_path, dpi=300)
    # print(f"Figure saved successfully to: {save_path}")

    plt.show()



def analyze_collinearity_and_disentangle():
    # 1. Load the data
    animal_ids = ["SZ038"]
    df = load_dataframes_for_animal_summary(
        animal_ids=animal_ids,
        df_name='DA_vs_features',
        day_0='2023-11-30',
        hemisphere_qc=1,
        file_format='parquet'
    )

    # Drop rows with missing values in our target columns
    df = df.dropna(subset=['NRI', 'IRI', 'DA'])
    df = df[(df['IRI'] > 1)].copy()

    print("--- 1. Quantifying Collinearity ---")

    # Correlation
    corr = df[['NRI', 'IRI']].corr(method='spearman').iloc[0, 1]
    print(f"Spearman correlation between NRI and IRI: {corr:.3f}")

    # Calculate VIF
    # We need to add a constant (intercept) for statsmodels VIF calculation
    X = df[['NRI', 'IRI']]
    X_with_const = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]

    print("\nVariance Inflation Factors (VIF):")
    print(vif_data[vif_data['Variable'] != 'const'])
    print("(VIF > 5 indicates problematic collinearity)\n")

    print("--- 2. Disentangling via Orthogonalization (Residuals) ---")

    # Regress IRI on NRI to find the variance of IRI not explained by NRI
    # IRI = beta * NRI + intercept + error
    # The 'error' (residuals) is our orthogonalized IRI
    model_iri_on_nri = sm.OLS(df['IRI'], sm.add_constant(df['NRI'])).fit()
    df['IRI_residual'] = model_iri_on_nri.resid

    # Let's also do the reverse just in case you want to isolate NRI from IRI
    model_nri_on_iri = sm.OLS(df['NRI'], sm.add_constant(df['IRI'])).fit()
    df['NRI_residual'] = model_nri_on_iri.resid

    # 3. Visualization
    sns.set_context("talk")
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot A: DA vs Raw IRI (Confounded)
    sns.regplot(data=df, x='IRI', y='DA', ax=axes[0],
                scatter_kws={'alpha': 0.1, 'color': 'gray'}, line_kws={'color': 'red'})
    axes[0].set_title('DA Peak vs. Raw IRI\n(Confounded with NRI)')
    axes[0].set_xlabel('Raw IRI (s)')
    axes[0].set_ylabel('DA Peak Amplitude (z-score)')

    # Plot B: DA vs Residualized IRI (Disentangled)
    sns.regplot(data=df, x='IRI_residual', y='DA', ax=axes[1],
                scatter_kws={'alpha': 0.1, 'color': 'gray'}, line_kws={'color': 'blue'})
    axes[1].set_title('DA Peak vs. Residualized IRI\n(Independent of NRI)')
    axes[1].set_xlabel('IRI Residuals\n(Deviation from expected IRI given the time)')
    axes[1].set_ylabel('DA Peak Amplitude (z-score)')

    sns.despine()
    for ax in axes:
        ax.set_xlim(0, 5)
    plt.tight_layout()

    # Save the figure
    # save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, 'DA_vs_Orthogonalized_IRI.png')
    # plt.savefig(save_path, dpi=300)
    # print(f"\nFigure saved successfully to: {save_path}")

    plt.show()






if __name__ == "__main__":
    # plot_nri_by_iri_bins_violin()
    analyze_collinearity_and_disentangle()