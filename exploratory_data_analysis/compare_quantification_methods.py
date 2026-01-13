import os
import glob
import random
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import config
from figure_making.data_loader import load_session_dataframe, load_dataframes_for_animal_summary


def calculate_auc_and_add_to_features(animal_id, session_long_name, max_search_sec=1.0):
    """
    Calculates AUC for the excitatory 'hump' only.
    Integrates from reward until the signal returns to baseline or a new reward arrives.
    """
    # 1. Load data products
    zscore = load_session_dataframe(animal_id, 'zscore', session_long_name=session_long_name)
    expreward_df = load_session_dataframe(animal_id, 'expreward_df', session_long_name=session_long_name)
    # features_df = load_session_dataframe(animal_id, 'DA_vs_features', session_long_name=session_long_name)

    if zscore is None or expreward_df is None:
        return None

    hemispheres = [col.replace('green_', '') for col in zscore.columns if col.startswith('green_')]
    auc_results = []

    for idx, row in expreward_df.iterrows():
        t_rew = row['reward_time']
        iri_post = row['IRI_post']  # Time to next reward
        block = row['block']
        trial = row['trial']
        t_next_rew = row['next_reward_time']

        # Hard limit for the search window to prevent drift interference
        limit = max_search_sec

        # Extract the search window for this specific reward
        search_df = zscore[(zscore['time_recording'] >= t_rew) &
                           (zscore['time_recording'] < t_rew + limit)].copy()
        search_df_for_amplitude = zscore[(zscore['time_recording'] >= t_rew) &
                           (zscore['time_recording'] < t_rew + 0.5) & (zscore['time_recording'] < t_next_rew)].copy()

        if not search_df.empty:
            for hemi in hemispheres:
                signal_col = f'green_{hemi}'
                if signal_col in search_df.columns:
                    # --- AUC ---
                    # Baseline-subtract using the value at the moment of reward delivery
                    baseline_val = search_df[signal_col].iloc[0:8].mean()
                    search_df['zeroed_signal'] = search_df[signal_col] - baseline_val

                    # --- Zero-Crossing Detection ---
                    # Find the first point after reward (skipping the first 250 ms (10 samples) for noise)
                    # where the signal drops back to or below the baseline.
                    below_baseline = search_df.iloc[20:][search_df['zeroed_signal'] <= 0]

                    if not below_baseline.empty:
                        # Signal returned to baseline; integrate only until this point
                        t_end = below_baseline['time_recording'].iloc[0]
                        integration_window = search_df[search_df['time_recording'] <= t_end]
                    else:
                        # Signal stayed above baseline until next reward or search limit
                        integration_window = search_df
                        t_end = search_df['time_recording'].iloc[-1]

                    # Calculate AUC using trapezoidal integration
                    auc_val = np.trapz(integration_window['zeroed_signal'],
                                       x=integration_window['time_recording'])

                    # --- amplitude ---
                    z_min = search_df_for_amplitude[signal_col].min()
                    z_max = search_df_for_amplitude[signal_col].max()
                    amplitude = z_max - z_min

                    auc_results.append({
                        'trial': trial,
                        'block': block,
                        'NRI': row['time_in_port'] if 'time_in_port' in row else np.nan,
                        'IRI': row['IRI_prior'] if 'IRI_prior' in row else row['IRI'],
                        'IRI_post': iri_post,
                        'hemisphere': hemi,
                        'AUC': auc_val,
                        'AMP': amplitude,
                        'auc_duration': t_end - t_rew  # Time the 'hump' lasted
                    })

    # Convert to DataFrame and prepare for merge
    auc_mapping_df = pd.DataFrame(auc_results)

    # 2. Merge back into the features dataframe
    # features_df['NRI_key'], features_df['IRI_key'] = features_df['NRI'].round(4), features_df['IRI'].round(4)
    # auc_mapping_df['NRI_key'], auc_mapping_df['IRI_key'] = auc_mapping_df['NRI'].round(4), auc_mapping_df['IRI'].round(
    #     4)

    # updated_features = pd.merge(
    #     features_df,
    #     auc_mapping_df[['NRI_key', 'IRI_key', 'hemisphere', 'AUC', 'IRI_post', 'auc_duration']],
    #     on=['NRI_key', 'IRI_key', 'hemisphere'],
    #     how='left'
    # ).drop(columns=['NRI_key', 'IRI_key'])

    return auc_mapping_df


def save_updated_features(df, animal_id, session_long_name):
    """Saves the updated DA_AUC_Amp_features dataframe to the processed data directory."""
    processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PROCESSED_DATA_SUBDIR)
    target_path = os.path.join(processed_dir, f"{animal_id}_{session_long_name}_DA_AUC_Amp_features.parquet")
    df.to_parquet(target_path)
    print(f"File updated and saved: {target_path}")


def compare_metrics(df):
    """
        Generates a comparison between peak amplitude (AMP) and excitatory AUC
        with a linear regression line and statistical annotation.
        """
    # Define plot limits based on quantiles to exclude extreme outliers
    y_lim_upper = df['AUC'].quantile(0.9995)
    x_lim_upper = df['AMP'].quantile(0.9995)

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    sns.set_context("talk")

    # 1. Plot the regression line and scatter points
    # regplot combines scatterplot and a linear regression fit
    sns.regplot(
        data=df,
        x='AMP',
        y='AUC',
        ax=axes,
        scatter_kws={'s': 2, 'alpha': 0.4},  # 's' controls the point size
        line_kws={'color': 'red', 'label': 'Linear Fit'}
    )

    # 2. Calculate Statistics (Senior Analyst Touch)
    # Drop NaNs to ensure regression calculation succeeds
    clean_df = df[['AMP', 'AUC']].dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_df['AMP'], clean_df['AUC'])

    # Annotate plot with R-squared and p-value
    stats_text = f"$R^2 = {r_value ** 2:.3f}$\n$p = {p_value:.2e}$"
    axes.text(
        0.05, 0.95, stats_text,
        transform=axes.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Formatting
    axes.set_title("DA Peak Amplitude vs. Excitatory AUC")
    axes.set_xlabel("Peak-to-Trough (z-score)")
    axes.set_ylabel("Hump AUC (z*s)")
    axes.set_xlim([0, x_lim_upper])
    axes.set_ylim([-1, y_lim_upper])

    plt.tight_layout()
    # Recommendation: Use savefig for thesis figures instead of plt.show()
    # plt.savefig("DA_Quantification_Comparison.png", dpi=300)
    plt.show()


def batch_process_and_save(animal_ids, random_plot=False):
    """
    Iterates through a list of animals, finds all sessions with DA_vs_features data,
    calculates the hump-only AUC, and overwrites the files.
    """
    for animal in animal_ids:
        print(f"\n--- Processing Animal: {animal} ---")

        # Define the path to the processed data directory for the animal
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)

        if not os.path.exists(processed_dir):
            print(f"Warning: Directory not found for {animal}: {processed_dir}")
            continue

        # Find all DA_vs_features files to determine which sessions are available
        search_pattern = os.path.join(processed_dir, f"{animal}_*_DA_vs_features.parquet")
        session_files = glob.glob(search_pattern)

        if not session_files:
            print(f"No DA_vs_features files found for {animal}.")
            continue

        for file_path in session_files:
            # Extract the session_long_name from the filename
            # Filename format: {animal}_{session_long_name}_DA_vs_features.parquet
            base_name = os.path.basename(file_path)
            # Remove animal prefix and the suffix to isolate session ID
            session_id = base_name.replace(f"{animal}_", "").replace("_DA_vs_features.parquet", "")

            print(f"Updating session: {session_id}")

            # 1. Calculate the new metrics (Hump AUC and IRI_post)
            # This uses the calculate_auc_and_add_to_features function we refined earlier
            updated_df = calculate_auc_and_add_to_features(animal, session_id)

            if updated_df is not None:
                # 2. Save/Overwrite the session dataframe
                save_updated_features(updated_df, animal, session_id)

                # 3. Optional: Generate comparison plots for spot-checking
                if random_plot:
                    if random.random() < 0.1:
                        print(f"Generating spot-check plot for {session_id}...")
                        compare_metrics(updated_df, animal, session_id)
            else:
                print(f"Skipping {session_id} due to processing error.")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    # animal_id = 'SZ042'
    # session_long_name = '2023-12-14T15_51'
    # updated_features = calculate_auc_and_add_to_features(animal_id, session_long_name, max_search_sec=2.0)

    # updated_features['auc_duration'].hist(), plt.show()
    # updated_features = updated_features[updated_features['IRI_post'] > 2]
    # compare_metrics(updated_features, animal_id, session_long_name)

    # target_animals = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    # batch_process_and_save(target_animals)

    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    master_df1 = load_dataframes_for_animal_summary(animal_ids, 'DA_AUC_Amp_features',
                                                                day_0='2023-11-30', hemisphere_qc=1,
                                                                file_format='parquet')

    animal_ids = ["RK007", "RK008"]
    master_df2 = load_dataframes_for_animal_summary(animal_ids, 'DA_AUC_Amp_features',
                                                                day_0='2025-06-17', hemisphere_qc=1,
                                                                file_format='parquet')
    df = pd.concat([master_df1, master_df2], ignore_index=True)
    df = df[df['IRI_post'] > 2]
    compare_metrics(df)
    print('hello')