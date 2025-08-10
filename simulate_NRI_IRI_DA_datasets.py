import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


# --- Configuration ---
n_samples = 1000
# Small constant to add before taking a logarithm to avoid log(0) errors
epsilon = 1e-9

print(f"Generating {n_samples} data points for each scenario...")

# --- 1. Generate Predictor Variables (NRI and IRI) ---

# Generate NRI from an exponential distribution with a mean of 5 seconds.
# This naturally keeps most values within the 0-15s range.
nri = np.random.exponential(scale=5, size=n_samples)

# Generate IRI from an exponential distribution with a mean of 2 seconds.
# P(NRI > IRI) is naturally ~71% with these parameters, matching the condition
# that IRI is smaller than NRI for most observations.
iri = np.random.exponential(scale=2, size=n_samples)

# --- 2. Generate Dopamine for Each Scenario ---

# Scenario 1: Dopamine is completely driven by IRI
# DA = beta_IRI * log(IRI) + intercept + noise
intercept_1 = 8.0
beta_iri_1 = 5.0
noise_1 = np.random.normal(0, 2.0, size=n_samples)
dopamine_1 = intercept_1 + (beta_iri_1 * np.log(iri + epsilon)) + noise_1
df_scenario1 = pd.DataFrame({'NRI': nri, 'IRI': iri, 'DA': dopamine_1})


# Scenario 2: Dopamine is completely driven by NRI
# DA = beta_NRI * log(NRI) + intercept + noise
intercept_2 = 10.0
beta_nri_2 = 7.0
noise_2 = np.random.normal(0, 2.0, size=n_samples)
dopamine_2 = intercept_2 + (beta_nri_2 * np.log(nri + epsilon)) + noise_2
df_scenario2 = pd.DataFrame({'NRI': nri, 'IRI': iri, 'DA': dopamine_2})


# Scenario 3: Mostly NRI-Driven, a Little IRI-Driven
# DA = beta_NRI * log(NRI) + beta_IRI * log(IRI) + intercept + noise
intercept_3 = 9.0
beta_nri_3 = 7.0  # Stronger coefficient
beta_iri_3 = 3.5  # Weaker coefficient
noise_3 = np.random.normal(0, 2.0, size=n_samples)
dopamine_3 = (intercept_3 +
              (beta_nri_3 * np.log(nri + epsilon)) +
              (beta_iri_3 * np.log(iri + epsilon)) +
              noise_3)
df_scenario3 = pd.DataFrame({'NRI': nri, 'IRI': iri, 'DA': dopamine_3})

# plot DA vs. NRI in the three scenarios
def get_mean_sem_DA_for_feature(df, var='NRI', sample_per_bin=250):
    df_sorted = df.sort_values(by=var).reset_index(drop=True)
    total_sample = len(df_sorted)
    complete_bin = int(total_sample / sample_per_bin)
    bin_num = complete_bin
    if total_sample % sample_per_bin >= 20:
        bin_num += 1
    arr_bin_center = np.zeros(bin_num)
    arr_mean = np.zeros(bin_num)
    arr_sem = np.zeros(bin_num)
    for i in range(complete_bin):
        arr_bin_center[i] = (df_sorted.iloc[sample_per_bin * i][var] + df_sorted.iloc[sample_per_bin * (i + 1) - 1][
            var]) / 2
        arr_mean[i] = df_sorted.iloc[sample_per_bin * i:sample_per_bin * (i + 1)]['DA'].mean()
        arr_sem[i] = df_sorted.iloc[sample_per_bin * i:sample_per_bin * (i + 1)]['DA'].sem()
    if total_sample % sample_per_bin >= 20:
        arr_bin_center[-1] = (df_sorted.iloc[sample_per_bin * complete_bin][var] + df_sorted.iloc[-1][var]) / 2
        arr_mean[-1] = df_sorted.iloc[sample_per_bin * complete_bin:]['DA'].mean()
        arr_sem[-1] = df_sorted.iloc[sample_per_bin * complete_bin:]['DA'].sem()
    mean_sem_df = pd.DataFrame({'bin_center': arr_bin_center, 'mean': arr_mean, 'sem': arr_sem})
    return mean_sem_df

def figg_DA_vs_IRI_v2(master_df, IRI_group_size=20, axes=None):
    data = master_df[master_df['NRI'] >= master_df['IRI']]
    # data = master_df
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    # bins = [0, 0.8, 1.8, 2.9, 4.1, 5.5, 7.3, 9.6, np.inf]
    bins = [0, 2, 5, np.inf]
    bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    bin_labels[-1] = f'>{bins[-2]}'
    data['cat_code'] = pd.cut(data['NRI'], bins=bins, labels=bin_labels)
    data_plot = defaultdict(pd.DataFrame)
    cat_num = len(bin_labels)
    palette_to_use = list(sns.color_palette('Reds_r', n_colors=cat_num + 1))[:cat_num]
    for cat in bin_labels:
        subset = data[data['cat_code'] == cat]
        mean_sem = get_mean_sem_DA_for_feature(subset, var='IRI', sample_per_bin=IRI_group_size)
        data_plot[cat] = mean_sem
    for i, cat in enumerate(bin_labels):
        x = data_plot[cat]['bin_center']
        y = data_plot[cat]['mean']
        y_err = data_plot[cat]['sem']
        line = axes.plot(x, y, linewidth=0.5, label=cat, color=palette_to_use[i])
        axes.fill_between(x, y - y_err, y + y_err, color=line[0].get_color(), alpha=0.4, edgecolor='none')
    # axes.set_ylim([1.6, 4.1])
    # axes.set_title('DA vs. IRI Split by NRI Groups')
    axes.set_xlabel('Reward Time from Prior Reward (s)')
    axes.set_ylabel('Peak Amplitude', y=0.3)
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.legend(title='Reward Time from Entry (s)',
                title_fontsize='x-small',
                fontsize='x-small',
                ncol=3)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes

figg_DA_vs_IRI_v2(df_scenario1, IRI_group_size=40)
figg_DA_vs_IRI_v2(df_scenario2, IRI_group_size=40)
figg_DA_vs_IRI_v2(df_scenario3, IRI_group_size=40)