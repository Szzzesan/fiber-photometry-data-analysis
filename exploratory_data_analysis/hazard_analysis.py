import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from scipy.stats import zscore
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# from pymer4.models import Lmer

# Import from your repository structure
import config
from figure_making import data_loader
import quality_control as qc

# Define our standard colors
COLOR_LOW = sns.color_palette('Set2')[0]
COLOR_HIGH = sns.color_palette('Set2')[1]
BIN_PALETTE = sns.color_palette('Reds_r', n_colors=4)


def extract_survival_data(animal_ids):
    """
    Extracts two levels of survival data:
    1. Trial-level: Total patch residence time (for Time from Entry plots)
    2. Reward-level: Every single inter-reward interval (for Time from Prev Reward plots)
    """
    trial_records = []
    reward_records = []

    for animal in animal_ids:
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            continue

        session_files = [f for f in os.listdir(processed_dir) if f.endswith('trial_df.parquet')]

        for session_id in range(len(session_files)):
            trial_df = data_loader.load_session_dataframe(animal, 'trial_df', session_id=session_id)
            expreward_df = data_loader.load_session_dataframe(animal, 'expreward_df', session_id=session_id)

            if trial_df is None or expreward_df is None:
                continue

            valid_trials = trial_df[trial_df['is_valid_trial'] == True].copy()
            expreward_df['block'] = expreward_df['block'].astype(float)

            for _, trial in valid_trials.iterrows():
                trial_id = trial['trial']
                entry_t = trial['exp_entry']
                exit_t = trial['exp_exit']

                trial_rewards = expreward_df[expreward_df['trial'] == trial_id]
                if len(trial_rewards) == 0:
                    continue  # Skip unrewarded trials for this specific analysis

                block_val = trial_rewards['block'].iloc[0]

                # --- 1. TRIAL LEVEL DATA (For Time from Entry) ---
                total_patch_time = exit_t - entry_t
                final_give_up_time = exit_t - trial_rewards['reward_time'].max()

                if total_patch_time > 0:
                    trial_records.append({
                        'animal': animal,
                        'session_id': session_id,
                        'context': block_val,
                        'time_from_entry': total_patch_time,
                        'time_from_final_rew': final_give_up_time,  # Used for quartile grouping in Comb 3
                        'event_observed': 1  # Exit is always observed at trial level
                    })

                # --- 2. REWARD LEVEL DATA (For Time from Previous Reward) ---
                for _, rew in trial_rewards.iterrows():
                    rew_t = rew['reward_time']
                    next_rew_t = rew['next_reward_time']

                    # How deep into the trial are we starting this specific wait?
                    time_from_entry_at_reward = rew_t - entry_t

                    if pd.isna(next_rew_t):
                        # This is the final reward. The interval ends with an EXIT.
                        duration = exit_t - rew_t
                        event = 1  # Exit observed
                    else:
                        # Another reward came. The interval ends with a REWARD.
                        duration = next_rew_t - rew_t
                        event = 0  # Censored (animal survived the wait without leaving)

                    if duration > 0:
                        reward_records.append({
                            'animal': animal,
                            'session_id': session_id,
                            'context': block_val,
                            'time_from_entry_at_reward': time_from_entry_at_reward,  # Used for Comb 4
                            'time_from_prev_reward': duration,
                            'event_observed': event
                        })

    trial_df = pd.DataFrame(trial_records).dropna(subset=['context'])
    reward_df = pd.DataFrame(reward_records).dropna(subset=['context'])
    return trial_df, reward_df


def plot_sr(ax, df, duration_col, group_col, group_labels, colors, title, max_x=20):
    """
    Plots Survival Rate using lifelines.KaplanMeierFitter.
    """
    kmf = KaplanMeierFitter()

    for i, group in enumerate(group_labels):
        mask = df[group_col] == group
        if mask.sum() > 0:
            kmf.fit(df.loc[mask, duration_col], event_observed=df.loc[mask, 'event_observed'])
            kmf.plot_survival_function(ax=ax, color=colors[i], ci_show=True, label=str(group))

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"{duration_col.replace('_', ' ').title()} (s)")
    ax.set_ylabel("Survival Probability")
    sns.despine(ax=ax)
    ax.legend(frameon=True, loc='upper right')


def plot_discrete_hr(ax, df, duration_col, group_col, group_labels, colors, title, max_x=20, bin_width=0.2):
    """
    Plots Discrete Hazard Probability (Probability of exiting in the current bin,
    given survival up to this bin). Includes binomial standard error bands.
    """
    bins = np.arange(0, max_x + bin_width, bin_width)
    bin_centers = bins[:-1] + (bin_width / 2)

    for i, group in enumerate(group_labels):
        mask = df[group_col] == group
        if mask.sum() == 0:
            continue

        # Isolate the data for this specific group
        group_df = df[mask]

        # 1. Count ONLY actual exits in each bin
        exit_times = group_df[group_df['event_observed'] == 1][duration_col].values
        exits, _ = np.histogram(exit_times, bins=bins)

        # 2. Count ALL dropouts (both exits and censored) to calculate who is still at risk
        all_times = group_df[duration_col].values
        total_counts, _ = np.histogram(all_times, bins=bins)
        cumulative_dropouts = np.cumsum(total_counts)

        total_trials = len(all_times)
        at_risk = total_trials - np.insert(cumulative_dropouts[:-1], 0, 0)

        # 3. Calculate Discrete Hazard Probability
        hazard_prob = np.zeros_like(exits, dtype=float)
        se = np.zeros_like(exits, dtype=float)

        valid = at_risk > 0
        hazard_prob[valid] = exits[valid] / at_risk[valid]

        # 4. Calculate 90% Confidence Intervals (using 1.645 for Z-score)
        se[valid] = np.sqrt((hazard_prob[valid] * (1 - hazard_prob[valid])) / at_risk[valid])

        # Mask out bins with extremely low N (e.g., < 5) to prevent visually chaotic tails
        plot_mask = at_risk >= 5

        # Plot the main line
        ax.plot(bin_centers[plot_mask], hazard_prob[plot_mask], color=colors[i], label=str(group), lw=2)

        # Plot the confidence interval shading
        ax.fill_between(bin_centers[plot_mask],
                        np.maximum(0, hazard_prob[plot_mask] - 1.645 * se[plot_mask]),
                        np.minimum(1, hazard_prob[plot_mask] + 1.645 * se[plot_mask]),
                        color=colors[i], alpha=0.3)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, max_x)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"{duration_col.replace('_', ' ').title()} (s)")
    ax.set_ylabel("Discrete Hazard Probability")
    sns.despine(ax=ax)
    ax.legend(frameon=True, loc='upper left')


def generate_four_combinations(trial_df, reward_df, save_fig=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Pre-process bins for combinations 3 and 4
    bins_last_rew = [0, 2.5, 5, 7.5, 10]
    trial_df['grp_last_reward'] = pd.cut(trial_df['time_from_final_rew'], bins=bins_last_rew)
    labels_last_rew = trial_df['grp_last_reward'].dropna().unique().sort_values()

    bins_entry = [0, 3, 6, 9, 12]
    reward_df['grp_entry'] = pd.cut(reward_df['time_from_entry_at_reward'], bins=bins_entry)
    labels_entry = reward_df['grp_entry'].dropna().unique().sort_values()

    discrete_bin_size = 0.5

    # --- Combination 1: vs Time from Entry (TRIAL LEVEL) ---
    plot_sr(axes[0, 0], trial_df, 'time_from_entry', 'context', [0.4, 0.8], [COLOR_LOW, COLOR_HIGH],
            "1. SR vs Time from Entry (by Context)", max_x=20)
    plot_discrete_hr(axes[0, 1], trial_df, 'time_from_entry', 'context', [0.4, 0.8], [COLOR_LOW, COLOR_HIGH],
                     "1. HR vs Time from Entry (by Context)", max_x=20, bin_width=discrete_bin_size)

    # --- Combination 2: vs Time from Prev Reward (REWARD LEVEL) ---
    plot_sr(axes[1, 0], reward_df, 'time_from_prev_reward', 'context', [0.4, 0.8], [COLOR_LOW, COLOR_HIGH],
            "2. SR vs Time from Prev Reward (by Context)", max_x=10)
    plot_discrete_hr(axes[1, 1], reward_df, 'time_from_prev_reward', 'context', [0.4, 0.8], [COLOR_LOW, COLOR_HIGH],
                     "2. HR vs Time from Prev Reward (by Context)", max_x=10, bin_width=discrete_bin_size)

    # # --- Combination 3: vs Time from Entry (TRIAL LEVEL), by Final GUT ---
    # plot_sr(axes[2, 0], trial_df, 'time_from_entry', 'grp_last_reward', labels_last_rew, BIN_PALETTE,
    #         "3. SR vs Entry (by Final GUT Quartiles)", max_x=20)
    # plot_discrete_hr(axes[2, 1], trial_df, 'time_from_entry', 'grp_last_reward', labels_last_rew, BIN_PALETTE,
    #                  "3. HR vs Entry (by Final GUT Quartiles)", max_x=20, bin_width=discrete_bin_size)

    # --- Combination 4: vs Time from Prev Reward (REWARD LEVEL), by Entry Time ---
    plot_sr(axes[2, 0], reward_df, 'time_from_prev_reward', 'grp_entry', labels_entry, BIN_PALETTE,
            "4. SR vs Prev Reward (by Entry Time Quartiles)", max_x=10)
    plot_discrete_hr(axes[2, 1], reward_df, 'time_from_prev_reward', 'grp_entry', labels_entry, BIN_PALETTE,
                     "3. HR vs Prev Reward (by Entry Quartiles)", max_x=10, bin_width=discrete_bin_size)

    plt.tight_layout(pad=3.0)
    if save_fig:
        plt.savefig(
            os.path.join(config.MAIN_DATA_ROOT, config.PAPER_OUTPUT_SUBDIR, 'survival_discrete_hazard_censored.png'),
            dpi=300)
    plt.show()


def fit_cox_model(df, duration_col, event_col, cluster_col='animal'):
    """
    Prepares data and fits a Cox Proportional Hazards model.
    """
    # 1. Filter down to only the columns the model needs
    cols_to_keep = [duration_col, event_col, 'context', cluster_col]
    model_df = df[cols_to_keep].copy()

    # 2. Set 0.8 as the reference category
    # By creating a dummy variable where context 0.4 is 1 and 0.8 is 0,
    # the model treats 0.8 as the baseline. The resulting Hazard Ratio (HR)
    # will represent the hazard of 0.4 relative to 0.8.
    model_df['context_0.4'] = (model_df['context'] == 0.4).astype(int)
    model_df = model_df.drop('context', axis=1)

    # 3. Initialize and fit the model
    cph = CoxPHFitter()

    # The cluster_col parameter adjusts standard errors for repeated measures within animals
    cph.fit(model_df, duration_col=duration_col, event_col=event_col, cluster_col=cluster_col)

    return cph


def calculate_hr_per_animal(df, duration_col, event_col):
    """
    Fits a Cox PH model for each individual animal and extracts the Hazard Ratios.
    Reference context is set to 0.8.
    """
    results = []

    # Iterate over each unique animal in the dataset
    for animal in df['animal'].unique():
        animal_df = df[df['animal'] == animal].copy()

        # 1. Check for sufficient data: Need both contexts to make a comparison
        if len(animal_df['context'].unique()) < 2:
            print(f"Skipping {animal}: Missing one or both context conditions.")
            continue

        # 2. Set 0.8 as the reference by creating a dummy for 0.4
        animal_df['context_0.4'] = np.isclose(animal_df['context'], 0.4).astype(int)

        # Isolate the exact columns needed for the model
        model_df = animal_df[[duration_col, event_col, 'context_0.4']]

        cph = CoxPHFitter()

        try:
            # 3. Fit the model on just this animal's data
            cph.fit(model_df, duration_col=duration_col, event_col=event_col)

            # 4. Extract metrics from the fitted model
            hr = cph.hazard_ratios_['context_0.4']
            p_val = cph.summary.loc['context_0.4', 'p']
            ci_lower = cph.summary.loc['context_0.4', 'exp(coef) lower 95%']
            ci_upper = cph.summary.loc['context_0.4', 'exp(coef) upper 95%']

            results.append({
                'animal': animal,
                'HR_0.4_vs_0.8': hr,
                'p_value': p_val,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'n_observations': len(animal_df),
                'n_events': animal_df[event_col].sum()
            })

        except Exception as e:
            # Catch convergence errors (e.g., if an animal never exited in one context)
            print(f"Model failed to converge for {animal}: {e}")

    # Return a compiled DataFrame of all animal results
    return pd.DataFrame(results)


def expand_to_person_period(reward_df, bin_width=0.5):
    """
    Expands the interval-based reward data into a discrete 'trial-period' format.
    Every row represents a single 0.5s time bin the animal spent in the port.
    """
    print(f"Expanding data into {bin_width}s bins...")
    expanded_records = []

    for _, row in reward_df.iterrows():
        animal = row['animal']
        session_id = row['session_id']
        duration = row['time_from_prev_reward']
        event = row['event_observed']
        entry_t_start = row['time_from_entry_at_reward']
        context = row['context']

        if pd.isna(duration) or duration <= 0:
            continue

        # Calculate how many bins this specific wait interval spans
        num_bins = math.ceil(duration / bin_width)

        for b in range(num_bins):
            # The time at the start of this specific bin
            bin_start_time = b * bin_width

            # Is this the final bin for this interval?
            is_last_bin = (b == num_bins - 1)

            # The exit event can ONLY happen in the very last bin of the sequence
            bin_event = 1 if (is_last_bin and event == 1) else 0

            expanded_records.append({
                'animal': animal,
                'session_id': session_id,
                'time_from_prev_reward': bin_start_time,  # How long they've been waiting for *this* drop
                'time_from_entry': entry_t_start + bin_start_time,  # How long they've been in the patch overall
                'context': context,  # 0.4 or 0.8
                'is_exit': bin_event  # 1 if they left, 0 if they stayed
            })

    expanded_df = pd.DataFrame(expanded_records)
    print(f"Expanded {len(reward_df)} intervals into {len(expanded_df)} discrete time bins.")
    return expanded_df


def run_nongrouped_hazard_model(expanded_df):
    """
    Runs a Logistic Regression to predict the probability of exiting
    in a given bin based on our behavioral variables.
    """
    # 1. Standardize (Z-score) the continuous timers
    # This puts them on the same scale (mean=0, std=1)
    expanded_df['time_rew_z'] = zscore(expanded_df['time_from_prev_reward'])
    expanded_df['time_entry_z'] = zscore(expanded_df['time_from_entry'])


    print("\nFitting Discrete-Time Hazard Model (Logistic Regression)...")

    # We predict the binary 'is_exit' using our three main variables
    # formula = "is_exit ~ time_from_prev_reward + time_from_entry + C(context)"
    formula = (
        "is_exit ~ "
        "time_rew_z * time_entry_z + "
        "time_rew_z * C(context) + "
        "time_entry_z * C(context)"
    )

    # Fit the model
    model = smf.logit(formula=formula, data=expanded_df).fit(disp=0)  # disp=0 hides the optimization printout

    # Print standard summary
    print(model.summary())

    # Calculate and print Odds Ratios for easier interpretation
    odds_ratios = pd.DataFrame({
        "Odds Ratio (OR)": np.exp(model.params),
        "Lower 95% CI": np.exp(model.conf_int()[0]),
        "Upper 95% CI": np.exp(model.conf_int()[1]),
        "p-value": model.pvalues
    })

    print("\n--- Interpretive Odds Ratios ---")
    print(odds_ratios.round(4))

    return model


def run_nested_behavioral_model(expanded_df):
    """
    Fits a GEE model that clusters by Animal.
    To account for unique session intercepts, we include session as a factor.
    """
    # z-scored
    expanded_df['time_rew_z'] = zscore(expanded_df['time_from_prev_reward'])
    expanded_df['time_entry_z'] = zscore(expanded_df['time_from_entry'])
    # # no centering, no z-scoring
    # expanded_df['time_rew_z'] = expanded_df['time_from_prev_reward']
    # expanded_df['time_entry_z'] = expanded_df['time_from_entry']
    # # only centered
    # expanded_df['time_rew_z'] = expanded_df['time_from_prev_reward'] - expanded_df[
    #     'time_from_prev_reward'].mean()
    # expanded_df['time_entry_z'] = expanded_df['time_from_entry'] - expanded_df['time_from_entry'].mean()

    print("\nFitting Behavioral Hazard Model...")

    # 1. Ensure animal is treated as the grouping variable
    # 2. Ensure session is treated as a categorical factor for unique intercepts
    # expanded_df['animal_session'] = expanded_df['animal'].astype(str) + "_" + expanded_df['session_id'].astype(str)

    # Formula includes 2-way interactions
    # C(context) treats 0.4 vs 0.8 as a discrete switch
    formula = (
        "is_exit ~ "
        "time_from_prev_reward * time_from_entry + "
        "time_from_prev_reward * C(context) + "
        "time_from_entry * C(context)"
    )

    # We group by 'animal' to allow for animal-level variance/slopes
    # We use an Exchangeable correlation structure to account for the nesting
    model = smf.gee(
        formula=formula,
        data=expanded_df,
        groups=expanded_df['animal'],
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()
    )

    result = model.fit()

    # We typically don't want to see 50 session intercepts in the summary,
    # so we'll filter the results for the variables we actually care about.
    full_summary = result.summary()

    # Extracting Odds Ratios for the core variables
    params = result.params
    conf = result.conf_int()
    odds_ratios = pd.DataFrame({
        "Odds Ratio (OR)": np.exp(params),
        "Lower 95% CI": np.exp(conf[0]),
        "Upper 95% CI": np.exp(conf[1]),
        "p-value": result.pvalues
    })

    # Filter out the session-specific intercepts from the display for clarity
    main_effects_or = odds_ratios[~odds_ratios.index.str.contains('session_id')]

    print("\n--- Core Interpretive Odds Ratios ---")
    print(main_effects_or.round(4))

    return result


def plot_vertical_forest_hazards(model_result):
    """
    Generates a Vertical Forest Plot (Whisker Plot) of Odds Ratios and 95% CIs.
    X-axis: Regressor Names
    Y-axis: Odds Ratio (Multiplier effect on leaving probability)
    """
    # 1. Extract params and confidence intervals
    params = model_result.params
    conf_int = model_result.conf_int()

    # 2. Convert from Log-Odds to Odds Ratios (OR = exp(beta))
    plot_df = pd.DataFrame({
        'Variable': params.index,
        'OR': np.exp(params.values),
        'Lower': np.exp(conf_int[0].values),
        'Upper': np.exp(conf_int[1].values)
    })

    # 3. Filter out the Intercept and Session IDs
    exclude = ['Intercept', 'session_id']
    plot_df = plot_df[~plot_df['Variable'].str.contains('|'.join(exclude))]

    # reorder
    custom_order = [
        'time_from_prev_reward',
        'time_from_entry',
        'C(context)[T.0.8]',
        'time_from_prev_reward:C(context)[T.0.8]',
        'time_from_entry:C(context)[T.0.8]',
        'time_from_prev_reward:time_from_entry'
    ]
    plot_df = plot_df.set_index('Variable').reindex(custom_order).reset_index()

    # 4. Human-readable names
    name_map = {
        'time_from_prev_reward': 'Time \nfrom Reward',
        'time_from_entry': 'Time \nfrom Entry',
        'C(context)[T.0.8]': 'Context',
        'time_from_prev_reward:time_from_entry': 'Time from Reward *\nTime from Entry',
        'time_from_prev_reward:C(context)[T.0.8]': 'Time from Reward *\nContext',
        'time_from_entry:C(context)[T.0.8]': 'Time from Entry *\nContext'
    }
    plot_df['Variable'] = plot_df['Variable'].map(lambda x: name_map.get(x, x))


    # 5. Plotting
    plt.figure(figsize=(10, 6))

    # Vertical error bars
    # yerr requires the distance from the point to the lower/upper bounds
    lower_err = plot_df['OR'] - plot_df['Lower']
    upper_err = plot_df['Upper'] - plot_df['OR']

    plt.errorbar(x=plot_df['Variable'], y=plot_df['OR'],
                 yerr=[lower_err, upper_err],
                 fmt='o', color='black', ecolor='red', capsize=8, markersize=10, lw=2.5)

    # Add the "Line of Null Effect" at OR = 1.0
    plt.axhline(y=1, color='blue', linestyle='--', alpha=0.6, label='No Effect')

    # Standard labels
    plt.ylabel("Odds Ratio (Effect on Leaving)")
    plt.xlabel("Regressor")
    plt.xticks(rotation=10, ha='right', fontsize=11)
    # plt.title("Drivers of Behavioral Exit Hazard", pad=20)

    # Optional: Use log scale for Y if effects vary by orders of magnitude
    # plt.yscale('log')

    # plt.grid(axis='y', alpha=0.2)
    sns.despine()
    plt.tight_layout()
    plt.show()


def calculate_vif(df, columns=['time_from_prev_reward', 'time_from_entry']):

    """
    Calculates the Variance Inflation Factor (VIF) and the corresponding R-squared
    to check for multicollinearity.
    """
    print(f"Calculating VIF and R-squared for: {', '.join(columns)}...\n")

    # 1. Subset the dataframe to just the columns we care about
    X = df[columns].dropna()

    # 2. Add a constant for accurate regression math
    X_with_constant = add_constant(X)

    # 3. Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_constant.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_constant.values, i)
        for i in range(X_with_constant.shape[1])
    ]

    # 4. Remove the 'const' row from the final output
    vif_data = vif_data[vif_data['Variable'] != 'const'].reset_index(drop=True)

    # 5. Reverse-engineer the R-squared from the VIF
    vif_data["R_squared"] = 1.0 - (1.0 / vif_data["VIF"])

    # 6. Display results nicely
    print("--- Collinearity Check ---")
    print(vif_data.round(3).to_string(index=False))
    print("-" * 40)
    print("Interpretation Guide:")
    print(" • VIF = 1.0 (R-squared = 0.0) : Completely Independent")
    print(" • VIF < 5.0 (R-squared < 0.8) : Moderate correlation (Safe for modeling)")
    print(" • VIF > 5.0 (R-squared > 0.8) : High correlation (Consider dropping one)")

    return vif_data

def export_person_period_data(expanded_df):
    expanded_df['time_rew_centered'] = expanded_df['time_from_prev_reward'] - expanded_df[
        'time_from_prev_reward'].mean()
    expanded_df['time_entry_centered'] = expanded_df['time_from_entry'] - expanded_df['time_from_entry'].mean()
    expanded_df['context_idx'] = (expanded_df['context'] > 0.5).astype(int)
    export_cols = [
        'animal',
        'session_id',
        'is_exit',
        'time_rew_centered',
        'time_entry_centered',
        'context_idx'
    ]
    export_df = expanded_df[export_cols].dropna()
    output_dir = os.path.join(config.PROJECT_ROOT, config.PROCESSED_DATA_SUBDIR)
    output_path = os.path.join(output_dir, "survival_data_for_R.csv")
    export_df.to_csv(output_path, index=False)
    print(f"Data successfully cleaned, centered, and exported to:\n{output_path}")


if __name__ == '__main__':
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]

    print("Extracting censored trial and reward durations...")
    t_df, r_df = extract_survival_data(animal_ids)

    # Cox Proportional Hazard Model
    # # --- Calculate Trial-Level HRs per Animal ---
    # trial_hr_df = calculate_hr_per_animal(t_df, duration_col='time_from_entry', event_col='event_observed')
    # print("Trial-Level HRs per Animal:")
    # print(trial_hr_df.to_string(index=False))
    # trial_hr_transposed = trial_hr_df.set_index('animal').T
    # output_dir = os.path.join(config.MAIN_DATA_ROOT, config.PAPER_OUTPUT_SUBDIR, 'time_from_entry_hr_per_ani.csv')
    # trial_hr_transposed.to_csv(output_dir)
    #
    # # --- Calculate Reward-Level HRs per Animal ---
    # reward_hr_df = calculate_hr_per_animal(r_df, duration_col='time_from_prev_reward', event_col='event_observed')
    # print("\nReward-Level HRs per Animal:")
    # print(reward_hr_df.to_string(index=False))
    # reward_hr_transposed = reward_hr_df.set_index('animal').T
    # output_dir = os.path.join(config.MAIN_DATA_ROOT, config.PAPER_OUTPUT_SUBDIR, 'time_from_reward_hr_per_ani.csv')
    # reward_hr_transposed.to_csv(output_dir)

    print("Plotting all 4 true combinations...")
    generate_four_combinations(t_df, r_df, save_fig=False)

    # print("Running statistical model..."
    # 1. Expand the data
    discrete_df = expand_to_person_period(r_df, bin_width=0.5)

    # # 2. Run the stats
    # # model_results = run_nongrouped_hazard_model(discrete_df[discrete_df['animal']=='SZ036'])
    # # model_results = run_glmm_hazard_model(discrete_df)
    # model_results = run_nested_behavioral_model(discrete_df)
    # # calculate_vif(discrete_df, ['time_from_prev_reward', 'time_from_entry'])
    # plot_vertical_forest_hazards(model_results)

    # 3. Export the csv survival data
    # export_person_period_data(discrete_df)
    print('hello')

