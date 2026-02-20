import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
import config
import data_loader
import time


def run_lmem_analysis(master_df):
    """
    Preprocesses data and runs the Linear Mixed Effects Model (LMEM).
    Returns the fitted results and the processed dataframe used for the model.
    """
    # --- Step 1: Data Preparation ---
    # Filter and copy to avoid SettingWithCopyWarning
    df = master_df[(master_df['IRI'] > 1)].copy()

    # Create compound identifier for nested random effects if needed (though strict animal grouping is used below)
    df['animal_hemisphere'] = df['animal'].astype(str) + '_' + df['hemisphere'].astype(str)

    # Log transform
    df['logNRI'] = np.log(df['NRI'])
    df['logIRI'] = np.log(df['IRI'])

    # Standardize regressors
    for col in ['logNRI', 'logIRI']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_std'] = (df[col] - mean) / std

    # --- Step 2: Define and Fit the Model ---
    # Note: Treatment reference for block is '0.8' (High), reference for side is 'ipsi'
    model_formula = (
        "DA ~ (logNRI_std + logIRI_std "
        "+ C(block, Treatment('0.8')) "
        "+ C(side_relative, Treatment('ipsi')))**2 "
        "+ C(hemisphere, Treatment('right'))"
    )

    random_slopes = (
        "1 + "  # Random Intercept
        "logNRI_std + "  # Random Slope for NRI (Time)
        "logIRI_std + "  # Random Slope for IRI
        "C(block, Treatment(reference='0.8'))"  # Random Slope for Context
    )

    print("Fitting LMEM... this may take a moment.")
    model = smf.mixedlm(
        model_formula,
        data=df,
        groups=df["animal"],
        re_formula=random_slopes,
        vc_formula={
            "session": "0 + C(session)",
            "site": "0 + C(animal_hemisphere)"
        }
    )

    # Using 'powell' as in your snippet, though 'lbfgs' is often faster if it converges
    model_results = model.fit(method="powell", reml=False)

    print("Model converged.")
    print(model_results.summary())

    return model_results, df


def calculate_approx_se(model_results, animal, fixed_main_name, fixed_inter_name=None, random_main_name=None):
    """
    Calculates the approximate Standard Error (SE) for a specific animal's coefficient.
    SE = sqrt( Var(Fixed) + Var(Random_Conditional) )

    Args:
        model_results: Fitted MixedLMResults.
        animal: Animal ID.
        fixed_main_name: Name of the main fixed effect (e.g., 'logNRI_std').
        fixed_inter_name: Name of the interaction fixed effect (e.g., 'logNRI_std:C(side)[T.contra]').
                          If None (e.g., for Ipsi side), only main effect variance is used.
        random_main_name: Name of the random effect term. If None, assumes same as fixed_main_name.
    """
    # 1. Variance of the Fixed Effect Part
    cov_params = model_results.cov_params()

    if fixed_inter_name is None:
        # Just the main effect: Var(Beta_main)
        var_fixed = cov_params.loc[fixed_main_name, fixed_main_name]
    else:
        # Main + Interaction: Var(Beta_main + Beta_inter)
        # = Var(M) + Var(I) + 2*Cov(M, I)
        var_m = cov_params.loc[fixed_main_name, fixed_main_name]
        var_i = cov_params.loc[fixed_inter_name, fixed_inter_name]
        cov_mi = cov_params.loc[fixed_main_name, fixed_inter_name]
        var_fixed = var_m + var_i + 2 * cov_mi

    # 2. Variance of the Random Effect Part (Conditional Variance of BLUP)
    # The random effect name might differ slightly in the results object (e.g., categorical names)
    # We try to find the matching key in random_effects_cov
    var_random = 0.0

    if random_main_name is None:
        random_main_name = fixed_main_name

    # Check if this animal has random effects calculated
    if animal in model_results.random_effects_cov:
        animal_cov = model_results.random_effects_cov[animal]

        # Try to find the exact key in the random covariance matrix
        # The keys usually match the re_formula terms
        # Sometimes statsmodels renames them (e.g. 'C(block)[T.0.4]').
        # We do a fuzzy match if exact match fails.
        matched_key = None
        if random_main_name in animal_cov.index:
            matched_key = random_main_name
        else:
            # Fallback: check if any key contains the main name
            for key in animal_cov.index:
                if random_main_name in key:
                    matched_key = key
                    break

        if matched_key:
            var_random = animal_cov.loc[matched_key, matched_key]

    # 3. Total Standard Error
    total_se = np.sqrt(var_fixed + var_random)
    return total_se


def get_marginal_estimates(model_results, regressor_name):
    """
    Calculates the Marginal Mean estimate and SE for a main effect,
    averaging over its interactions (Side and Block).

    Formula: Beta_Marginal = Beta_Main + 0.5 * Beta_Int_Side + 0.5 * Beta_Int_Block
    """
    params = model_results.params
    cov_params = model_results.cov_params()

    # 1. Base weights: Main Effect = 1.0
    weights = {regressor_name: 1.0}

    # 2. Identify Interactions to marginalize over
    # We look for interactions with 'side_relative' and 'block'
    # Note: We assume 50/50 split for marginalization

    # Helper to find interaction name in params
    def find_interaction(main, inter_term_fragment):
        # Statsmodels names can be "A:B" or "B:A"
        for p in params.index:
            if main in p and inter_term_fragment in p and ":" in p:
                return p
        return None

    # Define interaction partners (The variables we want to average over)
    # If regressor is Time, we avg over Side and Context.
    # Usually for categorical main effects (Side), we avg over the other factors (Context).
    # Time and IRI are continuous and standardized, so they are 0 at the mean.

    # Common fragments for your model
    side_frag = "side_relative"
    block_frag = "block"

    # Find interactions
    term_side = find_interaction(regressor_name, side_frag)
    term_block = find_interaction(regressor_name, block_frag)

    if term_side:
        weights[term_side] = 0.5
    if term_block:
        weights[term_block] = 0.5

    # 3. Calculate Estimate (Linear Combination)
    estimate = sum(params[name] * w for name, w in weights.items())

    # 4. Calculate Variance (w' * Cov * w)
    # Var(aX + bY) = a^2Var(X) + b^2Var(Y) + 2abCov(X,Y)
    # Generalized: sum(w_i * w_j * Cov(i, j))
    variance = 0.0
    for name_i, w_i in weights.items():
        for name_j, w_j in weights.items():
            variance += w_i * w_j * cov_params.loc[name_i, name_j]

    se = np.sqrt(variance)

    return estimate, se, weights


def figg_LMEM_coefficients_v3(model_results, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

        # Define mapping and order
    name_map = {
        'C(block, Treatment(\'0.8\'))[T.0.4]': 'context',
        'logNRI_std': 'time',
        'logIRI_std': 'IRI',
        'C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'side',
    }

    # Calculate data for plot
    plot_data = []

    # Process known main effects to get MARGINAL means
    for tech_name, simple_name in name_map.items():
        if tech_name in model_results.params:
            est, se, _ = get_marginal_estimates(model_results, tech_name)
            plot_data.append({
                'name': simple_name,
                'estimate': est,
                'lower': est - 1.96 * se,
                'upper': est + 1.96 * se,
                'is_main': True
            })

    # Process interactions (keep as raw coefficients)
    # We filter for names containing ':' that match our key terms
    for p in model_results.params.index:
        if ':' in p and p not in name_map:
            # Generate a simplified name
            simple = p
            for k, v in name_map.items():
                simple = simple.replace(k, v)
            # Cleanup naming
            simple = simple.replace(":", " * ")

            # Only keep interactions relevant to our terms
            if any(x in simple for x in ['time', 'context', 'IRI', 'side']):
                est = model_results.params[p]
                # CI from results
                ci = model_results.conf_int().loc[p]

                plot_data.append({
                    'name': simple,
                    'estimate': est,
                    'lower': ci[0],
                    'upper': ci[1],
                    'is_main': False
                })

    # Convert to DataFrame
    df_plot = pd.DataFrame(plot_data)

    # Sort
    desired_order = [
        'time', 'IRI', 'context',
        'time * side', 'IRI * side', 'context * side',
        'time * IRI', 'time * context', 'IRI * context'
    ]
    df_plot['sort_cat'] = pd.Categorical(df_plot['name'], categories=desired_order, ordered=True)
    df_plot = df_plot.sort_values('sort_cat').dropna()

    # Plot
    errors = df_plot['upper'] - df_plot['estimate']  # Error for top part
    # errorbar expects symmetric or (2, N)
    # We calculate symmetric approximation or absolute
    yerr = [df_plot['estimate'] - df_plot['lower'], df_plot['upper'] - df_plot['estimate']]

    axes.errorbar(
        x=df_plot['name'],
        y=df_plot['estimate'],
        yerr=yerr,
        fmt='o',
        color='black',
        capsize=2,
        linewidth=1.5,
        markersize=4,
        ecolor='red'
    )

    axes.axhline(y=0, color='grey', linestyle='--')
    # axes.set_title('Coefficient (95% CI)')
    axes.set_ylabel('Coefficient Estimate (95% CI)')
    axes.set_xticklabels(df_plot['name'], rotation=10, ha='right', fontsize=11)
    axes.set_xlabel('Regressor', fontsize=15, x=0.58)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    if return_handle:
        plt.tight_layout()
        fig.show()
        return fig, axes


def plot_single_marginal_coeff(model_results, regressor_tech_name, display_name, ax):
    """
    Plots a single coefficient (Marginal Mean) in its own axis.
    """
    est, se, _ = get_marginal_estimates(model_results, regressor_tech_name)
    lower = est - 1.96 * se
    upper = est + 1.96 * se

    # Plot
    ax.errorbar(
        x=[display_name],
        y=[est],
        yerr=[[est - lower], [upper - est]],
        fmt='o', color='black', capsize=2, linewidth=1.5, markersize=4, ecolor='red'
    )

    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xticklabels([display_name], rotation=10, ha='right', fontsize=11)
    # ax.set_ylabel('')
    # ax.set_title(display_name, y=1.02)
    # ax.set_ylabel('Coefficient') # Optional, maybe redundant next to the main plot
    sns.despine(ax=ax)


def plot_animal_coefficients(model_results, df, regressor_name, display_name, xlim=[-0.1, 0.65], ax=None):
    """
    Plots the animal-specific coefficients (Fixed Effect + Random Effect) for a given regressor.
        1. The 'Population' dot is the Marginal Mean (averaged over Context/Side).
        2. The Animal dots are centered around this mean (i.e., they include 0.5 * Context Interaction).

    Args:
        model_results: The fitted MixedLMResults object.
        df: The dataframe used to fit the model (needed to check which animal has which side).
        regressor_name (str): The name of the regressor to plot (e.g., 'logNRI_std', 'logIRI_std', "C(block, Treatment('0.8'))[T.0.4]").
        display_name (str): The regressor name I'd like to display (e.g. 'Time', 'IRI', 'Context')
        ax: Matplotlib axis (optional).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 8))
        return_handle = True
    else:
        fig = None
        return_handle = False

    params = model_results.params
    random_effects = model_results.random_effects

    # --- 1. Calculate Marginal Metrics for Population ---
    # This automatically includes +0.5 * Side_Int and +0.5 * Block_Int
    pop_est, pop_se, weights_used = get_marginal_estimates(model_results, regressor_name)

    # Identify which specific interaction terms were used, so we can apply them to animals
    # Keys in weights_used are the parameter names
    term_side_int = None
    term_block_int = None

    for term in weights_used:
        if term == regressor_name: continue
        if 'side_relative' in term: term_side_int = term
        if 'block' in term: term_block_int = term

    # Coefficients
    fixed_main = params[regressor_name]
    fixed_side_int = params[term_side_int] if term_side_int else 0.0
    fixed_block_int = params[term_block_int] if term_block_int else 0.0

    # --- 2. Animal Processing ---
    # Logic:
    # We want to represent the animal's "Average" response across Contexts.
    # Animal Ipsi = (Beta_Main + u_Main) + 0.5 * (Beta_Block + u_Block?)
    # To keep it simple and match the population marginal mean:
    # Common Shift = 0.5 * fixed_block_int
    # Animal Ipsi  = Fixed_Main + Common_Shift + u_Main
    # Animal Contra = Fixed_Main + Common_Shift + Fixed_Side_Int + u_Main

    common_shift = 0.5 * fixed_block_int

    target_order = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    name_map = {k: f"Animal {i + 1}" for i, k in enumerate(target_order)}
    # name_map.update({k: k for k in target_order})  # Fallback

    plot_data = []
    available_animals = set(df['animal'].unique())
    ordered_animals = [a for a in target_order if a in available_animals]

    for animal in ordered_animals:
        # Get Random Effect
        animal_re = 0.0
        re_series = random_effects.get(animal, pd.Series(dtype=float))

        # Fuzzy match for RE key
        re_key = regressor_name
        if re_key not in re_series:
            for k in re_series.index:
                if regressor_name in k:
                    re_key = k
                    break
        if re_key in re_series:
            animal_re = re_series[re_key]

        # Calculate Values
        animal_sides = df[df['animal'] == animal]['side_relative'].unique()

        # IPSI
        if 'ipsi' in animal_sides:
            # Mean Ipsi = Base + RE + 0.5 * Context_Effect
            val = fixed_main + animal_re + common_shift
            # SE: We use the SE of the Main term for the error bars (approximation)
            se = calculate_approx_se(model_results, animal, regressor_name, None, re_key)
            plot_data.append({'ID': animal, 'Side': 'Ipsilateral', 'Val': val, 'SE': se})

        # CONTRA
        if 'contra' in animal_sides:
            # Mean Contra = Base + RE + Side_Effect + 0.5 * Context_Effect
            val = fixed_main + animal_re + fixed_side_int + common_shift
            se = calculate_approx_se(model_results, animal, regressor_name, term_side_int, re_key)
            plot_data.append({'ID': animal, 'Side': 'Contralateral', 'Val': val, 'SE': se})

    # --- 3. Plotting ---
    c_ipsi = sns.color_palette("Paired")[1]
    c_contra = sns.color_palette("Paired")[5]
    animal_to_y = {a: i for i, a in enumerate(ordered_animals)}
    offset = 0.1

    # Plot Animals
    for item in plot_data:
        y_center = animal_to_y[item['ID']]
        y_pos = y_center - offset if item['Side'] == 'Ipsilateral' else y_center + offset
        color = c_ipsi if item['Side'] == 'Ipsilateral' else c_contra

        lower = item['Val'] - 1.96 * item['SE']
        upper = item['Val'] + 1.96 * item['SE']

        ax.plot([lower, upper], [y_pos, y_pos], color=color, lw=1, alpha=0.8)
        ax.plot(item['Val'], y_pos, 'o', color=color, markersize=3)

    # Plot Population Marginal Mean
    y_summ = len(ordered_animals)
    ax.plot([pop_est - 1.96 * pop_se, pop_est + 1.96 * pop_se], [y_summ, y_summ], color='black', lw=1.5)
    ax.plot(pop_est, y_summ, 'o', color='black', markersize=4)

    # Styling
    y_ticks = list(range(len(ordered_animals))) + [y_summ]
    y_labels = [name_map.get(a, a) for a in ordered_animals] + ["Population\nFixed Effect"]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"{display_name}")
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(pop_est, color='purple', linestyle='--', alpha=0.5)

    sns.despine(ax=ax)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)

    if return_handle:
        plt.tight_layout()
        fig.show()
        return fig, ax


def setup_axes():
    """
        Sets up a composite figure layout with nested gridspecs.

        Layout:
        - Row 1: Split 6:1
            - Col 1: Main Summary (A) -> Excludes 'side'
            - Col 2: Side Summary (B) -> Only 'side' (large range)
        - Row 2: Split 3:3:1
            - Col 1: Time (C)
            - Col 2: IRI (D)
            - Col 3: Context (E)
        """
    fig = plt.figure(figsize=(12, 10))

    # 1. Outer Grid: 2 Rows (Height Ratio 1:2)
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=fig)

    # 2. Top Row Grid (Nested): 2 Columns (Width Ratio 6:1)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0], width_ratios=[8, 1])
    ax_summary_main = fig.add_subplot(gs_top[0])
    ax_summary_side = fig.add_subplot(gs_top[1])

    # 3. Bottom Row Grid (Nested): 3 Columns (Width Ratio 3:3:1)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], width_ratios=[3, 1, 3])
    ax_time = fig.add_subplot(gs_bottom[0])
    ax_iri = fig.add_subplot(gs_bottom[2])
    ax_context = fig.add_subplot(gs_bottom[1])

    axes_dict = {
        'summary_main': ax_summary_main,
        'summary_side': ax_summary_side,
        'time': ax_time,
        'iri': ax_iri,
        'context': ax_context
    }

    # 4. Add Lettering
    # Define order of axes to label
    axes_to_letter = ['summary_main', 'time', 'context', 'iri']
    lettering = 'abcd'

    for i, key in enumerate(axes_to_letter):
        ax = axes_dict[key]

        # User-provided text transform for precise placement
        # (-20/72, +7/72) moves text left by 20 points and up by 7 points relative to (0,1)
        ax.text(
            0.0, 1.0, lettering[i],
            transform=(ax.transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
            fontsize=16,
            va='bottom',
            fontfamily='sans-serif',
            weight='bold'
        )

    # Optional: Adjust spacing to prevent overlap
    plt.subplots_adjust(hspace=0.4, wspace=0.15)

    return fig, axes_dict


def main():
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    # animal_ids=["SZ036"]
    master_df1 = data_loader.load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features',
                                                                day_0='2023-11-30', hemisphere_qc=1,
                                                                file_format='parquet')

    animal_ids = ["RK007", "RK008"]
    master_df2 = data_loader.load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features',
                                                                day_0='2025-06-17', hemisphere_qc=1,
                                                                file_format='parquet')
    master_DA_features_df = pd.concat([master_df1, master_df2], ignore_index=True)
    model_results, master_df = run_lmem_analysis(master_DA_features_df)
    tic = time.time()
    fig, axes = setup_axes()
    figg_LMEM_coefficients_v3(model_results, axes=axes['summary_main'])
    plot_single_marginal_coeff(model_results, 'C(side_relative, Treatment(\'ipsi\'))[T.contra]', 'side',
                               ax=axes['summary_side'])
    plot_animal_coefficients(model_results, master_df, regressor_name='logNRI_std', display_name='Time of Reward Delivery',
                             xlim=[-0.15, 0.65], ax=axes['time'])
    plot_animal_coefficients(model_results, master_df, regressor_name='logIRI_std', display_name='IRI',
                             xlim=[-0.15, 0.65], ax=axes['iri'])
    plot_animal_coefficients(model_results, master_df, regressor_name='C(block, Treatment(\'0.8\'))[T.0.4]',
                             display_name='Context', xlim=[-0.05, 0.25], ax=axes['context'])

    c_ipsi = sns.color_palette("Paired")[1]
    c_contra = sns.color_palette("Paired")[5]
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_ipsi, label='Ipsilateral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_contra, label='Contralateral'),
        # Line2D([0], [0], marker='o', color='black', markerfacecolor='black', label='Marginal Mean')
    ]
    axes['iri'].legend(handles=legend_elements, loc='upper right',
                       title='Side rel.\nReward Delivery\n(Investment Port)')
    # Remove Y-axis labels and tick labels
    axes['iri'].set_ylabel('')
    axes['iri'].set_yticklabels([])
    axes['context'].set_ylabel('')
    axes['context'].set_yticklabels([])
    print(f'Coefficient plot took {time.time() - tic:.2f} seconds')

    # save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    #
    # # Create directory if it doesn't exist
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #     print(f"Created directory: {save_dir}")
    #
    # filename = "fig_4-5_LMEM_whiskers.png"
    # # You can also save as .pdf or .svg for vector graphics
    # # filename = "LMEM_Composite_Marginal.pdf"
    #
    # full_path = os.path.join(save_dir, filename)
    # fig.savefig(full_path, dpi=300, bbox_inches='tight')
    # print(f"Saved figure to: {full_path}")
    fig.show()


if __name__ == '__main__':
    main()

    # name_map
    # name_map = {
    #     'C(block, Treatment(\'0.8\'))[T.0.4]': 'context',
    #     'logNRI_std': 'time',
    #     'logIRI_std': 'IRI',
    #     'C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'side',
    #     'logNRI_std:C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'time * side',
    #     'logNRI_std:logIRI_std': 'time * IRI',
    #     'logNRI_std:C(block, Treatment(\'0.8\'))[T.0.4]': 'time * context',
    #     'logIRI_std:C(block, Treatment(\'0.8\'))[T.0.4]': 'IRI * context',
    #     'logIRI_std:C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'IRI * side',
    #     'C(block, Treatment(\'0.8\'))[T.0.4]:C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'context * side',
    #     # 'logNRI_std:logIRI_std:C(block, Treatment(reference=\'0.8\'))[T.0.4]': 'time * IRI * context',
    #     # 'logIRI_std:C(block, Treatment(\'0.8\'))[T.0.4]:C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'IRI * context * side',
    #     # 'logNRI_std:logIRI_std:C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'time * IRI * side',
    #     # 'logNRI_std:C(block, Treatment(\'0.8\'))[T.0.4]:C(side_relative, Treatment(\'ipsi\'))[T.contra]': 'time * context * side'
    # }
