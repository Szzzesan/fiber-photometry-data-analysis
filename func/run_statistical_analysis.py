import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, FTestPower, FTestAnovaPower, NormalIndPower


def run_statistical_analysis(df, test_type):
    if test_type == 'pearson':
        corr_nri_right, p_nri_right = stats.pearsonr(df['NRI'], df['DA_right'])
        corr_nri_left, p_nri_left = stats.pearsonr(df['NRI'], df['DA_left'])
        corr_iri_right, p_iri_right = stats.pearsonr(df['IRI'], df['DA_right'])
        corr_iri_left, p_iri_left = stats.pearsonr(df['IRI'], df['DA_left'])
        print(
            f"Pearson Correlation Results:\nNRI vs DA_right: r={corr_nri_right}, p={p_nri_right}\nNRI vs DA_left: r={corr_nri_left}, p={p_nri_left}\nIRI vs DA_right: r={corr_iri_right}, p={p_iri_right}\nIRI vs DA_left: r={corr_iri_left}, p={p_iri_left}")

    elif test_type == 'spearman':
        corr_nri_right, p_nri_right = stats.spearmanr(df['NRI'], df['DA_right'])
        corr_nri_left, p_nri_left = stats.spearmanr(df['NRI'], df['DA_left'])
        corr_iri_right, p_iri_right = stats.spearmanr(df['IRI'], df['DA_right'])
        corr_iri_left, p_iri_left = stats.spearmanr(df['IRI'], df['DA_left'])
        print(
            f"Spearman Correlation Results:\nNRI vs DA_right: r={corr_nri_right}, p={p_nri_right}\nNRI vs DA_left: r={corr_nri_left}, p={p_nri_left}\nIRI vs DA_right: r={corr_iri_right}, p={p_iri_right}\nIRI vs DA_left: r={corr_iri_left}, p={p_iri_left}")

    elif test_type == 'linear_regression':
        model_right = smf.ols('DA_right ~ NRI + IRI + NRI:IRI', data=df).fit()
        model_left = smf.ols('DA_left ~ NRI + IRI + NRI:IRI', data=df).fit()
        print("Linear Regression Results for DA_right:")
        print(model_right.summary())
        print("Linear Regression Results for DA_left:")
        print(model_left.summary())

        # Use the effect size for power calculation (you can choose either model's effect size)
        # effect_size = effect_size_right if effect_size_right > effect_size_left else effect_size_left


    elif test_type == 'anova':
        df['NRI_group'] = pd.qcut(df['NRI'], q=3, labels=['low', 'medium', 'high'])
        df['IRI_group'] = pd.qcut(df['IRI'], q=3, labels=['low', 'medium', 'high'])
        model_right = smf.ols('DA_right ~ C(NRI_group) * C(IRI_group)', data=df).fit()
        model_left = smf.ols('DA_left ~ C(NRI_group) * C(IRI_group)', data=df).fit()
        anova_table_right = sm.stats.anova_lm(model_right, typ=2)
        anova_table_left = sm.stats.anova_lm(model_left, typ=2)
        print("ANOVA Results for DA_right:")
        print(anova_table_right)
        print("ANOVA Results for DA_left:")
        print(anova_table_left)

    else:
        print("Invalid test type. Choose from 'pearson', 'spearman', 'linear_regression', 'anova'.")
        return

    # Power Calculation
    def compute_effect_size_correlation(r):
        return r ** 2 / (1 - r ** 2) if r != 1 else np.inf  # Avoid division by zero
    def compute_effect_size_ftest(r_squared):
        return r_squared / (1 - r_squared) if r_squared < 1 else np.inf
    alpha = 0.05
    sample_size = len(df)

    if test_type in ['pearson', 'spearman']:
        effect_size_right = compute_effect_size_correlation(corr_nri_right)
        effect_size_left = compute_effect_size_correlation(corr_nri_left)
        power_analysis = NormalIndPower()
        power_right = power_analysis.solve_power(effect_size=effect_size_right, nobs1=sample_size, alpha=alpha)
        power_left = power_analysis.solve_power(effect_size=effect_size_left, nobs1=sample_size, alpha=alpha)
    elif test_type == 'linear_regression':
        effect_size_right = compute_effect_size_ftest(model_right.rsquared)
        effect_size_left = compute_effect_size_ftest(model_left.rsquared)
        df_num = 3
        df_denom = sample_size - df_num - 1
        power_analysis = FTestPower()
        power_right = power_analysis.solve_power(effect_size=effect_size_right, nobs=sample_size, df_num=df_num, df_denom=df_denom, alpha=alpha)
        power_left = power_analysis.solve_power(effect_size=effect_size_left, nobs=sample_size, df_num=df_num, df_denom=df_denom, alpha=alpha)

        print(f"Estimated Power for DA_right: {power_right}")
        print(f"Estimated Power for DA_left: {power_left}")
    elif test_type == 'anova':
        effect_size_right = compute_effect_size_ftest(model_right.rsquared)
        effect_size_left = compute_effect_size_ftest(model_left.rsquared)
        power_analysis = FTestAnovaPower()
        power_right = power_analysis.solve_power(effect_size=effect_size_right, nobs=sample_size, alpha=alpha, k_groups=5)
        power_left = power_analysis.solve_power(effect_size=effect_size_left, nobs=sample_size, alpha=alpha, k_groups=5)

    print(f"Estimated Power for DA_right: {power_right}")
    print(f"Estimated Power for DA_left: {power_left}")

# Example Usage
# run_statistical_analysis(df_combined, 'linear_regression')
