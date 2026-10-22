import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.svm import SVC

# Import from your repository structure
import config
from figure_making import data_loader
import quality_control as qc


def build_continuous_session_data(animal_id, session_id=None, session_long_name=None, step=0.1, post_time=5.0):
    """
    Transforms discrete trial/reward data into continuous 100ms bins for the SVM.
    Mimics the original _get_var_df function.
    """
    # Load required dataframes using your data_loader
    trial_df = data_loader.load_session_dataframe(animal_id, 'trial_df', session_id, session_long_name)
    expreward_df = data_loader.load_session_dataframe(animal_id, 'expreward_df', session_id, session_long_name)

    if trial_df is None or expreward_df is None:
        return None

    # FIX: Ensure the block column is numeric (float) rather than string
    expreward_df['block'] = expreward_df['block'].astype(float)

    # Filter for valid trials
    valid_trials = trial_df[trial_df['is_valid_trial'] == True].copy()

    # Exclude severe outliers (mean + 3*std of duration)
    valid_trials['duration'] = valid_trials['exp_exit'] - valid_trials['exp_entry']
    outlier_cutoff = valid_trials['duration'].mean() + 3 * valid_trials['duration'].std()
    valid_trials = valid_trials[valid_trials['duration'] <= outlier_cutoff]

    res_dict = {
        'trial_id': [], 'time_from_entry': [], 'time_from_reward': [],
        'block_value': [], 'stay_leave': [], 'true_exit_time': []
    }

    for _, trial in valid_trials.iterrows():
        trial_id = trial['trial']
        entry_t = trial['exp_entry']
        exit_t = trial['exp_exit']

        # Get rewards for this specific trial
        trial_rewards = expreward_df[expreward_df['trial'] == trial_id]['reward_time'].values

        if len(trial_rewards) == 0:
            continue  # Skip unrewarded trials

        # Determine block context (0.4 or 0.8) from expreward_df
        # Since we casted to float above, this will now be a numerical float
        block_val = expreward_df[expreward_df['trial'] == trial_id]['block'].iloc[0]

        # Create continuous time array from entry to (exit + post_time)
        t_from_entry_arr = np.arange(0, (exit_t - entry_t) + post_time, step)

        # Determine stay/leave state (0 while in port, 1 after exit)
        stay_leave = (t_from_entry_arr > (exit_t - entry_t)).astype(int)

        # Calculate time_from_reward for each bin
        t_reward_arr = t_from_entry_arr.copy()
        sub_count = 0
        for reward_time in trial_rewards:
            rew_t_relative = reward_time - entry_t
            # Reset time_from_reward when a reward occurs
            mask = (t_reward_arr + sub_count) > rew_t_relative
            t_reward_arr[mask] -= (rew_t_relative - sub_count)
            sub_count = rew_t_relative

        res_dict['trial_id'].append(np.repeat(trial_id, len(t_from_entry_arr)))
        res_dict['time_from_entry'].append(t_from_entry_arr)
        res_dict['time_from_reward'].append(t_reward_arr)
        res_dict['block_value'].append(np.repeat(block_val, len(t_from_entry_arr)))
        res_dict['stay_leave'].append(stay_leave)
        res_dict['true_exit_time'].append(np.repeat(exit_t - entry_t, len(t_from_entry_arr)))

    if not res_dict['time_from_entry']:
        return None

    # Concatenate all lists into a single dataframe
    cont_var_df = pd.DataFrame({
        k: np.concatenate(v) for k, v in res_dict.items()
    })

    return cont_var_df


def get_svm_coefficients_and_predictions(animal_ids):
    """
    Fits the SVM per session, extracts whitened coefficients, and calculates predictions.
    """
    svm_records = []
    all_predictions = []

    for animal in animal_ids:
        # Get processed directory to count sessions
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            continue

        session_files = [f for f in os.listdir(processed_dir) if f.endswith('trial_df.parquet')]

        for session_id in range(len(session_files)):
            cont_df = build_continuous_session_data(animal, session_id=session_id)
            if cont_df is None or len(cont_df) < 50:
                continue

            features = cont_df[['time_from_entry', 'time_from_reward', 'block_value']]
            target = cont_df['stay_leave']

            # Whiten the features for cross-session coefficient comparison
            features_whitened = (features - features.mean()) / features.std()

            # Fit Linear SVM
            model = SVC(kernel='linear', C=1, gamma=1)
            model.fit(features_whitened, target)

            # Save coefficients
            coefs = model.coef_[0]
            svm_records.append({
                'animal': animal,
                'session_id': session_id,
                'entry_coef': coefs[0],
                'reward_coef': coefs[1],
                'block_coef': coefs[2]
            })

            # For Fig 3.1j: Calculate decision function (distance to boundary)
            # We predict using UNWHITENED features here to get raw time projections,
            # or we calculate the point where decision_function crosses 0.
            # Following colleague's Style 1 logic: SVM Predict = boundary_dist + (time_from_entry - time_from_reward)
            pred_model = SVC(kernel='linear', C=1, gamma=1)
            pred_model.fit(features, target)  # Unwhitened for time scale
            cont_df['boundary_distance'] = pred_model.decision_function(features)

            # Filter for bins right before they leave
            leave_times = cont_df[cont_df['stay_leave'] == 1].groupby('trial_id').first().reset_index()
            leave_times['SVM Predict'] = leave_times['boundary_distance'] + (
                        leave_times['time_from_entry'] - leave_times['time_from_reward'])
            leave_times['animal'] = animal
            all_predictions.append(leave_times)

    return pd.DataFrame(svm_records), pd.concat(all_predictions, ignore_index=True)


def fig3_1g_svm_coef(svm_df, save_fig=False):
    """
    Replicates Figure 3.1g: Boxplot of whitened SVM coefficients.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    coef_only = svm_df[['reward_coef', 'entry_coef', 'block_coef']].copy()

    # Statistical Testing
    print("--- 1-Sample T-Tests against 0 ---")
    for pred in ['reward_coef', 'entry_coef', 'block_coef']:
        t_stat, p_val = stats.ttest_1samp(coef_only[pred], 0)
        print(f"{pred}: p = {p_val:.4e}, mean = {coef_only[pred].mean():.4f}")

    # Plot
    sns.boxplot(data=coef_only, ax=ax, palette='Greys', dodge=False, fliersize=0)
    sns.stripplot(data=coef_only, ax=ax, color='black', alpha=0.3, jitter=True)

    ax.set_ylabel('SVM Coefficients')
    ax.set_xticklabels(['Time from\nReward', 'Time from\nEntry', 'Context'])
    ax.axhline(0, linestyle='--', color='k', alpha=0.5)

    # Significance bars logic (simplified for replication)
    # Dynamic Statistical Testing and Annotation
    # Define pairs to compare: (Column 1, Column 2, X-coordinate 1, X-coordinate 2)
    y_lim = ax.get_ylim()
    base_y = y_lim[1] + 0.2  # The original starting height above the data
    y_step = (y_lim[1] - y_lim[0]) * 0.1  # Standard step height

    # Define pairs to compare: (Column 1, Column 2, X-coord 1, X-coord 2, custom_y_height)
    comparisons = [
        ('reward_coef', 'entry_coef', 0, 1, base_y),  # Stays where it was
        ('entry_coef', 'block_coef', 1, 2, base_y - 0.4),  # Lowered by 0.4
        ('reward_coef', 'block_coef', 0, 2, base_y + (2 * y_step))  # Stays at its highest position
    ]

    print("--- Paired T-Tests Between Coefficients ---")
    for pred1, pred2, x1, x2, y_height in comparisons:
        # Calculate paired t-test
        t_stat, p_val = stats.ttest_rel(coef_only[pred1], coef_only[pred2])

        # Determine star rating
        if p_val < 0.001:
            sig_str = '***'
        elif p_val < 0.01:
            sig_str = '**'
        elif p_val < 0.05:
            sig_str = '*'
        else:
            sig_str = 'ns'

        print(f"{pred1} vs {pred2}: p = {p_val:.4e}, t = {t_stat:.4f} ({sig_str})")

        # Draw the annotation bar only if significant
        if sig_str != 'ns':
            ax.plot([x1, x2], [y_height, y_height], color='k', alpha=0.8)
            # The text goes slightly above the drawn line
            ax.text((x1 + x2) / 2, y_height + 0.05, sig_str,
                    fontsize=10, ha='center', va='bottom', fontweight='bold')

    # Expand the y-axis ceiling to neatly frame the highest statistical bracket
    highest_bar = base_y + (2 * y_step)
    ax.set_ylim(y_lim[0], highest_bar + 0.5)

    sns.despine()
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR, 'fig3_1g_svm_coef.png'), dpi=300)
    plt.show()


def fig3_1j_svm_summary(predictions_df, save_fig=False):
    """
    Replicates Figure 3.1j: True vs Predicted Exit Times.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Drop NaNs
    df_clean = predictions_df.dropna(subset=['true_exit_time', 'SVM Predict'])

    # Calculate R-squared
    X = sm.add_constant(df_clean['true_exit_time'])
    y = df_clean['SVM Predict']
    r2 = sm.OLS(y, X).fit().rsquared
    print(f'SVM Predict R-squared: {r2:.3f}')

    # Plotting
    sns.scatterplot(data=df_clean, x='true_exit_time', y='SVM Predict', alpha=0.5, ax=ax, s=15, color='goldenrod')
    sns.regplot(data=df_clean, x='true_exit_time', y='SVM Predict', scatter=False, ax=ax, color='goldenrod')

    # Formatting
    max_val = max(df_clean['true_exit_time'].max(), df_clean['SVM Predict'].max())
    limit = min(20, int(np.ceil(max_val / 5) * 5))  # Cap at 20 or nearest 5

    ax.plot([-30, 30], [-30, 30], color='k', linestyle='--', alpha=0.5)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('True Exit Time (s)')
    ax.set_ylabel('Projected Exit Time (s)')

    ax.text(0.05, 0.95, f'$r^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=12, va='top', ha='left')

    sns.despine()
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR, 'fig3_1j_svm_summary.png'),
                    dpi=300)
    plt.show()


def _get_surface(x, y, z, x_range=None, y_range=None):
    """
    Mathematical helper from calc_svm.py to calculate the true 3D plane
    from the boundary coordinates.
    """
    if x_range is None:
        x_range = [np.min(x), np.max(x)]
    if y_range is None:
        y_range = [np.min(y), np.max(y)]

    assert len(x) == len(y) == len(z) >= 3
    x0, x1, x2 = x[0], x[1], x[2]
    y0, y1, y2 = y[0], y[1], y[2]
    z0, z1, z2 = z[0], z[1], z[2]
    ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
    vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]
    u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
    point = np.array([x0, y0, z0])
    normal = np.array(u_cross_v)
    unit_normal = normal / np.linalg.norm(normal)

    d = -point.dot(normal)
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 200),
                         np.linspace(y_range[0], y_range[1], 200))
    z_surf = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z_surf, unit_normal


def plot_3d_svm_surface(animal_id, session_id=0, save_fig=False):
    """
    Plots the 3D SVM decision boundary for a single session,
    reproducing the visual settings from calc_svm.py.
    """
    # 1. Get the continuous binned data
    cont_df = build_continuous_session_data(animal_id, session_id=session_id)
    if cont_df is None or len(cont_df) < 50:
        print(f"Not enough valid data for {animal_id} session {session_id}.")
        return

    # 2. Fit the Unwhitened Model
    # We use unwhitened data here so the axes map directly to real seconds
    feature_cols = ['time_from_entry', 'time_from_reward', 'block_value']
    X = cont_df[feature_cols]
    y = cont_df['stay_leave']

    model = SVC(kernel='linear', C=1, gamma=1)
    model.fit(X, y)

    # 3. Setup Mesh Limits (x1: block, x2: entry, x3: reward)
    x1_lim = [0.2, 1.0]
    x2_lim = [cont_df['time_from_entry'].min(), cont_df['time_from_entry'].max()]
    x3_lim = [cont_df['time_from_reward'].min(), cont_df['time_from_reward'].max()]

    # 4. Generate the 3D Meshgrid to find the boundary
    x1, x2, x3 = np.meshgrid(np.linspace(x1_lim[0], x1_lim[1], 4),
                             np.linspace(x2_lim[0], x2_lim[1], 4),
                             np.linspace(x3_lim[0], x3_lim[1], 105))
    x1, x2, x3 = x1[1:3, 1:3], x2[1:3, 1:3], x3[1:3, 1:3]

    # Must match the order of 'feature_cols' above for accurate prediction
    mesh_df = pd.DataFrame({
        'time_from_entry': x2.ravel(),
        'time_from_reward': x3.ravel(),
        'block_value': x1.ravel()
    })

    # Find where the model flips from 0 to 1 along the time_from_reward axis
    y_pred_mesh = model.predict(mesh_df).reshape(x1.shape)
    boundary_arr = np.linspace(x3_lim[0], x3_lim[1], y_pred_mesh.shape[2])[np.argmax(y_pred_mesh, axis=2)]

    # Calculate the plottable plane
    x1_surf, x2_surf, boundary_arr, _ = _get_surface(
        x1[:, :, 0].ravel(), x2[:, :, 0].ravel(), boundary_arr.ravel(),
        x_range=x1_lim, y_range=x2_lim
    )

    # 5. Prep the scatter points
    df = cont_df.copy()
    df['y_pred'] = model.predict(X)

    line_color = 'darkred'
    surf_color = '#d9d9ff'  # pale purple
    exit_color = 'firebrick'
    stay_color = 'steelblue'

    df['color'] = df['stay_leave'].apply(lambda x: exit_color if x == 1 else stay_color)

    # Split points into above and below boundary to avoid z-order rendering issues
    if (boundary_arr[0, -1] - boundary_arr[-1, 0]) / (x3_lim[1] - x3_lim[0]) > 0.7:
        df_below = df[df['y_pred'] == 1]
        df_above = df[df['y_pred'] == 0]
    else:
        df_below = df[df['y_pred'] == 0]
        df_above = df[df['y_pred'] == 1]

    # 6. Build the Figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Plot the surface edge lines
    ax.plot(x1_surf[[0, -1], 0], x2_surf[[0, -1], 0], boundary_arr[[0, -1], 0], c=line_color, zorder=-10)
    ax.plot(x1_surf[-1, [0, -1]], x2_surf[-1, [0, -1]], boundary_arr[-1, [0, -1]], c=line_color, zorder=-10)

    # Scatter points below surface
    ax.scatter(df_below['block_value'], df_below['time_from_entry'], df_below['time_from_reward'],
               c=df_below['color'], s=5, alpha=1)

    # Draw the boundary surface
    ax.plot_surface(x1_surf, x2_surf, boundary_arr, alpha=0.5, color=surf_color)

    # Scatter points above surface
    ax.scatter(df_above['block_value'], df_above['time_from_entry'], df_above['time_from_reward'],
               c=df_above['color'], s=5, alpha=1)

    # Styling and View Angle
    ax.set_xlabel('Block Value')
    ax.set_ylabel('Time from Entry')
    ax.set_zlabel('Time from Reward')
    ax.set_xlim(x1_lim[0], x1_lim[1])
    ax.set_ylim(x2_lim[0], x2_lim[1])
    ax.set_zlim(x3_lim[0], x3_lim[1])
    ax.view_init(azim=315, elev=20)
    fig.suptitle(f'SVM Decision Boundary: {animal_id} (Session {session_id})')

    if save_fig:
        # Save to single-animal figure folder
        save_path = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.FIGURE_SUBDIR)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{animal_id}_session{session_id}_svm_3d.png'), dpi=300)

    plt.show()


def plot_2d_svm_projections(animal_id, session_id=0, save_fig=False):
    """
    Plots a 2D projection of the SVM decision boundary (Entry vs Reward time),
    separated into two subplots for the Low (0.4) and High (0.8) blocks.
    """
    # 1. Get the continuous binned data
    cont_df = build_continuous_session_data(animal_id, session_id=session_id)
    if cont_df is None or len(cont_df) < 50:
        print(f"Not enough valid data for {animal_id} session {session_id}.")
        return

    # 2. Fit the Unwhitened Model
    feature_cols = ['time_from_entry', 'time_from_reward', 'block_value']
    X = cont_df[feature_cols]
    y = cont_df['stay_leave']

    model = SVC(kernel='linear', C=1, gamma=1)
    model.fit(X, y)

    # Extract raw coefficients to solve the 2D line equation
    w = model.coef_[0]
    b = model.intercept_[0]
    entry_coef, reward_coef, block_coef = w[0], w[1], w[2]

    # 3. Setup the Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    blocks = [0.4, 0.8]
    block_names = ['Low Context (0.4)', 'High Context (0.8)']

    # Use Set2 colors for the different contexts
    context_colors = [sns.color_palette('Set2')[0], sns.color_palette('Set2')[1]]

    max_entry = cont_df['time_from_entry'].max()
    max_reward = cont_df['time_from_reward'].max()

    for i, ax in enumerate(axes):
        block_val = blocks[i]

        # Filter data for this specific block
        block_df = cont_df[cont_df['block_value'] == block_val]

        # Separate Stay (0) and Leave (1) points
        stay_df = block_df[block_df['stay_leave'] == 0]
        leave_df = block_df[block_df['stay_leave'] == 1]

        # Plot Scatter Points
        ax.scatter(stay_df['time_from_entry'], stay_df['time_from_reward'],
                   c='steelblue', s=10, alpha=0.3, label='Stay', edgecolors='none')
        ax.scatter(leave_df['time_from_entry'], leave_df['time_from_reward'],
                   c='firebrick', s=15, alpha=0.8, label='Leave', edgecolors='none')

        # Calculate the 2D Decision Boundary Line
        # 0 = w1*x + w2*y + w3*block + b  =>  y = (-w1*x - w3*block - b) / w2
        x_line = np.array([0, max_entry])
        y_line = (-entry_coef * x_line - block_coef * block_val - b) / reward_coef

        # Plot Boundary
        ax.plot(x_line, y_line, color=context_colors[i], linewidth=2.5, linestyle='--',
                label=f'SVM Boundary')

        # Formatting
        ax.set_title(block_names[i], color=context_colors[i], fontweight='bold')
        ax.set_xlabel('Time from Entry (s)')
        if i == 0:
            ax.set_ylabel('Time from Reward (s)')

        # Lock axes to data limits so the line doesn't distort the view
        ax.set_xlim(0, max_entry + 1)
        ax.set_ylim(0, max_reward + 0.5)
        ax.legend(loc='upper right', title='Decision')

    sns.despine()
    plt.tight_layout()

    if save_fig:
        # Save to single-animal figure folder
        save_path = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.FIGURE_SUBDIR)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{animal_id}_session{session_id}_svm_2d.png'), dpi=300)

    plt.show()


def plot_overlaid_svm_boundaries(animal_id, session_id=0, save_fig=False):
    """
    Plots the 2D SVM decision boundaries for both the Low (0.4) and High (0.8)
    blocks on the same axes, without the raw scatter data.
    """
    # 1. Get the continuous binned data
    cont_df = build_continuous_session_data(animal_id, session_id=session_id)
    if cont_df is None or len(cont_df) < 50:
        print(f"Not enough valid data for {animal_id} session {session_id}.")
        return

    # 2. Fit the Unwhitened Model
    feature_cols = ['time_from_entry', 'time_from_reward', 'block_value']
    X = cont_df[feature_cols]
    y = cont_df['stay_leave']

    model = SVC(kernel='linear', C=1, gamma=1)
    model.fit(X, y)

    # Extract raw coefficients to solve the 2D line equations
    w = model.coef_[0]
    b = model.intercept_[0]
    entry_coef, reward_coef, block_coef = w[0], w[1], w[2]

    # 3. Setup the Figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Get max ranges for plotting the lines
    max_entry = cont_df['time_from_entry'].max()
    max_reward = cont_df['time_from_reward'].max()
    x_line = np.array([0, max_entry + 5])  # Extend slightly past max data

    # Use Set2 colors
    color_low = sns.color_palette('Set2')[0]
    color_high = sns.color_palette('Set2')[1]

    # 4. Calculate and Plot the Low Context (0.4) Boundary
    y_line_low = (-entry_coef * x_line - block_coef * 0.4 - b) / reward_coef
    ax.plot(x_line, y_line_low, color=color_low, linewidth=3,
            label='Low Context Boundary (0.4)')

    # 5. Calculate and Plot the High Context (0.8) Boundary
    y_line_high = (-entry_coef * x_line - block_coef * 0.8 - b) / reward_coef
    ax.plot(x_line, y_line_high, color=color_high, linewidth=3,
            label='High Context Boundary (0.8)')

    # Optional: Fill the space between them lightly to emphasize the shift margin
    # ax.fill_between(x_line, y_line_low, y_line_high, color='grey', alpha=0.1)

    # 6. Formatting
    ax.set_title(f'SVM Decision Boundaries by Context\n{animal_id} (Session {session_id})', fontweight='bold')
    ax.set_xlabel('Time from Entry (s)', fontsize=11)
    ax.set_ylabel('Time from Reward (s)', fontsize=11)

    # Lock axes to data limits so we don't zoom out to infinity
    ax.set_xlim(0, max_entry)
    ax.set_ylim(0, max_reward + 0.5)

    ax.legend(loc='upper right', frameon=True)
    sns.despine()
    plt.tight_layout()

    if save_fig:
        # Save to single-animal figure folder
        save_path = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.FIGURE_SUBDIR)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{animal_id}_session{session_id}_svm_boundaries_overlaid.png'), dpi=300)

    plt.show()


if __name__ == '__main__':
    # Use your specific list of animals
    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]

    print("Extracting continuous behavioral matrices and fitting SVMs...")
    svm_coef_df, predictions_df = get_svm_coefficients_and_predictions(animal_ids)

    # print("\n--- Generating Figure 3.1g ---")
    # fig3_1g_svm_coef(svm_coef_df, save_fig=False)

    # print("\n--- Generating Figure 3.1j ---")
    # fig3_1j_svm_summary(predictions_df, save_fig=False)

    print("\n--- Generating decision boundary plot ---")
    example_animal = "SZ036"
    for session_id in range(1, 10):
        plot_2d_svm_projections(example_animal, session_id=session_id, save_fig=False)
        plot_overlaid_svm_boundaries(example_animal, session_id=session_id, save_fig=False)
    # plot_3d_svm_surface(example_animal, session_id=0, save_fig=False)