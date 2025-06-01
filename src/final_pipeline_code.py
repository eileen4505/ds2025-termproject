import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer  # Class for imputing missing values using K-Nearest Neighbors
from sklearn.preprocessing import StandardScaler
from IPython.display import display

# Data loading
try:
    df = pd.read_csv("CMU_Sleep.csv")
except FileNotFoundError:  # Raised when attempting to access a file that does not exist
    print("CSV file not found.")

# Convert columns to numeric types
''' 
If a column contains whitespace (' '), empty strings (''), or string-based values ('NaN', 'null', etc.),
pandas treats the entire column as an object (string) type.
Even if most values are numeric, this prevents numerical analysis such as machine learning, 
statistical operations, or imputation.

To resolve this, the following columns are explicitly converted to float64.
Values that cannot be converted are coerced to NaN for further processing.
'''
columns_to_convert = [
    "demo_race",
    "demo_gender",
    "demo_firstgen",
    "term_units",
    "Zterm_units_ZofZ"
]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check data types after conversion
print("Data types after numeric conversion:")
print(df[columns_to_convert].dtypes)

# Convert the 'cohort' column into one-hot encoded variables
df = pd.get_dummies(df, columns=["cohort"], prefix="cohort")

# Check the resulting one-hot encoded cohort columns
print([col for col in df.columns if col.startswith("cohort_")])

def get_feature_importance(X, y, model, top_n=None, title="Feature Importance", figsize=(8, 5)):
    """
    Trains the given model and outputs & visualizes feature importance.

    Args:
        X (DataFrame): Input features
        y (Series or ndarray): Target variable
        model (sklearn estimator): Model to train
        top_n (int): Display only the top N important features (None to display all)
        title (str): Title of the plot
        figsize (tuple): Size of the figure for visualization

    Returns:
        DataFrame: Table of feature importances
    """
    model.fit(X, y)

    # Extract feature importance
    if hasattr(model, "feature_importances_"):  # Check if the model has the 'feature_importances_' attribute
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # If 'feature_importances_' is not available, check for 'coef_' attribute
        importances = model.coef_.ravel() if len(model.coef_.shape) == 2 else model.coef_
    else:
        raise ValueError("This model does not provide feature importance.")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    if top_n:
        importance_df = importance_df.head(top_n)

    # Visualization
    plt.figure(figsize=figsize)
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return importance_df

# RandomForestRegressor is a model that improves prediction accuracy for regression tasks using an ensemble of decision trees.
from sklearn.ensemble import RandomForestRegressor

# Define the list of features to be used for model training.
features = [
    'TotalSleepTime', 'bedtime_mssd', 'midpoint_sleep', 'daytime_sleep',
    'cum_gpa', 'Zterm_units_ZofZ', 'demo_gender', 'demo_race', 'demo_firstgen',
    'term_units', 'frac_nights_with_data'
]

# Features such as subject_id, study, and cohort are excluded from the predictors to prevent data leakage, overfitting, and to enhance generalizability.

# Create a new DataFrame (df_clean) by selecting only the relevant features and the target variable ('term_gpa'),
# and dropping rows with missing values (NaN).
df_clean = df[features + ['term_gpa']].dropna()

# Define the input features (X) from df_clean based on the selected feature list.
X = df_clean[features]

# Define the target variable (y) from the 'term_gpa' column.
y = df_clean['term_gpa']

# Use the RandomForestRegressor model to calculate feature importances and visualize the results
importance_table = get_feature_importance(
    X,  
    y,  
    model=RandomForestRegressor(random_state=42),
    title="Feature Importance for term_gpa"
)

# Define the list of features to be used for model training.
features = [
    'TotalSleepTime', 'bedtime_mssd', 'midpoint_sleep', 'daytime_sleep',
    'cum_gpa', 'demo_gender', 'demo_race', 'demo_firstgen',
    'term_gpa', 'frac_nights_with_data'
]

# Variables such as subject_id, study, and cohort are excluded from the prediction features 
# to prevent overfitting, ensure generalization, and avoid data leakage.

# Create a new DataFrame (df_clean) by selecting only the relevant features and the target variable ('term_units') from the original DataFrame,
# and removing rows that contain missing values (NaNs).
df_clean = df[features + ['term_units']].dropna()

# Define input features (X) from the df_clean DataFrame using the selected features.
X = df_clean[features]

# Define the target variable (y) from the 'term_units' column in df_clean.
y = df_clean['term_units']

# Use the RandomForestRegressor model to compute feature importances and visualize the results
importance_table = get_feature_importance(
    X,  
    y,  
    model=RandomForestRegressor(random_state=42),
    title="Feature Importance for term_units"
)

def preprocess_missing_values(df: pd.DataFrame, columns_to_convert: list) -> pd.DataFrame:
    """
    Handles missing values in selected columns of the CMU Sleep dataset and prints information about imputed cells.

    Processing strategy:
    - demo_* columns: Impute with the mode value.
    - term_units column: Impute using KNN Imputer (includes the study column).
    - Zterm_units_ZofZ column: Computed using two-step Z-score normalization as described in the dataset guide:
        1) First Z-score within each 'study' group.
        2) Second Z-score across all participants.
    - Prints the location and values of imputed cells for review.

    Args:
        df (pd.DataFrame): DataFrame to perform missing value imputation on.
        columns_to_convert (list): List of target columns to track for missing value imputation.

    Returns:
        pd.DataFrame: The DataFrame with missing values processed.
    """

    # Record missing positions before imputation (for tracking imputed values)
    missing_mask = df[columns_to_convert].isnull()

    # Impute demo_* (demographic) columns with mode value
    for col in ["demo_race", "demo_gender", "demo_firstgen"]:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # Impute 'term_units' using KNN Imputer
    knn_features = [
        'cum_gpa', 'TotalSleepTime', 'daytime_sleep', 'bedtime_mssd',
        'midpoint_sleep', 'term_gpa', 'frac_nights_with_data',
        'term_units'
    ]
    knn_df = df[knn_features].copy()

    imputer = KNNImputer(n_neighbors=5)
    knn_result = imputer.fit_transform(knn_df)
    df["term_units"] = knn_result[:, knn_features.index("term_units")]

    # Round term_units to integer values after imputation
    df["term_units"] = np.round(df["term_units"]).astype(int)

    # Compute two-step Z-score for 'Zterm_units_ZofZ'
    df["Z1"] = (
        df["term_units"] - df.groupby("study")["term_units"].transform("mean")
    ) / df.groupby("study")["term_units"].transform("std")

    z1_mean = df["Z1"].mean()
    z1_std = df["Z1"].std()
    df["Zterm_units_ZofZ"] = (df["Z1"] - z1_mean) / z1_std

    df.drop(columns=["Z1"], inplace=True)

    # Print remaining missing values
    print("Remaining missing values after processing:")
    print(df.isnull().sum())

    # Show example of imputed cells
    filled_values = []
    for col in columns_to_convert:
        for idx in missing_mask.index[missing_mask[col]]:
            value = df.loc[idx, col]
            filled_values.append({"index": idx, "column": col, "filled_value": value})

    filled_df = pd.DataFrame(filled_values)
    print("\nExamples of imputed values (up to 20):")
    print(filled_df.head(20))

    # Compare mean of imputed 'term_units' values to overall mean
    filled_term_units = filled_df[filled_df["column"] == "term_units"]["filled_value"].astype(float)
    overall_mean = df["term_units"].mean()

    print("\n'term_units' column:")
    print(f"  Mean of imputed values: {filled_term_units.mean():.2f}")
    print(f"  Overall mean: {overall_mean:.2f}")

    return df

df = preprocess_missing_values(df, columns_to_convert)

# Replace invalid value 2 in the 'demo_firstgen' column with the mode value

# Calculate the mode (most frequent value) among valid values (0 or 1)
valid_mode = df.loc[df['demo_firstgen'].isin([0, 1]), 'demo_firstgen'].mode().iloc[0]

# Replace any instance of the invalid value 2 with the valid mode
df.loc[df['demo_firstgen'] == 2, 'demo_firstgen'] = valid_mode

# Display the value distribution after correction
print("Distribution of 'demo_firstgen' values:")
print(df['demo_firstgen'].value_counts())

# List of target features for outlier detection
outlier_features = [ 
    "cum_gpa", "TotalSleepTime", "daytime_sleep", "bedtime_mssd", 
    "Zterm_units_ZofZ", "midpoint_sleep"
]

def detect_outliers_zscore(df, features, threshold=2.5):
    """
    Detect outliers for the specified features using the Z-score method.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        features (list): A list of feature (column) names to check for outliers.
        threshold (float): Absolute Z-score threshold for flagging an outlier.

    Returns:
        dict: A dictionary mapping each feature to a DataFrame containing the 
              indices, feature name, original value, and Z-score of detected outliers.
    """
    outliers_summary = {}

    for col_name in features:
        col_mean = df[col_name].mean()
        col_std = df[col_name].std()
        z_scores = (df[col_name] - col_mean) / col_std
        is_outlier_condition = z_scores.abs() > threshold

        detected_outliers_df = df.loc[is_outlier_condition, [col_name]]
        detected_outliers_df["index"] = detected_outliers_df.index
        detected_outliers_df["z_score"] = z_scores[is_outlier_condition]
        detected_outliers_df["feature"] = col_name
        detected_outliers_df.rename(columns={col_name: "value"}, inplace=True)

        outliers_summary[col_name] = detected_outliers_df[["index", "feature", "value", "z_score"]]

    return outliers_summary

# Run the function
outliers_by_feature = detect_outliers_zscore(df, outlier_features)

def visualize_and_report_outliers(df: pd.DataFrame, target_gpa_column: str, outlier_info_dict: dict, feature_list: list):
    """
    Visualize outliers for each feature and display the summary of detected outliers based on Z-score.

    Args:
        df (pd.DataFrame): Original DataFrame.
        target_gpa_column (str): Name of the target GPA column.
        outlier_info_dict (dict): Dictionary containing Z-score-based outlier information for each feature.
        feature_list (list): List of features to analyze.
    """
    print(f"\nTarget variable: {target_gpa_column}\n")

    total_outliers = 0  # Counter to keep track of total outliers

    # Add a flag column for outlier status
    df["OutlierFlag"] = "Non-Outlier"

    for feature in feature_list:
        outlier_idx = outlier_info_dict[feature]["index"]

        # Mark detected outliers in the flag column
        df.loc[df.index.isin(outlier_idx), "OutlierFlag"] = "Outlier"

        # Prepare data for visualization
        plot_df = df[[feature, target_gpa_column, "OutlierFlag"]].dropna()

        # Scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=plot_df,
            x=feature, y=target_gpa_column,
            hue="OutlierFlag", style="OutlierFlag",
            palette={"Non-Outlier": "skyblue", "Outlier": "salmon"},
            markers={"Non-Outlier": "o", "Outlier": "X"},
            s=80, alpha=0.7
        )
        sns.regplot(
            data=plot_df[plot_df["OutlierFlag"] == "Non-Outlier"],
            x=feature,
            y=target_gpa_column,
            scatter=False,  # Only draw the regression line (scatter already drawn)
            line_kws={"color": "steelblue", "linestyle": "--"}
        )
        plt.title(f'{feature} vs. {target_gpa_column}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Output detected outlier information
        result_df = outlier_info_dict[feature]
        count = len(result_df)
        total_outliers += count

        print(f"\n [ {feature} Outlier Detection Result (Z-score, threshold=2.5) ]")
        print(f"Number of outliers: {count}")
        if count > 0:
            # Merge visualization outliers with Z-score data
            outlier_rows = plot_df[plot_df["OutlierFlag"] == "Outlier"]
            merged = outlier_rows.merge(result_df, left_index=True, right_on="index")
            print("Outlier examples (top 5):")
            print(merged[[feature, "z_score", target_gpa_column]].head(5).to_string(index=False))
        else:
            print("No outliers detected.")
        print("-" * 60)

    print(f"\nTotal number of detected outliers across all features: {total_outliers} (based on Z-score)\n")

# Run visualization and reporting
visualize_and_report_outliers(
    df=df,
    target_gpa_column="term_gpa",
    outlier_info_dict=outliers_by_feature,
    feature_list=outlier_features
)

def process_and_visualize_feature_outliers(
    df_original,
    feature_to_process,
    all_major_features,
    outlier_info_for_feature,
    target_gpa_col="term_gpa",
    impute_condition_logic=None,
    knn_k=5,
    visualize=True
    ):
    """
    Identifies and imputes outliers for a specific feature using KNNImputer,
    based on a logical condition related to the GPA target variable,
    and optionally visualizes the results.

    Args:
        df_original (pd.DataFrame): Original input DataFrame
        feature_to_process (str): Name of the feature to impute
        all_major_features (list): List of features to use for KNN training
        outlier_info_for_feature (pd.DataFrame): Z-score-based outlier info for this feature
        target_gpa_col (str): Target variable for GPA (default: "term_gpa")
        impute_condition_logic (callable): Function that defines the logic to decide which outliers to impute
        knn_k (int): Number of neighbors for KNN imputation
        visualize (bool): Whether to visualize imputation results
    Returns:
        pd.DataFrame: A new DataFrame with outliers imputed (original remains unchanged)
    """
    
    df_processed = df_original.copy()  # Copy the original DataFrame to avoid modifying it

    # Merge the outlier info with the actual GPA values for further condition-based filtering
    merged_info = pd.merge(outlier_info_for_feature, df_processed[[target_gpa_col]], left_on='index', right_index=True)

    # Apply the logic function to determine which outliers to impute
    indices_to_impute = merged_info.loc[impute_condition_logic(merged_info, target_gpa_col), 'index'].tolist()

    if not indices_to_impute:
        print(f"  [{feature_to_process}] No outliers matched the imputation condition.")
        print("-" * 70)
        return df_processed

    print(f"  [{feature_to_process}] {len(indices_to_impute)} outliers identified for imputation.")
    print(f"  Values before imputation:\n{df_processed.loc[indices_to_impute, [feature_to_process, target_gpa_col]]}")

    # Create a temporary copy and mark the selected outliers as NaN
    df_temp = df_processed.copy()
    df_temp.loc[indices_to_impute, feature_to_process] = np.nan

    # Explicitly define the features to use for KNN imputation
    knn_features = [
        "TotalSleepTime", "cum_gpa", "midpoint_sleep", "daytime_sleep",
        "Zterm_units_ZofZ", "bedtime_mssd", "term_units", "frac_nights_with_data",
        "term_gpa"
    ]
    data_for_imputer = df_temp[knn_features]

    # Standardize the data before applying KNN (as it is scale-sensitive)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_imputer)
    imputer = KNNImputer(n_neighbors=knn_k)
    imputed_scaled_data = imputer.fit_transform(scaled_data)

    # Restore original scale after imputation
    recovered_data = scaler.inverse_transform(imputed_scaled_data)

    imputed_df = pd.DataFrame(
        recovered_data,
        columns=knn_features,
        index=data_for_imputer.index
    )

    # Update the imputed values in the copied DataFrame
    df_processed.loc[indices_to_impute, feature_to_process] = imputed_df.loc[indices_to_impute, feature_to_process]

    print(f"  [{feature_to_process}] Values after KNN imputation:\n{df_processed.loc[indices_to_impute, [feature_to_process, target_gpa_col]]}")

    
    # Visualization (Before/After Imputation)
    if visualize:
        plt.figure(figsize=(8, 6))  

        # Non-outliers (Original): Original data points that are not imputed 
        non_imputed_original_data = df_original.drop(index=indices_to_impute, errors='ignore')
        sns.scatterplot(
            data=non_imputed_original_data,
            x=feature_to_process, y=target_gpa_col,
            color="skyblue",      # Color for non-outliers (background)
            marker="o",           # Circular marker
            s=60,                 # Dot size (slightly smaller for background feel)
            alpha=0.5,            # Transparency (lighter for background)
            edgecolor='white',    # Dot border color
            label="Non-Imputed (Original)"  # Legend label
        )

        # Original Outlier Values: Outliers before imputation (red X)
        sns.scatterplot(
            x=df_original.loc[indices_to_impute, feature_to_process],
            y=df_original.loc[indices_to_impute, target_gpa_col],  # GPA remains unchanged
            color="salmon",       # Outlier color
            marker="X",           # X-shaped marker
            s=100,                # Dot size (highlighted)
            alpha=0.9,            # Less transparent for visibility
            edgecolor='darkred',  # Border color for X marker
            linewidth=1.5,        # Border width for X
            label="Original Outlier Value"  # Legend label
        )

        # Imputed Values: New values imputed using KNN (blue O)
        sns.scatterplot(
            x=df_processed.loc[indices_to_impute, feature_to_process],
            y=df_processed.loc[indices_to_impute, target_gpa_col],  # GPA is the same
            color="dodgerblue",   # Color for imputed values
            marker="o",           # Circle marker (distinguished by color/border)
            s=100,                # Same size as outliers for comparison
            alpha=0.9,
            edgecolor='navy',     # Distinct border color
            linewidth=1.5,
            label="Imputed Value"  # Legend label
        )

        # Regression trend line for non-outlier data
        if not non_imputed_original_data.empty:  # Plot only if non-imputed data exists
            sns.regplot(
                data=non_imputed_original_data,
                x=feature_to_process, y=target_gpa_col,
                scatter=False,  # Skip scatter as it's already plotted
                line_kws={"color": "steelblue", "linestyle": "--", "linewidth": 2}  # Regression line style
                # Note: regplot does not auto-add to legend
            )

        plt.title(f"[{feature_to_process}] vs {target_gpa_col}: Outlier Imputation Impact", fontsize=14)
        plt.xlabel(feature_to_process, fontsize=12)
        plt.ylabel(target_gpa_col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)  # Show grid
        plt.legend(title="Data Points", loc='best', frameon=True, fontsize=9)  # Configure legend
        plt.tight_layout()  # Auto layout adjustment
        plt.show()  # Display the plot

    print("-" * 70)
    return df_processed

# Define imputation logic for each feature

def cum_gpa_impute_logic(merged_info_df, gpa_col):
    # If cumulative GPA is very low (z < -2.5) but term GPA is 2.5 or higher, it contradicts the typical trend → mark for imputation
    return (merged_info_df['z_score'] < -2.5) & (merged_info_df[gpa_col] >= 2.5)

def total_sleep_time_impute_logic(merged_info_df, gpa_col):
    # If total sleep time is either very low (z < -2.5) or very high (z > 2.5) and term GPA is 2.5 or higher → mark for imputation
    condition_low_sleep_high_gpa = (merged_info_df['z_score'] < -2.5) & (merged_info_df[gpa_col] >= 2.5)
    condition_high_sleep_high_gpa = (merged_info_df['z_score'] > 2.5) & (merged_info_df[gpa_col] >= 2.5)
    return condition_low_sleep_high_gpa | condition_high_sleep_high_gpa

def daytime_sleep_impute_logic(merged_info_df, gpa_col):
    # If daytime sleep is very high (z > 2.5) and term GPA is 2.5 or higher → potential contradiction with typical sleep-academic performance patterns
    return (merged_info_df['z_score'] > 2.5) & (merged_info_df[gpa_col] >= 2.5)

def bedtime_mssd_impute_logic(merged_info_df, gpa_col):
    # If bedtime variability is very high (z > 2.5) and term GPA is 2.5 or higher → mark as exceptional case for imputation
    return (merged_info_df['z_score'] > 2.5) & (merged_info_df[gpa_col] >= 2.5)

def zterm_units_zofz_impute_logic(merged_info_df, gpa_col):
    # If normalized course load is very low (z < -2.5) and term GPA is 2.5 or higher,
    # or if it's very high (z > 2.5) and term GPA is 2.0 or lower → mark for imputation due to mismatch
    cond1 = (merged_info_df['z_score'] < -2.5) & (merged_info_df[gpa_col] >= 2.5)
    cond2 = (merged_info_df['z_score'] > 2.5) & (merged_info_df[gpa_col] <= 2.0)
    return cond1 | cond2

def midpoint_sleep_impute_logic(merged_info_df, gpa_col):
    # If sleep midpoint is extremely late (z > 2.5) and term GPA is 2.5 or higher → may contradict typical behavior-performance pattern
    return (merged_info_df['z_score'] > 2.5) & (merged_info_df[gpa_col] >= 2.5)


# Map each feature to its corresponding conditional outlier logic
features_to_process_map = {
    "cum_gpa": cum_gpa_impute_logic,
    "TotalSleepTime": total_sleep_time_impute_logic,
    "daytime_sleep": daytime_sleep_impute_logic,
    "bedtime_mssd": bedtime_mssd_impute_logic,
    "Zterm_units_ZofZ": zterm_units_zofz_impute_logic,
    "midpoint_sleep": midpoint_sleep_impute_logic,
}

df_final_processed = df.copy()  # Start with a copy of the original DataFrame

for feature_name in outlier_features:
    # Proceed only if the feature has a defined imputation logic
    if feature_name in features_to_process_map:
        current_outlier_info = outliers_by_feature[feature_name]
        logic_function_for_feature = features_to_process_map[feature_name]

        df_final_processed = process_and_visualize_feature_outliers(
            df_original=df_final_processed,
            feature_to_process=feature_name,
            all_major_features=outliers_by_feature,
            outlier_info_for_feature=current_outlier_info,
            target_gpa_col="term_gpa",
            impute_condition_logic=logic_function_for_feature,
            knn_k=5,
            visualize=True
        )

# Create derived features based on the final cleaned dataset
# These features aim to better represent the characteristics of the data and improve model performance

# Derived Feature 1: Sleep Consistency Index
# Definition: (TotalSleepTime) / (bedtime_mssd + 1)
# Purpose: Combines the quantity and regularity of sleep into one indicator
#          TotalSleepTime: Total sleep duration during the main sleep window (minutes)
#          bedtime_mssd: Mean squared successive difference of bedtime (in minutes). Higher values indicate more irregular sleep times.
#          Adding 1 to the denominator prevents division by zero (in case of perfect regularity),
#          and amplifies the index when bedtime variability is low to reflect positive effects of regularity.
# Interpretation: Higher values imply longer and more regular sleep.
df_final_processed['sleep_consistency_index'] = df_final_processed['TotalSleepTime'] / (df_final_processed['bedtime_mssd'] + 1)

# Derived Feature 2: Sleep Efficiency Relative to Academic Load (sleep_efficiency_z)
# Definition: (TotalSleepTime + daytime_sleep) / (np.exp(Zterm_units_ZofZ) + epsilon)
# Purpose: Measures total sleep input (main + nap) relative to academic burden (Z-score of term units)
#          np.exp is used to ensure the denominator is positive and reduce interpretability issues when Z is near or below zero
#          epsilon is added to avoid division by zero in rare cases where Zterm_units_ZofZ ≈ -∞
epsilon = 1e-6
df_final_processed['sleep_efficiency_z'] = \
    (df_final_processed['TotalSleepTime'] + df_final_processed['daytime_sleep']) / \
    (np.exp(df_final_processed['Zterm_units_ZofZ']) + epsilon)

# Derived Feature 3: Daytime Sleep Proportion (day_night_ratio)
# Definition: daytime_sleep / (TotalSleepTime + daytime_sleep)
# Purpose: Indicates how much of total sleep is taken during the day (nap). Higher values suggest greater dependency on daytime sleep.
total_sleep_for_ratio = df_final_processed['TotalSleepTime'] + df_final_processed['daytime_sleep']
df_final_processed['day_night_ratio'] = np.where(
    total_sleep_for_ratio > 0,
    df_final_processed['daytime_sleep'] / total_sleep_for_ratio,
    0
)

# Derived Feature 4: Sleep Debt
# Definition: np.maximum(0, RECOMMENDED_SLEEP_MINUTES - TotalSleepTime)
# Purpose: Captures how much the student is under-sleeping relative to a recommended amount (e.g., 7 hours = 420 minutes)
RECOMMENDED_SLEEP_MINUTES = 420
df_final_processed['sleep_debt'] = np.maximum(0, RECOMMENDED_SLEEP_MINUTES - df_final_processed['TotalSleepTime'])

# Preview the newly created derived features (top 5 rows)
created_derived_features = [
    'sleep_consistency_index', 'sleep_efficiency_z', 
    'day_night_ratio', 'sleep_debt'
]
print("\nPreview of derived features (top 5 rows):")
print(df_final_processed[created_derived_features].head())

# Optionally display descriptive statistics for the derived features
print("\nDescriptive statistics of derived features:")
print(df_final_processed[created_derived_features].describe())

# Full list of features including derived variables
features = [
    'TotalSleepTime', 'bedtime_mssd', 'midpoint_sleep', 'daytime_sleep',
    'cum_gpa', 'Zterm_units_ZofZ', 'demo_gender', 'demo_race', 'demo_firstgen',
    'term_units', 'frac_nights_with_data',

    # Added engineered features
    'sleep_consistency_index',
    'sleep_efficiency_z',
    'day_night_ratio',
    'sleep_debt'
]

# Prepare data by dropping rows with missing values
df_clean = df_final_processed[features + ['term_gpa']].dropna()
X = df_clean[features]
y = df_clean['term_gpa']

# Call feature importance visualization function
importance_table = get_feature_importance(
    X, y,
    model=RandomForestRegressor(random_state=42),
    title="Feature Importance for term_gpa (with engineered features)"
)

from sklearn.preprocessing import RobustScaler

# Sleep-related features to be scaled
sleep_features_to_scale = [
    'TotalSleepTime', 'bedtime_mssd', 'midpoint_sleep', 'daytime_sleep',
    'cum_gpa', 'term_units', 'Zterm_units_ZofZ',

    # Additional engineered features
    'sleep_consistency_index',
    'sleep_efficiency_z',
    'day_night_ratio',
    'sleep_debt'
]

# Apply RobustScaler
scaler = RobustScaler()
df_final_processed[sleep_features_to_scale] = scaler.fit_transform(df_final_processed[sleep_features_to_scale])

# Check the result
print(df_final_processed[sleep_features_to_scale].head())

# Define GPA binning thresholds and corresponding labels
bins = [0.0, 3.3, 3.7, 4.0]
labels = ["Low", "Mid", "High"]

# Assign GPA class labels based on term_gpa ranges
df_final_processed["gpa_class"] = pd.cut(df_final_processed["term_gpa"], bins=bins, labels=labels, include_lowest=True)

# Check the distribution of GPA classes
print(df_final_processed["gpa_class"].value_counts())

# Save the currently processed DataFrame to a CSV file
df_final_processed.to_csv("preprocessed_cmu_sleep.csv", index=False, encoding="utf-8-sig")

# --- Modelling Section ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    GridSearchCV,
    cross_val_predict,    # cross_val_predict: function that returns predictions for each cross-validation fold
    learning_curve,       # learning_curve: function that computes training and validation scores as training size grows
    KFold
)
from sklearn.pipeline import Pipeline           # Pipeline: links preprocessing steps to model training
from sklearn.tree import DecisionTreeClassifier # DecisionTreeClassifier: decision tree classification model
from sklearn.neighbors import KNeighborsClassifier  # KNeighborsClassifier: K-nearest neighbors classification model
from sklearn.linear_model import LinearRegression   # LinearRegression: basic linear regression model
from sklearn.preprocessing import PolynomialFeatures # PolynomialFeatures: expands features into polynomial terms
from sklearn.metrics import (
    confusion_matrix,          # confusion_matrix: computes the confusion matrix
    ConfusionMatrixDisplay     # ConfusionMatrixDisplay: tool to plot the confusion matrix
)


# [0] Load data and apply common preprocessing
# -------------------------------------------------------------------
df = pd.read_csv("preprocessed_cmu_sleep.csv")  # Load the preprocessed CSV data
df = df[df["frac_nights_with_data"] >= 0.8]      # Keep rows where ≥80% of nights have sleep data

# Features for classification (X_clf, y_clf) and regression (X_reg, y_reg)
feature = [
    "demo_gender",    # gender (0 = male, 1 = female)
    "demo_race",      # race code (0 = minority group, 1 = non-minority)
    "demo_firstgen",  # first-generation status (0 = not first-gen, 1 = first-gen)
    "TotalSleepTime", # total sleep time in minutes
    "bedtime_mssd",   # bedtime variability in minutes
    "midpoint_sleep", # midpoint of sleep period in minutes
    "daytime_sleep",  # daytime sleep duration in minutes
    "Zterm_units_ZofZ", # GPA converted to Z-score
]
X_clf, y_clf = df[feature], df["gpa_class"]  # Classification target: gpa_class (A/B/C/D)
X_reg, y_reg = df[feature], df["term_gpa"]   # Regression target: term_gpa (scale up to 4.0)


# Calculate and print the percentage of each class in y_clf
percentages = y_clf.value_counts(normalize=True) * 100
print(percentages)

# Exclude sleep variables
feature = [
    "demo_gender",    # gender (0 = male, 1 = female)
    "demo_race",      # race code (0 = minority group, 1 = non-minority)
    "demo_firstgen",  # first-generation status (0 = not first-gen, 1 = first-gen)
]
X_clf, y_clf = df[feature], df["gpa_class"]  # Classification target: gpa_class (A/B/C/D)
X_reg, y_reg = df[feature], df["term_gpa"]   # Regression target: term_gpa (scale up to 4.0)

# Only sleep data
feature = [
    "TotalSleepTime",          # total sleep time in minutes
    "bedtime_mssd",            # bedtime variability in minutes
    "midpoint_sleep",          # midpoint of sleep period in minutes
    "daytime_sleep",           # daytime sleep duration in minutes
    "sleep_consistency_index", # sleep consistency index
    "sleep_efficiency_z",      # sleep efficiency relative to academic load
    "day_night_ratio",         # proportion of daytime sleep
    "sleep_debt"               # sleep debt
]
X_clf, y_clf = df[feature], df["gpa_class"]  # Classification target: gpa_class (A/B/C/D)
X_reg, y_reg = df[feature], df["term_gpa"]   # Regression target: term_gpa (scale up to 4.0)


# [1] classification function
# -------------------------------------------------------------------
def def_clf(model, param_grid, k_values=[3,5,7]):
    """
    model       : model instance
    param_grid  : parameters for GridSearchCV
    k_values    : number of folds for KFold
    """

    # 1) Run GridSearchCV for each k in k_values and save results
    records = []
    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        gs = GridSearchCV(
            Pipeline([("model", model)]),  # Pipeline: ('model', model object)
            param_grid,
            cv=cv,
            scoring="accuracy",            # scoring metric: accuracy
            n_jobs=-1,                     # use all CPU cores
            verbose=1                      # print progress
        )
        gs.fit(X_clf, y_clf)              # fit the model and perform CV

        # convert results to DataFrame
        dfcv = pd.DataFrame(gs.cv_results_)
        dfcv["k"] = k                   # record which k produced these results
        records.append(dfcv)

    # 2) Combine results across k and rename columns
    dfall = pd.concat(records, ignore_index=True)
    dfall = dfall.rename(columns={
        "mean_test_score": "accuracy",  # average accuracy
    })

    # 3) Sort by accuracy descending and extract top 5
    dftop = dfall.sort_values("accuracy", ascending=False).head(5)

    # 4) Print top 5
    print(f"\n[{model.__class__.__name__}] Top-5 across k={k_values} (CV Accuracy)")
    print(dftop[["k","accuracy","params"]])

    # 5) The first row among the top 5 is the final best
    best = dftop.iloc[0]
    best_k, best_acc, best_params = int(best["k"]), best["accuracy"], best["params"]
    print(f"\n→ Final Best: k={best_k}, accuracy={best_acc:.4f}, params={best_params}")

    # 6) Plot confusion matrix for the best model
    best_pipe = Pipeline([("model", model)])
    best_pipe.set_params(**best_params) # set parameters found by grid search
    best_pipe.fit(X_clf, y_clf)         # retrain on full data to get classes
    cvb = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=42)
    ypred = cross_val_predict(best_pipe, X_clf, y_clf, cv=cvb)
    cm = confusion_matrix(y_clf, ypred,
                          labels=best_pipe.named_steps["model"].classes_)
    disp = ConfusionMatrixDisplay(cm,
             display_labels=best_pipe.named_steps["model"].classes_)
    disp.plot()
    disp.ax_.set_title(f"{model.__class__.__name__} CM (k={best_k})")

    # 7) Plot the learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe,             # model to evaluate: best model
        X_clf, y_clf,          # input data and labels
        cv=cvb,                # StratifiedKFold
        scoring="accuracy",    # scoring metric: accuracy
        train_sizes=np.linspace(0.1, 1.0, 5),  # training sample proportions
        n_jobs=-1
    )

    # axis=1: group fold scores for each training proportion
    train_mean = train_scores.mean(axis=1)  # training set accuracy
    train_std  = train_scores.std(axis=1)   # training set standard deviation
    val_mean   = val_scores.mean(axis=1)    # validation set accuracy
    val_std    = val_scores.std(axis=1)     # validation set standard deviation

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train Acc")  # plot training accuracy
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2)  # shade area by one standard deviation
    plt.plot(train_sizes, val_mean, label="Valid Acc")   # plot validation accuracy
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.2)  # shade area by one standard deviation
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title(f"{model.__class__.__name__} Learning Curve (k={best_k})")
    plt.legend()
    plt.show()

    return dftop  # return the top-5 DataFrame


# 7) Set params for each model
# DecisionTree params setup
dt_params = {
    "model__criterion":        ["gini","entropy"],
    "model__max_depth":        [None,5,10,20],
    "model__min_samples_split":[2,5,10],
    "model__min_samples_leaf": [1,2,4]
}
# KNN params setup
knn_params = {
    "model__n_neighbors": list(range(1,51)),
    "model__weights":     ["uniform","distance"],
    "model__p":           [1,2]
}

# 8) Call function for each model
# call function (DecisionTree)
dt = def_clf(DecisionTreeClassifier(random_state=42),
             dt_params, k_values=[3,5,7])
# call function (knn)
knn = def_clf(KNeighborsClassifier(),
              knn_params, k_values=[3,5,7])

from sklearn.metrics import mean_squared_error

# [2] regression function
# -------------------------------------------------------------------
def def_reg(model, param_grid, k_values=[3,5,7]):
    """
    model       : regression model instance
    param_grid  : parameter grid for GridSearchCV
    k_values    : list of KFold splits to use for the learning curve
    """
    # 1) Run GridSearchCV for each k in k_values
    records = []
    for k in k_values:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        gs = GridSearchCV(
            Pipeline([("model", model)]),
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_reg, y_reg)

        # convert results to DataFrame and compute RMSE
        dfcv = pd.DataFrame(gs.cv_results_)
        dfcv["rmse"] = np.sqrt(-dfcv["mean_test_score"])
        dfcv["k"]    = k
        records.append(dfcv)

    # 2) Combine results across k and extract top 3
    dfall = pd.concat(records, ignore_index=True)
    dftop = dfall.sort_values("rmse").head(3)

    # 3) Print top 3
    print(f"\n[{model.__class__.__name__}] Top-3 across k={k_values} (CV RMSE)")
    print(dftop[["k","rmse","params"]])

    # 4) Final best: k, RMSE, params
    best = dftop.iloc[0]
    best_k, best_rmse, best_params = int(best["k"]), best["rmse"], best["params"]
    print(f"\n→ Final Best: k={best_k}, RMSE={best_rmse:.4f}, params={best_params}")

    # 5) Train on full data with best parameters
    best_pipe = Pipeline([("model", model)])
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_reg, y_reg)

    # 6) Plot learning curve
    cvb = KFold(n_splits=best_k, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe,            # model to evaluate: best model
        X_reg, y_reg,         # input data and true labels
        cv=cvb,               # KFold cross-validator
        scoring="neg_mean_squared_error", # scoring metric: negative MSE
        train_sizes=np.linspace(0.1, 1.0, 5), # proportions of training samples
        n_jobs=-1
    )

    # axis=1: group scores from folds for each training proportion
    train_rmse = np.sqrt(-train_scores.mean(axis=1))  # training RMSE
    val_rmse   = np.sqrt(-val_scores.mean(axis=1))    # validation RMSE
    train_std  = np.sqrt(train_scores.std(axis=1))    # standard deviation for training
    val_std    = np.sqrt(val_scores.std(axis=1))      # standard deviation for validation

    plt.figure()
    plt.plot(train_sizes * len(X_reg), train_rmse, label="Train RMSE")  # plot training RMSE
    plt.fill_between(
        train_sizes * len(X_reg),
        train_rmse - train_std,
        train_rmse + train_std,
        alpha=0.2)  # shade area by one standard deviation
    plt.plot(train_sizes * len(X_reg), val_rmse, label="Valid RMSE")   # plot validation RMSE
    plt.fill_between(
        train_sizes * len(X_reg),
        val_rmse - val_std,
        val_rmse + val_std,
        alpha=0.2)  # shade area by one standard deviation
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title(f"{model.__class__.__name__} Learning Curve (k={best_k})")
    plt.ylim(0, 2)
    plt.legend()
    plt.show()

    return dftop  # return top-3 DataFrame


# 4) Call function (LinearRegression)
lr_params = {"model__fit_intercept": [True]}
lr = def_reg(LinearRegression(), lr_params, k_values=[3,5,7])

# [3] function for polynomial regression
# -------------------------------------------------------------------
def def_poly(degrees):
    """
    degrees : the polynomial degrees to search
    """

    # 1) run GridSearchCV
    loo = LeaveOneOut()
    gs  = GridSearchCV(
        Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),  # expand features by the specified degree
            ("model", LinearRegression())
        ]),
        {"poly__degree": degrees}, # specify candidate polynomial degrees
        cv=loo,
        scoring="neg_mean_squared_error", # returns negative MSE; lower MSE is better
        n_jobs=-1,                    # processes run in parallel; -1 uses all cores
        verbose=1                     # whether to print progress
    )
    gs.fit(X_reg, y_reg)        # train the model
    dfcv = pd.DataFrame(gs.cv_results_)       # convert results to a DataFrame

    # 2) convert negative MSE to RMSE
    dfcv["rmse"] = np.sqrt(-dfcv["mean_test_score"])

    # 3) print top 5 by RMSE
    print("\n[PolynomialRegression] degree별 RMSE")
    print(
        dfcv[["param_poly__degree","rmse"]]
        .rename(columns={"param_poly__degree":"degree"})
        .sort_values("degree")
        .to_string(index=False)
    )

    # 4) print best result
    best_d = gs.best_params_["poly__degree"]
    best_r = np.sqrt(-gs.best_score_)
    print(f"\n→ 최종 Best: degree={best_d}, RMSE={best_r:.4f}")

    # 2) build pipeline to retrain on full data with best degree
    best_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=best_d, include_bias=False)),
        ("model", LinearRegression())
    ])
    best_pipe.fit(X_reg, y_reg)

    # 3) calculate learning curve (RMSE for regression)
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe,
        X_reg, y_reg,
        cv=cv5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    # 4) convert to RMSE and standard deviation
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse   = np.sqrt(-val_scores.mean(axis=1))
    train_std  = np.sqrt(train_scores.std(axis=1))
    val_std    = np.sqrt(val_scores.std(axis=1))

    # 5) visualize learning curve
    plt.figure()
    plt.plot(train_sizes * len(X_reg), train_rmse, label="Train RMSE")
    plt.fill_between(train_sizes * len(X_reg),
                     train_rmse - train_std,
                     train_rmse + train_std,
                     alpha=0.2)
    plt.plot(train_sizes * len(X_reg), val_rmse, label="Valid RMSE")
    plt.fill_between(train_sizes * len(X_reg),
                     val_rmse - val_std,
                     val_rmse + val_std,
                     alpha=0.2)

    # fix y-axis from 0 to 2
    plt.ylim(0, 2)

    # set 5 ticks on y-axis (0.0, 0.5, 1.0, 1.5, 2.0)
    plt.yticks(np.linspace(0, 2, 5))

    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title(f"PolynomialRegression (degree={best_d}) Learning Curve")
    plt.legend()
    plt.show()

    return dfcv.sort_values("rmse").head(5)


# 5) call function (Polynomial regression)
poly = def_poly(degrees=[2,3,4])

from sklearn.inspection import permutation_importance

# Tool to measure feature importance
# Measure importance by shuffling features and observing change in prediction error

def P_importance(pipeline, X, y, feature_names,
                                scoring, title,
                                n_repeats=100, random_state=42
                                ):
    """
    Calculate permutation_importance,
    normalize into a DataFrame and display as a styled bar chart.

    pipeline      : already fitted Pipeline
    X, y          : feature/target data
    feature_names : list of feature names
    scoring       : scoring parameter for permutation_importance
    title         : title string for output
    n_repeats     : number of repeats
    random_state  : random seed
    """
    # 1) compute importance
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    # 2) convert to DataFrame
    imp = result.importances_mean  # get average importances


    df_imp = pd.DataFrame({
        "feature":        feature_names,  # feature_names: feature names
        "importance":     imp            # permutation importance score for each feature
    })

    # 3) Min-Max normalization
    df_imp["importance_norm"] = (
        df_imp["importance"] - df_imp["importance"].min()
    ) / (
        df_imp["importance"].max() - df_imp["importance"].min()
    )

    # 4) print results
    print(f"\n[{title}]")

    # 5) styled bar chart
    display(
        df_imp.style.bar(
            color="#FFA07A",
            subset=["importance_norm"],
            align=0
        )
    )

    return df_imp

# 1) Decision Tree
dt_pipe = Pipeline([("model", DecisionTreeClassifier(random_state=42))])
dt_pipe.set_params(**dt.iloc[0]["params"]).fit(X_clf, y_clf)
df_dt_imp = P_importance(
    dt_pipe, X_clf, y_clf, feature,
    scoring="accuracy",
    title="Decision Tree Feature Importances"
)

# 2) KNN
knn_pipe = Pipeline([("model", KNeighborsClassifier())])
knn_pipe.set_params(**knn.iloc[0]["params"]).fit(X_clf, y_clf)
df_knn_imp = P_importance(
    knn_pipe, X_clf, y_clf, feature,
    scoring="accuracy",
    title="KNN Feature Importances"
)

# 3) Linear Regression
lr_pipe = Pipeline([("model", LinearRegression())])
lr_pipe.set_params(**lr.iloc[0]["params"]).fit(X_reg, y_reg)
df_lr_imp = P_importance(
    lr_pipe, X_reg, y_reg, feature,
    scoring="neg_mean_squared_error",
    title="LinearRegression Feature Importances"
)


# 4) Polynomial Regression
poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression())
])
poly_pipe.set_params(poly__degree=int(poly.iloc[0]["param_poly__degree"])) \
         .fit(X_reg, y_reg)
df_poly_imp = P_importance(
    poly_pipe, X_reg, y_reg, feature,
    scoring="neg_mean_squared_error",
    title="Polynomial Regression Feature Importances"
)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

# 0) parameter grid for ensemble models
# RandomForest
rf_params = {
    # number of trees
    "model__n_estimators":    [100, 200, 500],
    # maximum tree depth, set low to prevent overfitting
    "model__max_depth":       [3, 5, 8],
    # minimum samples per leaf, set high to prevent overfitting
    "model__min_samples_leaf":[5, 10, 20],
    # number of features to consider at each split
    "model__max_features":    [0.3, 0.5, "sqrt"]
}

# GradientBoosting
gb_params = {
    # learning rate: controls correction of previous errors, set low to prevent overfitting
    "model__learning_rate": [0.05, 0.1],
    # number of trees
    "model__n_estimators":  [100, 200],
    # maximum tree depth, set low to prevent overfitting
    "model__max_depth":     [3, 5],
    # fraction of training data sampled at each step, set <0.8 to prevent overfitting
    "model__subsample":     [0.6, 0.8]
}

# XGBoost
xgb_params = {
    # number of trees
    "model__n_estimators":      [100, 200, 500],
    # maximum tree depth, set low to prevent overfitting
    "model__max_depth":         [3, 5, 8],
    # learning rate: controls correction of previous errors, set low to prevent overfitting
    "model__learning_rate":     [0.05, 0.1],
    # fraction of training data sampled at each step, set <0.8 to prevent overfitting
    "model__subsample":         [0.6, 0.8]
}


# 1) Random Forest Classifier
rf_clf = def_clf(
    RandomForestClassifier(random_state=42),
    rf_params,
    k_values=[3,5,7]
)

# Call permutation importance function
rf_clf_pipe = Pipeline([("model", RandomForestClassifier(random_state=42, n_jobs=-1))])
rf_clf_pipe.set_params(**rf_clf.iloc[0]["params"]).fit(X_clf, y_clf)
df_rf_clf_imp = P_importance(
    rf_clf_pipe,
    X_clf,
    y_clf,
    feature,
    scoring="accuracy",
    title="Random Forest Classifier Feature Importances"
)

# 2) Gradient Boosting Classifier
gb_clf = def_clf(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    k_values=[3,5,7]
)

# 3) Random Forest Regressor
rf_reg = def_reg(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_params,
    k_values=[3,5,7]
)

# Call permutation importance function
rf_reg_pipe = Pipeline([("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
rf_reg_pipe.set_params(**rf_reg.iloc[0]["params"]).fit(X_reg, y_reg)
df_rf_reg_imp = P_importance(
    rf_reg_pipe,
    X_reg,
    y_reg,
    feature,
    scoring="neg_mean_squared_error",
    title="Random Forest Regressor Feature Importances"
)

# 4) Gradient Boosting Regressor
gb_reg = def_reg(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    k_values=[3,5,7]
)

# 5) XGBClassifier
from sklearn.preprocessing import LabelEncoder

# LabelEncoder to convert y_clf's string classes (A, B, C, D) to numbers
le = LabelEncoder()
y_xgb = le.fit_transform(y_clf)   # 'A'→0, 'B'→1, 'C'→2, 'D'→3
y_clf_orig = y_clf.copy()
y_clf = y_xgb

xgb_clf = def_clf(
    XGBClassifier(
        eval_metric="mlogloss",        # use multiclass logistic loss
        objective="multi:softprob",    # output probabilities
        random_state=42,
        n_jobs=-1
    ),
    xgb_params,
    k_values=[3,5,7]
)

y_clf = y_clf_orig  # restore original string classes


# 6) XGBRegressor
xgb_reg = def_reg(
    XGBRegressor(
        objective="reg:squarederror",  # output regression loss
        random_state=42,
        n_jobs=-1
    ),
    xgb_params,
    k_values=[3,5,7]
)

def run_pipeline():
    """
    This is the top-level function required by the Open SW assignment.
    It executes the main steps of the pipeline.
    """
    print("[START] Running full pipeline...")

    # Reload data (already preprocessed in the script above)
    df = pd.read_csv("preprocessed_cmudata.csv")

    # Define features and targets
    features = [
        'TotalSleepTime', 'bedtime_mssd', 'midpoint_sleep', 'daytime_sleep',
        'cum_gpa', 'Zterm_units_ZofZ', 'demo_gender', 'demo_race', 'demo_firstgen',
        'term_units', 'frac_nights_with_data',
        'sleep_consistency_index', 'sleep_efficiency_z', 'day_night_ratio', 'sleep_debt'
    ]
    df_clean = df[features + ['term_gpa']].dropna()
    X = df_clean[features]
    y = df_clean['term_gpa']

    # Run RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    # Save results
    result_df = df_clean.copy()
    result_df["prediction"] = predictions
    result_df.to_csv("./output/results.csv", index=False)

    print("[DONE] Results saved to ./output/results.csv")
