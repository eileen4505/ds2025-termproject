
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def run_pipeline(data_path: str, model_name: str):
    """
    Executes a regression pipeline using the specified model.

    Parameters:
    - data_path (str): Path to the input CSV file
    - model_name (str): One of ['rf_reg', 'xgb_reg', 'lr', 'poly2']

    Saves:
    - CSV file with prediction results in the './output/' directory
    Prints:
    - R² and RMSE scores for evaluation
    """

    print(f"[START] Running pipeline for model: {model_name}")

    # Load data
    df = pd.read_csv(data_path)

    # Define input features and target variable
    features = [
        'TotalSleepTime', 'bedtime_mssd', 'midpoint_sleep', 'daytime_sleep',
        'cum_gpa', 'Zterm_units_ZofZ', 'demo_gender', 'demo_race', 'demo_firstgen',
        'term_units', 'frac_nights_with_data',
        'sleep_consistency_index', 'sleep_efficiency_z', 'day_night_ratio', 'sleep_debt'
    ]
    df_clean = df[features + ['term_gpa']].dropna()
    X = df_clean[features]
    y = df_clean['term_gpa']

    # Model selection based on user input
    if model_name == 'rf_reg':
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'xgb_reg':
        model = XGBRegressor(random_state=42, objective="reg:squarederror")
    elif model_name == 'lr':
        model = LinearRegression()
    elif model_name == 'poly2':
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression())
        ])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Train model and predict
    model.fit(X, y)
    preds = model.predict(X)

    # Evaluate
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save prediction results
    df_clean["prediction"] = preds
    os.makedirs("./output", exist_ok=True)
    output_path = f"./output/{model_name}_results.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")

    return {"model": model_name, "r2": r2, "rmse": rmse}

def run_all_models(data_path: str):
    """
    Runs multiple models and prints top 5 by R² score.

    Parameters:
    - data_path (str): Path to the input CSV file
    """
    model_list = ['rf_reg', 'xgb_reg', 'lr', 'poly2']
    results = []

    for model_name in model_list:
        try:
            result = run_pipeline(data_path, model_name)
            results.append(result)
        except Exception as e:
            print(f"Error running {model_name}: {e}")

    df_result = pd.DataFrame(results).sort_values("r2", ascending=False)

    print("\nTop 5 Models by R2 Score:")
    print(df_result.head(5)[["model", "r2", "rmse"]].to_string(index=False))

    best_model = df_result.iloc[0]
    print(f"\nBest Model: {best_model['model']} (R2: {best_model['r2']:.4f}, RMSE: {best_model['rmse']:.4f})")
