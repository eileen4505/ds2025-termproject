# Term Project: Sleep Habits and GPA Prediction
This project aims to analyze how sleep-related variables impact academic performance (GPA), using machine learning models on the CMU Sleep Dataset. We built an end-to-end pipeline that includes data preprocessing, exploratory data analysis, model training, evaluation, and interpretation.
## Project Objectives

- Investigate the relationship between sleep habits and GPA
- Compare predictive power of sleep-related vs. demographic/genetic variables
- Apply both classification and regression models to understand performance patterns
- Share the entire pipeline for reproducibility and further research


## Repository Structure

| File / Folder               | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `data/CMU_Sleep.csv`        |  Raw dataset before preprocessing             |
| `data/preprocessed_cmudata.csv`  | Final cleaned dataset used for modeling                      |
| `notebooks/`                | Step-by-step preprocessing and modeling Jupyter notebooks    |
| `src/pipeline.py`           | Top-level function: Full pipeline from preprocessing to evaluation |
| `output/`                   | Model results, evaluation images (e.g., boxplots, confusion matrices) |
| `README.md`                 | Project overview and usage guide                            |
| `LICENSE.txt`               | License (MIT)                                               |
| `requirements.txt`          | List of required packages                                   |
| `Term_Project_Final_Report.pdf` | Final report describing the methodology and findings        |

## Features

- **Preprocessing**:
  - Missing value imputation using KNN and mode substitution
  - Outlier detection and conditional removal (IQR method)
  - Feature scaling using `RobustScaler`
  - Domain-based binning of GPA into four performance levels (A–D)

- **Exploratory Analysis**:
  - Visualizations (histograms, boxplots, heatmaps)
  - Feature correlation matrix
  - Distribution analysis by GPA class

- **Modeling & Evaluation**:
  - Classification: Decision Tree, KNN, XGBoost (for GPA class)
  - Regression: Linear, Polynomial, Random Forest, Gradient Boosting (for term GPA)
  - Permutation importance & learning curves
  - Hyperparameter tuning using GridSearchCV
    
## How to Use

1. Clone this repository
2. Prepare your data file at `data/preprocessed_cmudata.csv`
3. Run a model using Python:

```python
from final_pipeline)code import run_pipeline, run_all_models

# Run a single model
run_pipeline(data_path="data/preprocessed_cmudata.csv", model_name="rf_reg")

# Or run all models and compare performance
run_all_models(data_path="data/preprocessed_cmudata.csv")
   
## Example Results

- Best Model: XGBoost Regressor  
  - R²: 0.9999  
  - RMSE: 0.0049

- Top 5 Models by R² Score:
  1. xgb_reg — R²: 0.9999, RMSE: 0.0049  
  2. rf_reg  — R²: 0.9122, RMSE: 0.1482  
  3. gbr_reg — R²: 0.7958, RMSE: 0.2260  
  4. poly2   — R²: 0.6689, RMSE: 0.2877  
  5. lr      — R²: 0.4426, RMSE: 0.3734  

Sleep-related features, when used in combination with cumulative GPA and term unit information, showed strong predictive power for academic performance.

## Contributors

- 박세렴 — final report writing ,open source SW
- 박형빈 — proposal,visualization, exploratory data analysis   
- 송민의 - preprocessing, evaluation
- 유호찬 - modeling, evaluation
- 이채민 - final report editing, presentation
