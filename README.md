# Term Project: Sleep Habits and GPA Prediction
This project aims to analyze how sleep-related variables impact academic performance (GPA), using machine learning models on the CMU Sleep Dataset. We built an end-to-end pipeline that includes data preprocessing, exploratory data analysis, model training, evaluation, and interpretation.
## Project Objectives

- Investigate the relationship between sleep habits and GPA
- Compare predictive power of sleep-related vs. demographic/genetic variables
- Apply both classification and regression models to understand performance patterns
- Share the entire pipeline for reproducibility and further research

## Repository Structure

| File / Folder              | Description |
|---------------------------|-------------|
| `final_pipeline_code.py`  | Full code pipeline from preprocessing to evaluation |
| `preprocessed_cmudata.csv`| Cleaned and transformed dataset used for modeling |
| `Term Project_Final Report.pdf` | Final report describing methodology and results |
| `requirements.txt`        | List of required packages |
| `README.md`               | Project overview and usage guide |

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

## Example Results

- Sleep-only model (Classification): Accuracy ~0.4507  
- Demographic-only model: Accuracy ~0.4179  
- Full-feature ensemble model: Accuracy ~0.6168  

Sleep-related features showed relatively higher predictive power than innate features.

## Contributors

- 박세렴 — final report writing ,open source SW
- 박형빈 — proposal,visualization, exploratory data analysis   
- 송민의 - preprocessing, evaluation
- 유호찬 - modeling, evaluation
- 이채민 - final report editing, presentation
