# Tesla Estimated Deliveries Prediction and Forecasting

This repository contains my Week 2 Data Science assignment. The project builds an end-to-end machine learning pipeline on Tesla deliveries and production data from 2015 to 2025.

## Problem Statement

Build an end-to-end ML pipeline to predict and forecast Tesla estimated deliveries using production, pricing, model, region, charging infrastructure, and time-based features.

## Target Variable

`Estimated_Deliveries`

This target is used because deliveries are the closest sales-related metric available in the dataset.

## Project Workflow

- Data loading and understanding
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Regression modeling
- Hyperparameter tuning
- Feature importance analysis
- Time series forecasting

## Models Used

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Tuned Random Forest Regressor

## Evaluation Metrics

- MAE
- MSE
- RMSE
- R² Score
- Adjusted R²

## Files

- `week2_PriyaGupta.ipynb` - Completed Jupyter Notebook
- `tesla_deliveries_dataset_2015_2025.csv` - Dataset
- `requirements.txt` - Required Python libraries
- `.gitignore` - Files/folders ignored by Git

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Open the notebook in VS Code or Jupyter Notebook and run all cells from top to bottom.
