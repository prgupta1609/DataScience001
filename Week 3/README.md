# Week 3: Customer Intelligence System

This project develops an end-to-end Customer Intelligence System using classification, ensemble learning, and clustering on the Country Data dataset.

## Objective

The goal is to segment countries into meaningful customer-style priority groups and then build classification models that can predict these segments for new records.

## Techniques Used

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Standard scaling
- K-Means clustering
- DBSCAN clustering
- PCA visualization
- Segment profiling and business interpretation
- Classification modeling
- Random Forest ensemble model
- XGBoost ensemble model
- Hyperparameter tuning using RandomizedSearchCV
- Confusion matrix and classification report
- Feature importance analysis

## Dataset

Files used:

- `Country-data.csv`
- `data-dictionary.csv`

The dataset contains country-level socio-economic indicators such as child mortality, income, GDP per capita, inflation, health spending, exports, imports, and life expectancy.

## Project Flow

1. Load and understand the data
2. Check missing values and duplicates
3. Perform EDA
4. Create additional meaningful features
5. Scale numerical features
6. Apply K-Means clustering
7. Create customer priority segment labels
8. Visualize clusters using PCA
9. Apply DBSCAN for density-based clustering and outlier detection
10. Train classification models using the created segment labels
11. Compare Random Forest and XGBoost
12. Tune ensemble models
13. Interpret feature importance
14. Derive actionable segmentation insights

## How to Run

Install required libraries:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```text
week3_PriyaGupta.ipynb
```

Run all cells from top to bottom.

## Author

Priya Gupta
