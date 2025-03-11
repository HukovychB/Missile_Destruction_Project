# Missile_Destruction_Project
Data Science and ML project in R

## Overview
This study explores the application of machine learning techniques to predict the destruction of missiles and drones based on historical attack data and meteorological conditions. By leveraging multiple predictive models, the research aims to identify key factors influencing interception success and improve forecasting accuracy.

## Motivation
The ability to predict missile and drone destruction is crucial for military strategy and defense planning. This study addresses the problem of integrating missile attack data with weather data, aiming to quantify the impact of different attack strategies and meteorological conditions on interception probabilities.

## Methodology
The study utilizes the publicly available *Massive Missile Attacks on Ukraine* dataset from Kaggle, which includes details on missile and drone launches (location, type, number, date), and interception success rates. This dataset is enriched with weather data from Meteostat to evaluate the effect of meteorological conditions such as temperature, wind speed, and precipitation on interception outcomes.

## Data Processing
- **Cleaning and preprocessing:** Handling missing values, grouping of values into meaningful categories for each variable based on warfare knowledge, one-hot encoding categorical variables.
- **Feature engineering:** Creating rolling averages and seasonal factors to improve model performance.

## Machine Learning Models
A range of models was employed, from simple linear regression to advanced ensemble techniques:
- **Ordinary Least Squares (OLS) Regression:** Baseline model for interpretability.
- **Elastic Net Regression:** Balances between Lasso and Ridge regression to improve feature selection.
- **Random Forest:** Ensemble-based decision trees to handle non-linearity.
- **Gradient Boosting (XGBoost):** Optimized boosting method for better predictive accuracy.
- **Multilayer Perceptron (MLP):** Neural network for capturing complex relationships.
- **k-Nearest Neighbors (kNN) and Support Vector Machine (SVM) Regression:** Additional models for comparison.
- **Ensemble Learning:** A weighted ensemble model combining predictions from multiple models to improve accuracy and robustness.

## Evaluation Metrics
Models were assessed using:
- **Root Mean Squared Error (RMSE):** Measures prediction error magnitude.
- **Mean Absolute Error (MAE):** Assesses overall deviation from true values.
- **R² (R-Squared):** Evaluates how well models explain variance in missile destruction.

## Results
- **Best Performing Models:** The ensemble models outperformed individual models, with the weighted ensemble achieving the lowest RMSE (1.85) and highest R² (0.991).
- **Key Predictors:** The number of missiles launched was the most significant factor affecting interception success. Missile type, launch method, and geographical origin also influenced destruction rates.
- **Minimal Impact of Weather:** Despite expectations, wind speed, precipitation, and other meteorological factors showed weak correlations with missile interception rates, suggesting that modern guidance systems are resilient to moderate weather conditions.
- **Permutation Tests:** Statistical tests confirmed that ensemble models significantly outperformed simpler models like OLS and Elastic Net, validating the robustness of the approach.
