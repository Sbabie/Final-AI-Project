 🌍 CO₂ Emissions Prediction System – SDG 13

This project presents a machine learning-based regression system that predicts CO₂ emissions from environmental and economic indicators. It supports **UN Sustainable Development Goal 13: Climate Action** by enabling early analysis and policy intervention based on emissions forecasts.

## 📌 Features
- Predicts CO₂ emissions using energy, population, urbanization, GDP, and renewable energy indicators.
- Trains and compares 3 regression models:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Automatically selects the best model based on R² score.
- Includes example visualization and live prediction.

## 🧠 Model Inputs
- `energy_consumption` (kWh)
- `population`
- `gdp_per_capita` (USD)
- `urbanization_rate` (%)
- `renewable_energy_pct` (%)
- `forest_area_pct` (%)

## 📊 Outputs
- Predicted `CO₂ emissions` (metric tons per capita)
- Model evaluation metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

## 📈 Sample Prediction
