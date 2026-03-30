# Model Design 1 (model_desgin_1) 🚦

This folder contains the software architecture for the first model based on the "Road Accident Data" dataset.

## 📊 Dataset Information
- **Source**: [Kaggle - Road Accident Dataset](https://www.kaggle.com/datasets/xavierberge/road-accident-dataset/data)
- **Rows**: 307,973
- **Columns**: 23
- **Localization**: Localized for Libyan cities (Arabic) in the dashboard.

## 🚀 Components
- `train.py`: Model training using XGBoost and SMOTE.
- `app.py`: Arabic Streamlit DSS dashboard.
- `api.py`: Flask REST API for system integration.
- `libyan_cities.json`: Geographic data for localization.

## 🛠️ Hyperparameters
- `N_ESTIMATORS = 200`
- `LEARNING_RATE = 0.1`
- `DATA_ROWS = None` (Set to None for full dataset training).
