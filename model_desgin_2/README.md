# Model Design 2 (model_desgin_2) 🧠

This folder contains the software architecture for the second model based on merged relational datasets.

## 📊 Dataset Information
- **Source**: [Kaggle - UK Road Safety Accidents and Vehicles](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles)
- **Accidents File**: 2,047,256 rows | 34 columns
- **Vehicles File**: 2,177,205 rows | 24 columns
- **Merged Capacity**: Supports over 4 million combined records.

## 🚀 Architectural Features
- **Dual-Model Inference**: Simultaneously predicts accident severity and vehicle type.
- **Libya Localization**: Full Arabic translation and mapping to 100+ Libyan cities.
- **Portability**: All components (Train, App, API) use dynamic path detection.

## 🛠️ Hyperparameters
- `N_ESTIMATORS = 200`
- `LEARNING_RATE = 0.1`
- `DATA_ROWS = None` (Optimized for large-scale training).
