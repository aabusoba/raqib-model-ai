# 🚦 Raqib AI - Traffic Decision Support System (DSS)

Raqib AI is a dual-model Traffic Intelligence Hub designed for Libya. It utilizes advanced machine learning (XGBoost) and data analysis to predict road accident severity and provide actionable safety recommendations.

## 🚀 Quick Start - Docker (Recommended)

The easiest way to run the entire system (both models + API + Gateway) is using Docker Compose.

### Prerequisites
- Docker and Docker Compose installed.

### Installation
1. Clone the repository.
2. Build and start the containers:
   ```bash
   docker-compose up -d --build
   ```
3. Access the system at:
   - **Main System Status (JSON):** [http://localhost:8080](http://localhost:8080)
   - **Model Design 1 Dashboard:** [http://localhost:8080/mod1/](http://localhost:8080/mod1/)
   - **Model Design 2 Dashboard:** [http://localhost:8080/mod2/](http://localhost:8080/mod2/)

---

## 🛠 Manual Local Installation

If you prefer to run the models independently without Docker:

### Prerequisites
- Python 3.9+ installed.

### Steps for each model (Design 1 or Design 2)

1. **Navigate to the model folder:**
   ```bash
   cd model_desgin_1  # or model_desgin_2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the AI Models (Mandatory for the first time):**
   This will sync data from Kaggle and generate weights (`.pkl`) and stats summary (`.json`).
   ```bash
   python train.py
   ```

4. **Launch the Dashboard (Streamlit):**
   ```bash
   streamlit run app.py
   ```

5. **Launch the API (Flask):**
   ```bash
   python api.py
   ```

---

## 📊 Feature Highlights
- **Model Design 1:** Focuses on statistical classification and classic ML.
- **Model Design 2:** Advanced XGBoost model integrating vehicle and driver data.
- **Lightweight Mode:** Once trained, the system runs using model weights (`.pkl`) and a tiny JSON summary, requiring NO local CSV storage.
- **Auto-Sync:** Real-time data synchronization from Kaggle HUB.

## 🔗 Routing Architecture
The system uses an Nginx gateway to route traffic:
- `/mod1/` -> Dashboard 1
- `/mod1/api/` -> API 1 (Swagger docs at `/mod1/api/apidocs/`)
- `/mod2/` -> Dashboard 2
- `/mod2/api/` -> API 2 (Swagger docs at `/mod2/api/apidocs/`)

---
© 2026 Raqib AI Hub - Optimized for Traffic Safety Analytics.
