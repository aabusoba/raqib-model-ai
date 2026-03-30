# Import the operating system module
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_processor import load_and_preprocess
from predictor import train_models
import pickle

# Define constants
DATA_ROWS = None
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
LEARNING_RATE = 0.1

def prepare_dataset_folder():
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
    os.makedirs('dataset')

def run_training():
    # Cleanup old root-level CSV if still there to save space
    old_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Road Accident Data.csv')
    if os.path.exists(old_csv):
        print(f"🧹 Clearing old project data file to save space: {os.path.basename(old_csv)}")
        os.remove(old_csv)

    print(f"--- High Accuracy Training (Design 1): Rows={DATA_ROWS}, Estimators={N_ESTIMATORS} ---")
    print("Loading data via Kaggle sync...")
    
    # Load and preprocess the raw data
    df = load_and_preprocess()
    df_subset = df if DATA_ROWS is None else df.head(DATA_ROWS)
    
    print("Splitting & Balancing data (SMOTE)...")
    train_df, test_df = train_test_split(df_subset, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Prepare local folders for CSV export (this is for split results)
    prepare_dataset_folder()
    train_df.to_csv('dataset/train.csv', index=False)
    test_df.to_csv('dataset/test.csv', index=False)
    
    print("Starting XGBoost training...")
    results = train_models(train_df, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
    m_s, m_v, l_s, l_v, cols, met_s, met_v = results
    
    print("\n" + "█"*40)
    print("  أداء نموذج خطورة الحوادث")
    print("█"*40)
    print(f"دقة التدريب: {met_s['train_acc']*100:.2f}%")
    print(f"دقة الاختبار:  {met_s['test_acc']*100:.2f}%")
    
    for cls in ['قاتل', 'خطير', 'بسيط']:
        if cls in met_s['report']:
            m = met_s['report'][cls]
            print(f"  [{cls:7}] دقة: {m['precision']:.2f} | استدعاء: {m['recall']:.2f} | F1: {m['f1-score']:.2f}")

    print("\n" + "█"*40)
    print("  أداء نموذج نوع المركبة")
    print("█"*40)
    print(f"دقة التدريب: {met_v['train_acc']*100:.2f}%")
    print(f"دقة الاختبار:  {met_v['test_acc']*100:.2f}%")
    
    top_v = sorted(met_v['report'].items(), key=lambda x: x[1]['support'] if isinstance(x[1], dict) else 0, reverse=True)[:5]
    for cls, m in top_v:
        if isinstance(m, dict):
            print(f"  [{cls[:15]:15}] Prec: {m['precision']:.2f} | Rec: {m['recall']:.2f}")

    # --- New: Save Dashboard Stats to JSON for lightweight deployment ---
    print("📊 Generating dashboard statistics summary...")
    from analyzer import get_dangerous_locations, get_peak_times, get_severity_report, cross_analysis_lighting
    
    stats = {
        "severity_distribution": get_severity_report(df_subset).to_dict(),
        "peak_hours": [{"hour": int(h), "count": int(c)} for h, c in get_peak_times(df_subset).items()],
        "top_cities": [{"city": city, "accidents": int(count)} for city, count in get_dangerous_locations(df_subset).items()],
        "lighting_impact": float(cross_analysis_lighting(df_subset)),
        "total_rows": len(df_subset)
    }
    
    import json
    with open('model_desgin_1/v1_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    # --- End Stats ---

    with open('model_severity.pkl', 'wb') as f:
        pickle.dump((m_s, l_s, cols, met_s), f)
    with open('model_vehicle.pkl', 'wb') as f:
        pickle.dump((m_v, l_v, cols, met_v), f)
    print("\nTraining Complete. Models and Stats saved.")


if __name__ == "__main__":
    run_training()
