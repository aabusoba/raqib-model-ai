from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import pandas as pd
import pickle
import os
import json
from data_processor import load_and_preprocess
from analyzer import get_dangerous_locations, get_peak_times, get_severity_report, cross_analysis_lighting
from predictor import predict_risk
from recommender import get_recommendations, get_safety_adjusted_risk

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_sev_path = os.path.join(current_dir, 'model_severity.pkl')
    m_veh_path = os.path.join(current_dir, 'model_vehicle.pkl')
    if not os.path.exists(m_sev_path) or not os.path.exists(m_veh_path): return None
    with open(m_sev_path, 'rb') as f: m_s, l_s, f_s, met_s = pickle.load(f)
    with open(m_veh_path, 'rb') as f: m_v, l_v, f_v, met_v = pickle.load(f)
    return m_s, m_v, l_s, l_v, f_s, met_s, met_v

def sanitize_json(data):
    return json.loads(json.dumps(data).replace('NaN', '0').replace('Infinity', '0'))

def get_dashboard_stats():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_dir, 'v1_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Fallback only if absolutely necessary
    df = load_and_preprocess()
    return {
        "severity_distribution": get_severity_report(df).to_dict(),
        "peak_hours": [{"hour": int(h), "count": int(c)} for h, c in get_peak_times(df).items()],
        "top_cities": [{"city": city, "accidents": int(count)} for city, count in get_dangerous_locations(df).items()],
        "lighting_impact_pct": cross_analysis_lighting(df)
    }

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(sanitize_json(get_dashboard_stats()))

@app.route('/api/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    model_data = load_models()
    if not model_data: return jsonify({"error": "Models not found"}), 500
    m_s, m_v, l_s, l_v, f_cols, met_s, met_v = model_data
    input_data = {
        'Hour': int(req_data.get('hour', 12)),
        'Day_of_Week': req_data.get('day', 'الأحد'),
        'Light_Conditions': req_data.get('light', 'ضوء النهار'),
        'Weather_Conditions': req_data.get('weather', 'صافي - لا رياح'),
        'Road_Surface_Conditions': req_data.get('surface', 'جاف'),
        'Junction_Control': req_data.get('junction', 'Give way or uncontrolled'),
        'Road_Type': req_data.get('road_type', 'طريق فردي'),
        'Urban_or_Rural_Area': req_data.get('area', 'حضري')
    }
    pred_sev = predict_risk(m_s, l_s, input_data, f_cols)
    pred_veh = predict_risk(m_v, l_v, input_data, f_cols)
    adj_sev, overwritten = get_safety_adjusted_risk(pred_sev, input_data)
    
    # Recommendations using cached impact if possible
    stats = get_dashboard_stats()
    analysis_res = {'increase_pct': stats.get('lighting_impact', 0), 'top_dangerous': stats.get('top_cities', [{'city': 'N/A'}])[0]['city']}
    preds_res = {'risk': pred_sev, 'time': f"{input_data['Hour']}:00"}
    recs = get_recommendations(analysis_res, preds_res, input_data)
    return jsonify(sanitize_json({
        "ai_prediction": {"severity": pred_sev, "vehicle_type": pred_veh},
        "safety_adjusted": {"severity": adj_sev, "is_overridden": overwritten},
        "recommendations": [{"text": r[0], "level": r[1]} for r in recs]
    }))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
