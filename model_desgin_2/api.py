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

# Explicit Swagger config for Proxy environments
app.config['SWAGGER'] = {
    'title': 'Raqib AI - Model Design 2',
    'uiversion': 3,
    'specs_route': '/apidocs/',
    'static_url_path': '/flasgger_static',
    'specs': [
        {
            'endpoint': 'apispec_1',
            'route': '/apispec_1.json',
            'rule_filter': lambda rule: True,
            'model_filter': lambda tag: True,
        }
    ],
}

# Template to force relative paths in Swagger UI
template = {
    "swagger": "2.0",
    "info": {
        "title": "Raqib AI Model 2 API",
        "description": "API for advanced XGBoost traffic prediction",
        "version": "2.0.0"
    },
    "basePath": "/mod2/api", # This must match Nginx rewrite
    "schemes": ["http", "https"]
}

swagger = Swagger(app, template=template)

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_sev_path = os.path.join(current_dir, 'model_severity_v2.pkl')
    m_veh_path = os.path.join(current_dir, 'model_vehicle_v2.pkl')
    if not os.path.exists(m_sev_path) or not os.path.exists(m_veh_path): return None
    with open(m_sev_path, 'rb') as f: m_s, l_s, f_s, met_s = pickle.load(f)
    with open(m_veh_path, 'rb') as f: m_v, l_v, f_v, met_v = pickle.load(f)
    return m_s, m_v, l_s, l_v, f_s, met_s, met_v

def sanitize_json(data):
    return json.loads(json.dumps(data).replace('NaN', '0').replace('Infinity', '0'))

def get_dashboard_stats():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_dir, 'v2_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f: return json.load(f)
    df = load_and_preprocess()
    top_cities = get_dangerous_locations(df).to_dict()
    return {
        "severity_distribution": get_severity_report(df).to_dict(),
        "peak_hours": [{"hour": int(h), "count": int(c)} for h, c in get_peak_times(df).items()],
        "top_cities": [{"city": city, "accidents": int(count)} for city, count in top_cities.items()]
    }

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dashboard stats (Model 2)"""
    return jsonify(sanitize_json(get_dashboard_stats()))

@app.route('/predict', methods=['POST'])
def predict():
    """Predict risk using XGBoost (Model 2)"""
    req_data = request.get_json()
    model_data = load_models()
    if not model_data: return jsonify({"error": "Models not found"}), 500
    m_s, m_v, l_s, l_v, f_cols, met_s, met_v = model_data
    input_data = {'Hour': int(req_data.get('hour', 12)), 'Day_of_Week': req_data.get('day', 'الأحد')}
    return jsonify({"prediction_index": int(predict_risk(m_s, l_s, input_data, f_cols))})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
