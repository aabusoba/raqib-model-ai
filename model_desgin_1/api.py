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
    'title': 'Raqib AI - Model Design 1',
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
        "title": "Raqib AI Model 1 API",
        "description": "API for classical traffic AI predictions",
        "version": "1.0.0"
    },
    "basePath": "/mod1/api", # This must match Nginx rewrite
    "schemes": ["http", "https"]
}

swagger = Swagger(app, template=template)

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
            stats = json.load(f)
        if 'map_data' not in stats:
            df = load_and_preprocess()
            stats['map_data'] = df[['Latitude', 'Longitude', 'Accident_Severity']].dropna().sample(min(1000, len(df))).to_dict('records')
            # Optionally save back, but since deployed, perhaps not
        return stats
    df = load_and_preprocess()
    # Sample map data for visualization (limit to 1000 points for performance)
    map_sample = df[['Latitude', 'Longitude', 'Accident_Severity']].dropna().sample(min(1000, len(df))).to_dict('records')
    return {
        "severity_distribution": get_severity_report(df).to_dict(),
        "peak_hours": [{"hour": int(h), "count": int(c)} for h, c in get_peak_times(df).items()],
        "top_cities": [{"city": city, "accidents": int(count)} for city, count in get_dangerous_locations(df).items()],
        "lighting_impact_pct": cross_analysis_lighting(df),
        "map_data": map_sample
    }

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get dashboard statistics data (Model 1)"""
    return jsonify(sanitize_json(get_dashboard_stats()))

@app.route('/hotspots', methods=['GET'])
def get_hotspots():
    """Get accident hotspots data (Model 1)"""
    df = load_and_preprocess()
    dangerous = get_dangerous_locations(df)
    # Assuming map data is available, but for now return dangerous locations
    return jsonify(sanitize_json({
        "dangerous_locations": {city: int(count) for city, count in dangerous.items()},
        "total_accidents": int(sum(dangerous.values()))
    }))

@app.route('/predictions', methods=['POST'])
def get_predictions():
    """Get AI predictions and recommendations (Model 1)"""
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
        'Urban_or_Rural_Area': req_data.get('area', 'حضري'),
        'Road_Type': 'طريق فردي',
        'Junction_Control': 'Give way or uncontrolled'
    }
    pred_sev = predict_risk(m_s, l_s, input_data, f_cols)
    pred_veh = predict_risk(m_v, l_v, input_data, f_cols)
    adj_sev, overwritten = get_safety_adjusted_risk(pred_sev, input_data)
    
    stats = get_dashboard_stats()
    analysis_res = {'increase_pct': stats.get('lighting_impact_pct', 0), 'top_dangerous': stats.get('top_cities', [{'city': 'N/A'}])[0]['city']}
    preds_res = {'risk': pred_sev, 'time': f"{input_data['Hour']}:00"}
    recs = get_recommendations(analysis_res, preds_res, input_data)
    
    return jsonify(sanitize_json({
        "ai_prediction": {"severity": pred_sev, "vehicle_type": pred_veh},
        "safety_adjusted": {"severity": adj_sev, "is_overridden": overwritten},
        "recommendations": [{"text": r[0], "level": r[1]} for r in recs]
    }))

@app.route('/performance', methods=['GET'])
def get_performance():
    """Get model performance metrics (Model 1)"""
    model_data = load_models()
    if not model_data: return jsonify({"error": "Models not found"}), 500
    m_s, m_v, l_s, l_v, f_s, met_s, met_v = model_data
    return jsonify(sanitize_json({
        "severity_model": {
            "test_accuracy": met_s.get('test_acc', 0),
            "report": met_s.get('report', {})
        },
        "vehicle_model": {
            "test_accuracy": met_v.get('test_acc', 0),
            "report": met_v.get('report', {})
        }
    }))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
