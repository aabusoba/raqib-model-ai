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

# Global Data & Model Cache
data_cache = None

def get_cached_data():
    global data_cache
    if data_cache is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'Road Accident Data.csv')
        data_cache = load_and_preprocess(data_path)
    return data_cache

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_sev_path = os.path.join(current_dir, 'model_severity.pkl')
    m_veh_path = os.path.join(current_dir, 'model_vehicle.pkl')
    
    if not os.path.exists(m_sev_path) or not os.path.exists(m_veh_path):
        return None
    with open(m_sev_path, 'rb') as f:
        m_s, l_s, f_s, met_s = pickle.load(f)
    with open(m_veh_path, 'rb') as f:
        m_v, l_v, f_v, met_v = pickle.load(f)
    return m_s, m_v, l_s, l_v, f_s, met_s, met_v

# sanitize json for NaN values
def sanitize_json(data):
    return json.loads(json.dumps(data).replace('NaN', '0').replace('Infinity', '0'))

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get accident statistics for dashboard
    ---
    responses:
      200:
        description: Accident statistics including severity, peak hours and top cities
    """
    df = get_cached_data()
    
    # 1. Severity Distribution
    sev_rep = get_severity_report(df).to_dict()
    
    # 2. Peak Hours (formatted for charts)
    peak_t = get_peak_times(df).sort_index()
    hours_list = [{"hour": int(h), "count": int(c)} for h, c in peak_t.items()]
    
    # 3. Top Cities
    top_cities = get_dangerous_locations(df).to_dict()
    cities_list = [{"city": city, "accidents": int(count)} for city, count in top_cities.items()]
    
    # 4. Environmental Analysis
    lighting_impact = cross_analysis_lighting(df)
    
    return jsonify(sanitize_json({
        "severity_distribution": sev_rep,
        "peak_hours": hours_list,
        "top_cities": cities_list,
        "lighting_impact_pct": lighting_impact
    }))

@app.route('/api/hotspots', methods=['GET'])
def get_hotspots():
    """
    Get accident hotspots coordinates
    ---
    responses:
      200:
        description: A list of accident hotspots with lat, lng and severity
    """
    df = get_cached_data()
    # Sample data for map performance in frontend
    sample_size = min(2000, len(df))
    map_data = df.sample(sample_size)[['Latitude', 'Longitude', 'Accident_Severity']]
    points = [
        {"lat": float(row['Latitude']), "lng": float(row['Longitude']), "severity": row['Accident_Severity']}
        for _, row in map_data.iterrows()
    ]
    return jsonify(points)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict accident risk and vehicle type
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            hour: {type: integer, example: 12}
            day: {type: string, example: "الأحد"}
            light: {type: string}
            weather: {type: string}
            surface: {type: string}
            junction: {type: string}
            road_type: {type: string}
            area: {type: string}
    responses:
      200:
        description: AI predictions and safety recommendations
    """
    req_data = request.get_json()
    model_data = load_models()
    if not model_data:
        return jsonify({"error": "Models not found. Run training first."}), 500
        
    m_s, m_v, l_s, l_v, f_cols, met_s, met_v = model_data
    
    # Required keys mapping from localized Streamlit inputs
    # Frontend should send these exactly
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
    
    # Safety Overrides
    adj_sev, overwritten = get_safety_adjusted_risk(pred_sev, input_data)
    
    # Recommendations
    df = get_cached_data()
    analysis_res = {'increase_pct': cross_analysis_lighting(df), 'top_dangerous': get_dangerous_locations(df).head(1)}
    preds_res = {'risk': pred_sev, 'time': f"{input_data['Hour']}:00"}
    recs = get_recommendations(analysis_res, preds_res, input_data)
    
    return jsonify(sanitize_json({
        "ai_prediction": {
            "severity": pred_sev,
            "vehicle_type": pred_veh
        },
        "safety_adjusted": {
            "severity": adj_sev,
            "is_overridden": overwritten
        },
        "recommendations": [{"text": r[0], "level": r[1]} for r in recs]
    }))

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """
    Get ML models performance reports
    ---
    responses:
      200:
        description: Classification reports and accuracy for severity and vehicle models
    """
    model_data = load_models()
    if not model_data:
        return jsonify({"error": "Models not found"}), 404
        
    _, _, _, _, _, met_s, met_v = model_data
    
    return jsonify(sanitize_json({
        "severity_model": {
            "accuracy": met_s['test_acc'],
            "report": met_s['report']
        },
        "vehicle_model": {
            "accuracy": met_v['test_acc'],
            "report": met_v['report']
        }
    }))

if __name__ == '__main__':
    # Initialize data on startup
    get_cached_data()
    app.run(host='0.0.0.0', port=5000, debug=True)
