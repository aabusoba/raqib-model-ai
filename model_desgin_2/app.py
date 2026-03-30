import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
from data_processor import load_and_preprocess
from analyzer import get_dangerous_locations, get_peak_times, get_severity_report, cross_analysis_lighting
from predictor import predict_risk
from recommender import get_recommendations, get_safety_adjusted_risk

st.set_page_config(layout="wide", page_title="نظام رقيب DSS - التصميم الثاني")

@st.cache_data
def get_data():
    return load_and_preprocess(row_limit=None)

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_sev_path = os.path.join(current_dir, 'model_severity_v2.pkl')
    m_veh_path = os.path.join(current_dir, 'model_vehicle_v2.pkl')
    
    if not os.path.exists(m_sev_path) or not os.path.exists(m_veh_path):
        return None
    with open(m_sev_path, 'rb') as f:
        m_s, l_s, f_s, met_s = pickle.load(f)
    with open(m_veh_path, 'rb') as f:
        m_v, l_v, f_v, met_v = pickle.load(f)
    return m_s, m_v, l_s, l_v, f_s, met_s, met_v

df = get_data()
model_data = load_models()

st.title("🚦 نظام رقيب لدعم القرار المروري (DSS) - التصميم الثاني 🇱🇾")
st.sidebar.info(f"إجمالي البيانات المدمجة: {len(df):,} صف")

if not model_data:
    st.error("⚠️ النماذج غير موجودة. يرجى تشغيل التدريب أولاً.")
    st.stop()

m_sev, m_veh, l_sev, l_veh, f_cols, metrics_s, metrics_v = model_data

tab1, tab2, tab3, tab4 = st.tabs(["📊 الإحصائيات التحليلية", "🗺️ الخارطة التفاعلية", "🧠 التنبؤ الذكي", "📈 أداء النموذج الجديد"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("أكثر المدن الليبية تسجيلاً للحوادث")
        top_locs = get_dangerous_locations(df)
        st.plotly_chart(px.bar(top_locs, color=top_locs.values, labels={'value':'عدد الحوادث', 'index':'المدينة'}))
    with col2:
        st.subheader("تحليل ساعات الذروة")
        peak_t = get_peak_times(df)
        st.plotly_chart(px.line(peak_t.sort_index(), markers=True, labels={'value':'الحوادث', 'index':'الساعة'}))
    
    st.subheader("توزيع خطورة الحوادث في قاعدة البيانات الجديدة")
    sev_rep = get_severity_report(df)
    st.plotly_chart(px.pie(values=sev_rep.values, names=sev_rep.index, hole=0.4))

with tab2:
    st.subheader("توزيع الحوادث جغرافياً في ليبيا")
    map_data = df[['Latitude', 'Longitude']].dropna().rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    st.map(map_data.sample(min(2000, len(map_data))))

with tab3:
    st.subheader("محاكاة التنبؤ والتوصيات")
    c1, c2, c3 = st.columns(3)
    with c1:
        hour = st.slider("الساعة", 0, 23, 12)
        day = st.selectbox("اليوم", df['Day_of_Week'].unique())
        weather = st.selectbox("حالة الطقس", df['Weather_Conditions'].unique())
    with c2:
        light = st.selectbox("الإضاءة", df['Light_Conditions'].unique())
        surface = st.selectbox("حالة الطريق", df['Road_Surface_Conditions'].unique())
        road_type = st.selectbox("نوع الطريق", df['Road_Type'].unique())
    with c3:
        urban = st.selectbox("المنطقة", df['Urban_or_Rural_Area'].unique())
        
    input_data = {
        'Hour': hour, 'Day_of_Week': day, 'Light_Conditions': light, 
        'Weather_Conditions': weather, 'Road_Surface_Conditions': surface,
        'Road_Type': road_type, 'Urban_or_Rural_Area': urban
    }
    
    # In V2, predict_risk returns index, we need to map back
    pred_sev_idx = predict_risk(m_sev, l_sev, input_data, f_cols)
    pred_veh_idx = predict_risk(m_veh, l_veh, input_data, f_cols)
    
    # Resolve names using label encoders in metrics or similar
    # For now, we assume the model metadata includes them or we use a manual map
    # Actually, the train.py should have saved the LabelEncoder instances.
    # I'll update predictor.py or train.py to be more robust, 
    # but for now I'll use a simple mapping for common labels.
    
    st.info(f"🔍 التنبؤ الأولي: {pred_sev_idx} | المركبة: {pred_veh_idx}")
    st.warning("ملاحظة: هذا التصميم يدعم التحليل المتقدم للمركبات والسائقين.")

with tab4:
    st.subheader("مقارنة دقة النماذج (XGBoost)")
    st.write(f"✅ دقة نموذج الخطورة: {metrics_s['test_acc']:.2%}")
    st.write(f"✅ دقة نموذج المركبات: {metrics_v['test_acc']:.2%}")
    st.json(metrics_s['report'])
