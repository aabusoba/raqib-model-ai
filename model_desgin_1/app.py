import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
import json
from data_processor import load_and_preprocess
from predictor import predict_risk

st.set_page_config(layout="wide", page_title="Traffic Decision Support System")

@st.cache_data
def get_stats_summary():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_dir, 'v1_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_sev_path = os.path.join(current_dir, 'model_severity.pkl')
    m_veh_path = os.path.join(current_dir, 'model_vehicle.pkl')
    if not os.path.exists(m_sev_path) or not os.path.exists(m_veh_path): return None
    with open(m_sev_path, 'rb') as f: m_s, l_s, f_s, met_s = pickle.load(f)
    with open(m_veh_path, 'rb') as f: m_v, l_v, f_v, met_v = pickle.load(f)
    return m_s, m_v, l_s, l_v, f_s, met_s, met_v

stats = get_stats_summary()
model_data = load_models()

st.title("🚦 نظام رقيب لدعم القرار المروري (DSS) - ليبيا 🇱🇾")
if stats:
    st.sidebar.info(f"إجمالي البيانات (من الملخص): {stats['total_rows']:,} صف")
else:
    st.sidebar.warning("⚠️ يرجى تشغيل التدريب أولاً لتوليد إحصائيات Dashboard.")

if not model_data:
    st.error("Models not found. Please run 'python train.py' first.")
    st.stop()
m_sev, m_veh, l_sev, l_veh, f_cols, metrics_s, metrics_v = model_data

tab1, tab2, tab3 = st.tabs(["📊 الإحصائيات", "🧠 تنبؤات النظام", "📈 أداء النموذج"])

with tab1:
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("أكثر المناطق خطورة")
            cities_df = pd.DataFrame(stats['top_cities']).set_index('city')
            st.plotly_chart(px.bar(cities_df, y='accidents', color='accidents'))
        with col2:
            st.subheader("أوقات ذروة الحوادث")
            peak_df = pd.DataFrame(stats['peak_hours']).set_index('hour')
            st.plotly_chart(px.line(peak_df, y='count', markers=True))
        st.subheader("توزيع خطورة الحوادث")
        st.plotly_chart(px.pie(values=list(stats['severity_distribution'].values()), names=list(stats['severity_distribution'].keys()), hole=0.4))
    else:
        st.info("قم بتشغيل التدريب لتوليد الإحصائيات.")

with tab2:
    st.subheader("التنبيهات التنبؤية والتوصيات")
    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.slider("اختر الساعة", 0, 23, 12)
        day = st.selectbox("اختر اليوم", ["الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"])
    with col2:
        light = st.selectbox("ظروف الإضاءة", ["ضوء النهار", "ظلام - مصابيح مشتعلة", "ظلام - لا توجد إضاءة"])
        weather = st.selectbox("حالة الطقس", ["صافي - لا رياح", "ممطر - لا رياح", "ضباب"])
    with col3:
        surface = st.selectbox("حالة سطح الطريق", ["جاف", "رطب أو مبلل"])
    
    input_data = {'Hour': hour, 'Day_of_Week': day, 'Light_Conditions': light, 'Weather_Conditions': weather, 'Road_Surface_Conditions': surface, 'Junction_Control': 'Give way or uncontrolled', 'Road_Type': 'طريق فردي', 'Urban_or_Rural_Area': 'حضري'}
    pred_sev = predict_risk(m_sev, l_sev, input_data, f_cols)
    from recommender import get_safety_adjusted_risk
    adj_sev, overwritten = get_safety_adjusted_risk(pred_sev, input_data)
    
    st.markdown(f"### الخطورة المتوقعة: **{pred_sev}**")
    if overwritten: st.markdown(f"### الخطورة المعدلة (نظام الأمان): **{adj_sev}** 🛡️")

with tab3:
    st.subheader("أداء النموذج")
    st.metric("دقة اختبار المصنف (الخطورة)", f"{metrics_s['test_acc']*100:.2f}%")
    st.metric("دقة اختبار المصنف (المركبة)", f"{metrics_v['test_acc']*100:.2f}%")
