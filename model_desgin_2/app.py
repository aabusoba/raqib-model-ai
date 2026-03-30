import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
import json
from data_processor import load_and_preprocess
from analyzer import get_dangerous_locations, get_peak_times, get_severity_report, cross_analysis_lighting
from predictor import predict_risk
from recommender import get_recommendations, get_safety_adjusted_risk

st.set_page_config(layout="wide", page_title="نظام رقيب DSS - التصميم الثاني")

@st.cache_data
def get_stats_summary():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_dir, 'v2_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_sev_path = os.path.join(current_dir, 'model_severity_v2.pkl')
    m_veh_path = os.path.join(current_dir, 'model_vehicle_v2.pkl')
    if not os.path.exists(m_sev_path) or not os.path.exists(m_veh_path): return None
    with open(m_sev_path, 'rb') as f: m_s, l_s, f_s, met_s = pickle.load(f)
    with open(m_veh_path, 'rb') as f: m_v, l_v, f_v, met_v = pickle.load(f)
    return m_s, m_v, l_s, l_v, f_s, met_s, met_v

stats = get_stats_summary()
model_data = load_models()

st.title("🚦 نظام رقيب لدعم القرار المروري (DSS) - التصميم الثاني 🇱🇾")
if stats:
    st.sidebar.info(f"إجمالي البيانات (من الملخص الإحصائي): {stats['total_rows']:,} صف")
else:
    st.sidebar.warning("⚠️ لم يتم العثور على الملخص الإحصائي. الخرائط قد لا تظهر بدون تدريب مسبق.")

if not model_data:
    st.error("⚠️ النماذج غير موجودة. يرجى تشغيل التدريب أولاً.")
    st.stop()

m_sev, m_veh, l_sev, l_veh, f_cols, metrics_s, metrics_v = model_data
tab1, tab2, tab3, tab4 = st.tabs(["📊 الإحصائيات التحليلية", "🗺️ الخارطة التفاعلية", "🧠 التنبؤ الذكي", "📈 أداء النموذج الجديد"])

with tab1:
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("أكثر المدن الليبية تسجيلاً للحوادث")
            cities_df = pd.DataFrame(stats['top_cities']).set_index('city')
            st.plotly_chart(px.bar(cities_df, y='accidents', color='accidents'))
        with col2:
            st.subheader("تحليل ساعات الذروة")
            peak_df = pd.DataFrame(stats['peak_hours']).set_index('hour')
            st.plotly_chart(px.line(peak_df, y='count', markers=True))
        st.subheader("توزيع خطورة الحوادث")
        st.plotly_chart(px.pie(values=list(stats['severity_distribution'].values()), names=list(stats['severity_distribution'].keys()), hole=0.4))
    else:
        st.info("قم بتشغيل التدريب لتوليد الإحصائيات.")

with tab2:
    st.subheader("توزيع الحوادث جغرافياً")
    st.info("الجزء التفاعلي للخريطة يحتاج لقاعدة البيانات الكاملة. يمكنك الوصول إليه محلياً فقط.")

with tab3:
    st.subheader("محاكاة التنبؤ والتوصيات")
    c1, c2, c3 = st.columns(3)
    with c1:
        hour = st.slider("الساعة", 0, 23, 12)
        day = st.selectbox("اليوم", ["الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"])
    with c2:
        light = st.selectbox("الإضاءة", ["ضوء النهار", "ظلام - مصابيح مشتعلة", "ظلام - لا توجد إضاءة"])
        surface = st.selectbox("حالة الطريق", ["جاف", "رطب أو مبلل", "ثلج"])
    with c3:
        urban = st.selectbox("المنطقة", ["حضري", "ريفي"])
        
    input_data = {'Hour': hour, 'Day_of_Week': day, 'Light_Conditions': light, 'Road_Surface_Conditions': surface, 'Urban_or_Rural_Area': urban, 'Weather_Conditions': 'صافي - لا رياح', 'Road_Type': 'طريق فردي'}
    pred_sev_idx = predict_risk(m_sev, l_sev, input_data, f_cols)
    st.info(f"🔍 التنبؤ الأولي بالخطورة: {pred_sev_idx}")

with tab4:
    st.subheader("أداء النموذج")
    st.write(f"✅ دقة نموذج الخطورة: {metrics_s['test_acc']:.2%}")
    st.write(f"✅ دقة نموذج المركبات: {metrics_v['test_acc']:.2%}")
    st.json(metrics_s['report'])
