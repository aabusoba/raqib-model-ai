import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
from data_processor import load_and_preprocess
from analyzer import get_dangerous_locations, get_peak_times, cross_analysis_lighting, get_severity_report
from predictor import predict_risk
from recommender import get_recommendations

st.set_page_config(layout="wide", page_title="Traffic Decision Support System")

@st.cache_data
def get_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Road Accident Data.csv')
    return load_and_preprocess(data_path)

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

df = get_data()
# DIAGNOSTIC: Print shape to confirm it's not a small balanced set
st.sidebar.write(f"إجمالي الصفوف المحملة: {len(df)}")
if len(df) < 5:
    st.sidebar.error("⚠️ خطأ في تحميل البيانات: البيانات فارغة أو ناقصة!")

model_data = load_models()

if model_data:
    m_sev, m_veh, l_sev, l_veh, f_cols, metrics_s, metrics_v = model_data
else:
    st.error("Models not found. Please run 'python train.py' first.")
    st.stop()

st.title("🚦 نظام رقيب لدعم القرار المروري (DSS) - ليبيا")

tab1, tab2, tab3, tab4 = st.tabs(["📊 الإحصائيات", "🗺️ بؤر الحوادث", "🧠 تنبؤات النظام", "📈 أداء النموذج"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("أكثر المناطق خطورة")
        top_locs = get_dangerous_locations(df)
        fig_locs = px.bar(top_locs, title="أعلى 10 مدن ليبية تأثراً")
        st.plotly_chart(fig_locs)
    with col2:
        st.subheader("أوقات ذروة الحوادث")
        peak_t = get_peak_times(df)
        fig_time = px.line(peak_t.sort_index(), title="الحوادث حسب ساعات اليوم")
        st.plotly_chart(fig_time)
    st.subheader("توزيع خطورة الحوادث")
    sev_rep = get_severity_report(df)
    fig_sev = px.pie(values=sev_rep.values, names=sev_rep.index, title="نسبة الخطورة")
    st.plotly_chart(fig_sev)

with tab2:
    st.subheader("الخارطة التفاعلية لبؤر الحوادث في ليبيا")
    map_data = df[['Latitude', 'Longitude']].dropna().rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    st.map(map_data.sample(min(1000, len(map_data))))

with tab3:
    st.subheader("التنبيهات التنبؤية والتوصيات")
    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.slider("اختر الساعة", 0, 23, 12)
        day = st.selectbox("اختر اليوم", df['Day_of_Week'].unique())
        urban = st.selectbox("منطقة حضرية أم ريفية", df['Urban_or_Rural_Area'].unique())
    with col2:
        light = st.selectbox("ظروف الإضاءة", df['Light_Conditions'].unique())
        weather = st.selectbox("حالة الطقس", df['Weather_Conditions'].unique())
        junction = st.selectbox("التحكم في التقاطع", df['Junction_Control'].unique())
    with col3:
        surface = st.selectbox("حالة سطح الطريق", df['Road_Surface_Conditions'].unique())
        road_type = st.selectbox("نوع الطريق", df['Road_Type'].unique())
    
    input_data = {
        'Hour': hour, 'Day_of_Week': day, 'Light_Conditions': light, 
        'Weather_Conditions': weather, 'Road_Surface_Conditions': surface,
        'Junction_Control': junction, 'Road_Type': road_type, 'Urban_or_Rural_Area': urban
    }
    pred_sev = predict_risk(m_sev, l_sev, input_data, f_cols)
    pred_veh = predict_risk(m_veh, l_veh, input_data, f_cols)
    from recommender import get_safety_adjusted_risk
    adj_sev, overwritten = get_safety_adjusted_risk(pred_sev, input_data)
    
    st.markdown(f"### الخطورة المتوقعة (الذكاء الاصطناعي): **{pred_sev}**")
    if overwritten:
        st.markdown(f"### الخطورة المعدلة (نظام الأمان): **{adj_sev}** 🛡️")
    st.markdown(f"### نوع المركبة المرجح: **{pred_veh}**")
    analysis_res = {'increase_pct': cross_analysis_lighting(df), 'top_dangerous': get_dangerous_locations(df).head(1)}
    preds_res = {'risk': pred_sev, 'time': f"{hour}:00"}
    recs = get_recommendations(analysis_res, preds_res, input_data)
    
    st.markdown("### 💡 إجراءات التدخل الموصى بها")
    for text, level in recs:
        if level == "error": st.error(text)
        elif level == "warning": st.warning(text)
        elif level == "success": st.success(text)
        else: st.info(text)

def sanitize_metrics(metrics_dict):
    import json
    return json.loads(json.dumps(metrics_dict).replace('NaN', '0').replace('Infinity', '0'))

with tab4:
    st.subheader("النتائج التفصيلية لتدريب النموذج")
    colA, colB = st.columns(2)
    with colA:
        st.metric("دقة اختبار المصنف (الخطورة)", f"{metrics_s['test_acc']*100:.2f}%")
        st.metric("دقة تدريب المصنف (الخطورة)", f"{metrics_s['train_acc']*100:.2f}%")
        st.write("تقرير تصنيف الخطورة:")
        st.json(sanitize_metrics(metrics_s['report']))
    with colB:
        st.metric("دقة اختبار المصنف (المركبة)", f"{metrics_v['test_acc']*100:.2f}%")
        st.metric("دقة تدريب المصنف (المركبة)", f"{metrics_v['train_acc']*100:.2f}%")
        st.write("تقرير تصنيف نوع المركبة:")
        st.json(sanitize_metrics(metrics_v['report']))
