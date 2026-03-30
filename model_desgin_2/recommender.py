# Define a function to generate structured safety recommendations
def get_recommendations(analysis, preds, inputs):
    # Initialize an empty list to collect individual recommendation messages
    recs = []
    
    # 1. Analyze weather and surface conditions for specific warnings
    # Check if 'rain' (مطر) is present in the recorded weather conditions
    if 'مطر' in str(inputs.get('Weather_Conditions', '')):
        # Append a warning about slippery road surfaces due to rain
        recs.append(("⚠️ تحذير: سطح الطريق منزلق بسبب الأمطار. ينصح بتخفيف السرعة وزيادة مسافة الأمان.", "warning"))
    
    # Check if 'snow' (ثلج) is present in the recorded weather conditions
    if 'ثلج' in str(inputs.get('Weather_Conditions', '')):
        # Append a critical danger alert for icy conditions and limited visibility
        recs.append(("🛑 خطر: تساقط ثلوج. تجنب القيادة إلا للضرورة القصوى واستخدم كاشفات الضباب.", "error"))

    # 2. Analyze lighting conditions for visibility-related alerts
    # Check if 'darkness' (ظلام) is recorded for the given timeframe
    if 'ظلام' in str(inputs.get('Light_Conditions', '')):
        # Append an informational tip about checking vehicle headlights
        recs.append(("💡 تنبيه رؤية: الإضاءة منخفضة في هذا التوقيت. تأكد من سلامة المصابيح الأمامية.", "info"))
        
    # 3. Analyze risk levels based on model predictions
    # Check if the predicted risk level is documented as 'Serious' (خطير) or 'Fatal' (قاتل)
    if preds.get('risk') in ['خطير', 'قاتل']:
        # Append a high-risk emergency alert based on historical data
        recs.append(("🚨 خطر مرتفع: المنطقة والوقت المحددين سجلا حوادث بليغة سابقاً. يرجى الحذر الشديد.", "error"))
    # Handle cases with lower predicted accident risks
    else:
        # Append a confirmation of current safety stability
        recs.append(("✅ خطر منخفض: المنطقة مستقرة حالياً، استمر في الالتزام بقواعد المرور.", "success"))
        
    # Return the finalized list of recommendation tuples
    return recs

# Define a function to logically adjust safety levels based on environmental guards
def get_safety_adjusted_risk(pred_sev, inputs):
    # Initialize the override status flag as false by default
    overridden = False
    
    # Determine the presence of severe atmospheric obstructions
    # Check for fog (ضباب) in the input parameters
    is_fog = 'ضباب' in str(inputs.get('Weather_Conditions', ''))
    # Check for snow (ثلج) in the input parameters
    is_snow = 'ثلج' in str(inputs.get('Weather_Conditions', ''))
    # Check for darkness (ظلام) in the input parameters
    is_dark = 'ظلام' in str(inputs.get('Light_Conditions', ''))
    
    # Apply safety guard: Force escalation if visibility is extremely limited by weather and darkness
    if (is_fog or is_snow) and is_dark:
        # Check if the original prediction was only minor ('Slight')
        if pred_sev == 'بسيط':
            # Escalate the risk to 'Serious' (خطير) and return the override flag
            return 'خطير', True
            
    # Return the existing severity level if conditions do not require escalation
    return pred_sev, False
