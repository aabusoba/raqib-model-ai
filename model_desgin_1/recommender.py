# Define a function to adjust the predicted risk level based on extreme environmental conditions
def get_safety_adjusted_risk(pred_risk, context):
    # Extract weather info from context, defaulting to an empty string
    weather = str(context.get('Weather_Conditions', ''))
    # Extract lighting info from context, defaulting to an empty string
    light = str(context.get('Light_Conditions', ''))
    # Extract road surface info from context, defaulting to an empty string
    surface = str(context.get('Road_Surface_Conditions', ''))
    
    # Check for extreme danger combination: Fog combined with Snow or Ice
    if 'ضباب' in weather and ('ثلج' in surface or 'صقيع' in surface):
        # Override risk to 'Fatal' and return adjustment flag
        return 'قاتل', True
    # Check for extreme danger combination: Fog combined with Darkness
    if 'ضباب' in weather and 'ظلام' in light:
        # Upgrade risk to 'Serious' and flag the override
        return 'خطير', True
    # Check for extreme danger combination: Snow combined with Darkness
    if 'ثلج' in surface and 'ظلام' in light:
        # Upgrade risk to 'Serious' and flag the override
        return 'خطير', True
        
    # Return original prediction if no extreme conditions are met
    return pred_risk, False

# Define a function to generate actionable safety recommendations based on analysis and context
def get_recommendations(analysis_results, predictions, context):
    # Initialize an empty list to store recommendation tuples (message, level)
    recommendations = []
    
    # Extract the raw prediction risk level from the predictions dictionary
    orig_risk = predictions.get('risk', '')
    # Get the adjusted risk level and an override indicator using context rules
    risk, is_overridden = get_safety_adjusted_risk(orig_risk, context)
    
    # Check if a safety override was applied
    if is_overridden:
        # Append a critical alert message explaining the risk upgrade
        recommendations.append((f"تعديل أمان: تم رفع مستوى التحذير من '{orig_risk}' إلى '{risk}' نظراً للظروف القاسية (ضباب/ثلج/ظلام).", "error"))
    
    # Handle recommendations for 'Fatal' risk level
    if risk == 'قاتل':
        # Append an emergency intervention recommendation
        recommendations.append(("عالي الخطورة: خطر داهم للحوادث المميتة. يجب نشر سيارات إسعاف ودوريات تدخل سريع فوراً.", "error"))
    # Handle recommendations for 'Serious' risk level
    elif risk == 'خطير':
        # Append a high-alert enforcement recommendation
        recommendations.append(("خطورة مرتفعة: توقع حوادث جسيمة. يجب تفعيل نظام الردار الآلي وزيادة التواجد الأمني.", "warning"))
    
    # Extract environmental factors from the context dictionary
    weather = context.get('Weather_Conditions', '')
    # Extract lighting conditions from the context dictionary
    light = context.get('Light_Conditions', '')
    # Check if weather conditions involve rain or snow
    if 'مطر' in str(weather) or 'ثلج' in str(weather):
        # Append a weather-specific advisory message
        recommendations.append(("تحذير طقس: ظروف انزلاق مرتفعة. ينصح بتقليل السرعة بنسبة 30% واستخدام أضواء الضباب.", "info"))
    # Check if lighting conditions involve darkness
    if 'ظلام' in str(light):
        # Append a visibility-specific infrastructure alert
        recommendations.append(("تنبيه رؤية: إضاءة منخفضة في هذا التاريخ. تأكد من عمل أعمدة الإنارة أو استخدم لوحات إرشادية عاكسة.", "info"))
    
    # Extract the road type classification from the context
    road_type = context.get('Road_Type', '')
    # Check if the location is at a roundabout
    if 'دوار' in str(road_type):
        # Append a traffic management recommendation for junctions
        recommendations.append(("سلامة التقاطعات: الدوارات تشهد ازدحاماً في هذا التوقيت. يفضل تنظيم الحركة يدوياً أو عبر إشارات ذكية.", "info"))
    
    # Extract the numerical hour of the day from the context
    hour = context.get('Hour', 0)
    # Check if the time falls within the early morning fatigued-driving window
    if 0 <= hour <= 5:
        # Append an enforcement alert for driver alertness
        recommendations.append(("أمن الطرق: ساعات الصباح الباكر تشهد حالات إرهاق للسائقين. ننصح بنشر نقاط تفتيش للتحقق من اليقظة.", "info"))
    
    # Retrieve the top dangerous locations from the analysis results
    top_locs = analysis_results.get('top_dangerous')
    # Validate that location data is available and not empty
    if top_locs is not None and not top_locs.empty:
        # Identify the city with the highest accident frequency
        top_city = top_locs.index[0]
        # Append a spatial planning recommendation for the hazardous area
        recommendations.append((f"تحليل مكاني: منطقة {top_city} هي الأكثر تسجيلاً للحوادث حالياً. يلزم مراجعة تخطيط الطرق هناك.", "warning"))

    # Return the collected recommendations or a success message if no risks were found
    return recommendations if recommendations else [("استقرار في المؤشرات: الحفاظ على بروتوكولات المراقبة العادية.", "success")]
