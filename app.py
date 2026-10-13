import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="MindPulse AI", layout="wide")

@st.cache_resource
def load_assets():
    with open('mindpulse_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('mindpulse_explainer.pkl', 'rb') as f: explainer = pickle.load(f)
    with open('mindpulse_encoders.pkl', 'rb') as f: encoders = pickle.load(f)
    return model, explainer, encoders

try:
    model, explainer, encoders = load_assets()
except FileNotFoundError:
    st.error("Assets not found. Please run MindPulse_Training.ipynb first.")
    st.stop()

st.title("MindPulse AI: Prescriptive Health & Habit Engine")

st.sidebar.header("Patient Demographics")
age = st.sidebar.slider("Age", 16, 80, 24)
gender = st.sidebar.selectbox("Gender", [c for c in encoders['Gender'].classes_ if c != 'nan'])
status = st.sidebar.selectbox("Current Status", [c for c in encoders['Working Professional or Student'].classes_ if c != 'nan'])

if status == 'Student':
    profession = st.sidebar.selectbox("Profession", ['Not Applicable'])
else:
    prof_options = [p for p in encoders['Profession'].classes_ if p != 'Not Applicable' and p != 'nan']
    profession = st.sidebar.selectbox("Profession", prof_options)

deg_options = [d for d in encoders['Degree'].classes_ if d != 'nan']
degree = st.sidebar.selectbox("Degree", deg_options)

family_history = st.sidebar.selectbox("Family History of Mental Illness", [c for c in encoders['Family History of Mental Illness'].classes_ if c != 'nan'])

st.sidebar.header("Lifestyle & Habit Parameters")
sleep_opts = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
sleep_duration = st.sidebar.selectbox("Sleep Duration", sleep_opts)

diet_opts = ['Unhealthy', 'Moderate', 'Healthy']
dietary_habits = st.sidebar.selectbox("Dietary Habits", diet_opts)

work_study_hours = st.sidebar.slider("Work/Study Hours (Daily)", 0.0, 16.0, 8.0, 0.5)
financial_stress = st.sidebar.slider("Financial Stress (1-5)", 1.0, 5.0, 3.0, 0.5)
daily_pressure = st.sidebar.slider("Daily Academic/Work Pressure (1-5)", 1.0, 5.0, 3.0, 0.5)
daily_satisfaction = st.sidebar.slider("Daily Study/Job Satisfaction (1-5)", 1.0, 5.0, 3.0, 0.5)

input_data = {
    'Gender': [gender], 'Age': [age], 'Working Professional or Student': [status],
    'Profession': [profession], 'Sleep Duration': [sleep_duration], 'Dietary Habits': [dietary_habits],
    'Degree': [degree], 'Work/Study Hours': [work_study_hours], 'Financial Stress': [financial_stress],
    'Family History of Mental Illness': [family_history], 'Daily_Pressure': [daily_pressure], 'Daily_Satisfaction': [daily_satisfaction]
}

input_df = pd.DataFrame(input_data)
encoded_input = input_df.copy()

# 1. Manual Ordinal Mapping (matching training)
sleep_map = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3}
diet_map = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
encoded_input['Sleep Duration'] = encoded_input['Sleep Duration'].map(sleep_map)
encoded_input['Dietary Habits'] = encoded_input['Dietary Habits'].map(diet_map)

# 2. Label Encoding for Nominal
for col in encoders.keys():
    encoded_input[col] = encoders[col].transform(encoded_input[col].astype(str))

# 3. STRICT FEATURE ALIGNMENT (Prevents background crashes)
encoded_input = encoded_input[model.feature_names_in_]

risk_prob = model.predict_proba(encoded_input)[0][1] * 100
shap_values = explainer.shap_values(encoded_input)

tab1, tab2, tab3 = st.tabs(["Diagnostic Assessment", "AI Feature Attribution (SHAP)", "Habit Action Plan"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted Depression Risk Probability", f"{risk_prob:.1f}%")
        if risk_prob < 40: st.success("Risk Level: LOW / MODERATE")
        elif risk_prob < 70: st.warning("Risk Level: ELEVATED")
        else: st.error("Risk Level: HIGH RISK")
    with col2:
        st.markdown("#### Patient Context Summary")
        st.dataframe(input_df.T, use_container_width=True)

with tab2:
    st.write("This chart explains the AI's logic. **Red bars** show habits increasing the risk score, while **blue bars** show protective factors.")
    
    patient_shap = shap_values[0]
    
    plot_df = pd.DataFrame({
        'Feature': input_df.columns,
        'SHAP Value': patient_shap
    })
    
    plot_df['Abs Impact'] = plot_df['SHAP Value'].abs()
    plot_df = plot_df.sort_values(by='Abs Impact', ascending=True) 
    
    colors = ['#ff4b4b' if val > 0 else '#1c83e1' for val in plot_df['SHAP Value']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df['Feature'], plot_df['SHAP Value'], color=colors)
    ax.set_xlabel('SHAP Value (Impact on Depression Risk)')
    ax.set_title('Personalized Risk Factors')
    
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--') 
    st.pyplot(fig)

with tab3:
    impacts = {col: val for col, val in zip(encoded_input.columns, shap_values[0])}
    top_risks = sorted([item for item in impacts.items() if item[1] > 0], key=lambda x: x[1], reverse=True)

    if not top_risks:
        st.success("Your current lifestyle habits are excellently balanced!")
    else:
        for feat, impact in top_risks[:4]:
            if feat == 'Sleep Duration': st.write("🟢 **Sleep Optimization:** Establish a strict digital curfew 60 minutes before bed.")
            elif feat == 'Work/Study Hours': st.write("🟡 **Workload Management:** Implement the Pomodoro technique (25 min work, 5 min break).")
            elif feat == 'Financial Stress': st.write("🔴 **Stress Mitigation:** Prioritize budgeting applications or financial counseling.")
            elif feat == 'Daily_Pressure': st.write("🔵 **Pressure Regulation:** Introduce micro-meditation sessions into your routine.")
            elif feat == 'Dietary Habits': st.write("🟣 **Nutritional Psychiatry:** Replace processed foods with complex carbohydrates.")
            elif feat == 'Daily_Satisfaction': st.write("🟠 **Engagement Restructuring:** Consider discussing role adjustments with supervisors or seeking lateral academic changes.")