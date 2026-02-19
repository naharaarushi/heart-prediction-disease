import streamlit as st
import pandas as pd
import joblib

model=joblib.load('knn_heart.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('column.pkl')

st.title("Hsteart disease prediction by arushi")
st.markdown("provide the following details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("Chest pain type",['TA','ATA','NAP','ASY'])
resting_bp=st.number_input("Resting blood pressure (mm,hg)",80,200)
cholesterol=st.number_input("Cholesterol (mm,dl)",100,600,200)
fasting_bs=st.selectbox("Fasting blood sugar >120 mg/dl",[0,1])
resting_ecg=st.selectbox("Resting egc ",['Normal','ST','LVH'])
max_hr=st.slider("Maximum heart rate achieved",60,220,150)
exercise_angina=st.selectbox("Exercise induced angina ",['Y','N'])
oldpeak=st.slider("Oldpeak",0.0,6.0,1.0)
st_slope=st.selectbox("ST slope ",['Up','Flat','Down'])

if st.button("predict"):
    raw_input = {
        'AGE': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }


    input_df=pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[expected_columns]
    scaler_input=scaler.transform(input_df)
    prediction=model.predict(scaler_input)[0]


    if prediction==0:
        st.success("low risk of heart disease")
    else:
        st.error("high risk of heart disease")