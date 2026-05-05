import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

#function to create a download link for a Dataframe as CSV file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Prediction CSV File</a>'
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=0, max_value=300
    )
    
    cholesterol = st.number_input(
        "Serum Cholesterol (mg/dl)", min_value=0
    )
    
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        ["<= 120 mg/dl", "> 120 mg/dl"]
    )
    
    resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    
    max_hr = st.number_input(
        "Maximum Heart Rate Achieved", min_value=60, max_value=202
    )
    
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina", ["Yes", "No"]
    )
    
    oldpeak = st.number_input(
        "Oldpeak (ST Depression)", min_value=0.0, max_value=10.0
    )
    
    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )


    # Convert categorical inputs to numeric
sex = 0 if sex == "Male" else 1
chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

# Create a DataFrame with user inputs
input_data = pd.DataFrame({
    #Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})


#algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'GridRandom']
#modelnames = ['tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'gridrf.pkl']

algonames = ['Logistic Regression']
modelnames = ['LogisticRegression.pkl']

predictions = []

def predict_heart_disease(data):
    for modelname in modelnames:
        model = pickle.load(open(modelname, 'rb'))
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions

# Create a submit button to make predictions
if st.button("Submit"):
    st.subheader('Results....')
    st.markdown('------------------------------')
    
    result = predict_heart_disease(input_data)

    for i in range(len(predictions)):
        st.subheader(algonames[i])
        if result[i][0] == 0:
            st.write("No heart disease detected.")
        else:
            st.write("Heart disease detected.")

        st.markdown('------------------------------')

with tab2:
    st.title("Upload CSV file for Bulk Prediction")
    st.subheader("Instructions to note before uploading the file")
    st.info("""1.\n
            2.n\
""")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        #To read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticRegression.pkl', 'rb'))

        #Ensure that the input data has the same columns as the training data
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        
        if set(expected_columns).issubset(input_data.columns):
            
            
            input_data['Prediction LR'] = 0

            for i in range(len(input_data)):
                arr = input_data.iloc[i, :-1].values
                input_data['Prediction LR'][i] = model.predict([arr])[0]
            input_data.to_csv('predictedHeartLR.csv')

            #Display the Prediction
            st.subheader("Predictions:")
            st.write(input_data)

            #Craete a button to download the updateed CSV file
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)

        else:
            st.warning("Please make sure the uploaded CSV file contains the required columns.")
    else:
            st.info("Please upload a CSV file to get predictions.")