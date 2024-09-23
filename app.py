import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pickle

# Load the saved encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the pre-trained model
model = load_model('my_model.keras')

# Predict Churn function
def predict_churn(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    # Create a DataFrame with input features
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # Encode categorical features
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
    encoded_geo = one_hot_encoder_geo.transform(input_data[['Geography']])
    encoded_geo_df = pd.DataFrame(encoded_geo, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.drop('Geography', axis=1), encoded_geo_df], axis=1)

    # Scale the input features
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)[0][0]
    
    return prediction

# Streamlit user input
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
creditScore = st.slider('CreditScore', 300, 850)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
numOfProducts = st.slider('NumOfProducts', 1, 4)
hasCrCard = st.selectbox('HasCrCard', [0, 1])
isActiveMember = st.selectbox('IsActiveMember', [0, 1])
estimatedSalary = st.number_input('EstimatedSalary')

# Prediction button
if st.button('Predict Churn'):
    churn_probability = predict_churn(creditScore, geography, gender, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember, estimatedSalary)
    if churn_probability > 0.5:
        st.error(f"The predicted probability of churn for this customer is: {churn_probability:.2f}")
    else:
        st.success(f"The predicted probability of churn for this customer is: {churn_probability:.2f}")
