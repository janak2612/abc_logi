import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

# Define the feature names as used during training
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input fields for each feature
input_data = {}
input_data['Delivery_Distance'] = st.slider('Delivery Distance', 0.0, 50.0, 25.0)
input_data['Traffic_Congestion'] = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
input_data['Weather_Condition'] = st.slider('Weather Condition (1-5)', 1, 5, 3)
input_data['Delivery_Slot'] = st.slider('Delivery Slot (1-3)', 1, 3, 2)
input_data['Driver_Experience'] = st.slider('Driver Experience (years)', 0, 20, 10)
input_data['Num_Stops'] = st.slider('Number of Stops', 0, 10, 5)
input_data['Vehicle_Age'] = st.slider('Vehicle Age (years)', 0, 10, 5)
input_data['Road_Condition_Score'] = st.slider('Road Condition Score (1-5)', 1, 5, 3)
input_data['Package_Weight'] = st.slider('Package Weight (kg)', 0.0, 50.0, 25.0)
input_data['Fuel_Efficiency'] = st.slider('Fuel Efficiency (km/l)', 0.0, 20.0, 10.0)
input_data['Warehouse_Processing_Time'] = st.slider('Warehouse Processing Time (minutes)', 0, 100, 50)

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction when button is clicked
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('There is a predicted **Delivery Delay**!')
    else:
        st.success('No predicted Delivery Delay.')
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
