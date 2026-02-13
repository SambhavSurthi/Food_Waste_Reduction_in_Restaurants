import streamlit as st
import pandas as pd
import datetime
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

# Set page config
st.set_page_config(page_title="Restaurant Demand Forecasting", layout="wide")

st.title("🍽️ Restaurant Demand Forecasting")
st.markdown("Predict future demand for better inventory management and waste reduction.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date = st.date_input("Date", min_value=datetime.date(2020, 1, 1))
        day_of_week = st.selectbox("Day of Week", options=[
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        is_weekend_val = st.selectbox("Is Weekend?", options=["No", "Yes"])
        is_weekend = 1 if is_weekend_val == "Yes" else 0
        
    with col2:
        season = st.selectbox("Season", options=["Winter", "Summer", "Monsoon", "Autumn"])
        weather_type = st.selectbox("Weather", options=["Sunny", "Cloudy", "Rainy", "Heavy Rain", "Cold", "Foggy"])
        
    with col3:
        special_occasion = st.selectbox("Special Occasion", options=["No Occasion", "Local Event", "Festival"])
        holiday_val = st.selectbox("Is Holiday?", options=["No", "Yes"])
        holiday = 1 if holiday_val == "Yes" else 0
        
    submit = st.form_submit_button("Predict Demand")

if submit:
    try:
        data = CustomData(
            date=str(date),
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            season=season,
            weather_type=weather_type,
            special_occasion=special_occasion,
            holiday=holiday
        )
        
        pred_df = data.get_data_as_data_frame()
        # st.write("Input Data:", pred_df) # Debugging

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Results - MultiOutput
        # results is a 2D array [[v1, v2, v3, v4, v5, v6]]
        preds = results[0]
        
        st.success("Demand Prediction Successful!")
        st.subheader("Predicted Values")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col4, res_col5, res_col6 = st.columns(3)
        
        with res_col1:
            st.metric("Total Customers Served", f"{preds[0]:.0f}")
        with res_col2:
            st.metric("LPG Usage (kg)", f"{preds[1]:.2f}")
        with res_col3:
            st.metric("Dosa Sold", f"{preds[2]:.0f}")
            
        with res_col4:
            st.metric("Idly Sold", f"{preds[3]:.0f}")
        with res_col5:
            st.metric("Vada Sold", f"{preds[4]:.0f}")
        with res_col6:
            st.metric("Puri Sold", f"{preds[5]:.0f}")

    except Exception as e:
        st.error(f"Error Occurred: {e}")
