
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -----------------------------
# Load saved model and objects
# -----------------------------

model = joblib.load("models/xgb_pm25_model.pkl")
encoder = joblib.load("models/encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="PM2.5 Prediction",
    page_icon="🌫️",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🌫️ PM2.5 Pollution Prediction")

st.write(
    "Enter the environmental and pollution measurements below "
    "to predict the PM2.5 concentration."
)


# -----------------------------
# Input fields
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    year = st.number_input(
        "Year",
        min_value=2013,
        max_value=2017,
        value=2017
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1
    )

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    PM10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=100.0
    )

    SO2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=20.0
    )

    NO2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=50.0
    )


with col2:

    CO = st.number_input(
        "CO",
        min_value=0.0,
        value=1000.0
    )

    O3 = st.number_input(
        "O3",
        min_value=0.0,
        value=50.0
    )

    TEMP = st.number_input(
        "Temperature (°C)",
        value=15.0
    )

    PRES = st.number_input(
        "Atmospheric Pressure",
        value=1010.0
    )

    DEWP = st.number_input(
        "Dew Point Temperature (°C)",
        value=5.0
    )

    RAIN = st.number_input(
        "Rainfall",
        min_value=0.0,
        value=0.0
    )

    WSPM = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=1.5
    )


# -----------------------------
# Categorical inputs
# -----------------------------

st.subheader("Location & Wind Information")

col3, col4 = st.columns(2)

with col3:

    wd = st.selectbox(
        "Wind Direction",
        [
            "N", "NNE", "NE", "ENE",
            "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW",
            "W", "WNW", "NW", "NNW"
        ]
    )

with col4:

    station = st.selectbox(
        "Monitoring Station",
        [
            "Aotizhongxin",
            "Changping",
            "Dingling",
            "Dongsi",
            "Guanyuan",
            "Gucheng",
            "Huairou",
            "Nongzhanguan",
            "Shunyi",
            "Tiantan",
            "Wanliu",
            "Wanshouxigong"
        ]
    )


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict PM2.5"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "year": [year],
        "month": [month],
        "day": [day],
        "hour": [hour],
        "PM10": [PM10],
        "SO2": [SO2],
        "NO2": [NO2],
        "CO": [CO],
        "O3": [O3],
        "TEMP": [TEMP],
        "PRES": [PRES],
        "DEWP": [DEWP],
        "RAIN": [RAIN],
        "WSPM": [WSPM],
        "wd": [wd],
        "station": [station]
    })

    # Separate categorical columns
    categorical_data = input_data[["wd", "station"]]

    numerical_data = input_data[
        [
            "year", "month", "day", "hour",
            "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
        ]
    ]

    # Encode categorical variables
    encoded_data = encoder.transform(categorical_data)

    encoded_columns = encoder.get_feature_names_out(
        ["wd", "station"]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns
    )

    # Combine numerical + encoded categorical features
    final_input = pd.concat(
        [
            numerical_data.reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Ensure exact feature order used during training
    final_input = final_input[feature_names]

    # Make prediction
    prediction = model.predict(final_input)[0]

    # Display result
    st.success(
        f"Predicted PM2.5 concentration: **{prediction:.2f} µg/m³**"
    )
