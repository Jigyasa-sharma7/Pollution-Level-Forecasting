# 🌫️ Pollution Level Forecasting using Machine Learning

A machine learning project for predicting **PM2.5 (Particulate Matter 2.5)** concentration using historical air-quality and meteorological data.

The project includes data cleaning, exploratory data analysis (EDA), feature engineering, machine learning model training, evaluation, and a Streamlit web application for PM2.5 prediction.

---

## 📌 Project Overview

Air pollution is a major environmental concern, and PM2.5 is one of the most important indicators of air quality.

This project uses historical air-quality data collected from multiple monitoring stations to analyze the factors affecting PM2.5 concentration and build a machine learning model capable of predicting PM2.5 levels.

### Main objectives

- Combine air-quality data from multiple monitoring stations.
- Clean and preprocess the dataset.
- Perform exploratory data analysis (EDA).
- Analyze relationships between PM2.5 and other pollutants/weather variables.
- Train and compare machine learning models.
- Select the best-performing model.
- Build a Streamlit application for PM2.5 prediction.

---

## 📊 Dataset

The dataset contains hourly air-quality and meteorological observations collected from **12 monitoring stations**.

### Time period

**March 2013 – February 2017**

### Original dataset

- Rows: **420,768**
- Columns: **19**

### Final modeling dataset

- Rows: **412,028**
- Columns: **19**
- Missing values: **0**
- Duplicate rows: **0**

### Main features

| Feature | Description |
|---|---|
| `PM2.5` | Fine particulate matter concentration — target variable |
| `PM10` | Particulate matter concentration |
| `SO2` | Sulfur dioxide concentration |
| `NO2` | Nitrogen dioxide concentration |
| `CO` | Carbon monoxide concentration |
| `O3` | Ozone concentration |
| `TEMP` | Temperature |
| `PRES` | Atmospheric pressure |
| `DEWP` | Dew point temperature |
| `RAIN` | Rainfall |
| `WSPM` | Wind speed |
| `wd` | Wind direction |
| `station` | Monitoring station |
| `year`, `month`, `day`, `hour` | Time-related features |

---

## 🧹 Data Cleaning

The original dataset contained missing values in several variables.

The cleaning process included:

- Combining data from 12 monitoring stations.
- Creating a unified `datetime` column.
- Handling missing PM2.5 values.
- Handling missing pollutant and meteorological values.
- Filling missing wind-direction values using station-wise information.
- Removing rows with missing target values.
- Checking and removing invalid values where necessary.
- Checking for duplicate rows.
- Converting columns to appropriate data types.

After cleaning:

```text
Final dataset shape: (412028, 19)
Missing values: 0
Duplicate rows: 0
