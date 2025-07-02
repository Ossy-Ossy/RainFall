import numpy as np
import pandas as pd
import joblib
import streamlit as st

x = pd.read_csv("C:\\Users\\hp\\x_rainfall_prediction.csv")
x.drop(x.columns[0],axis = 1 ,inplace = True)
df = pd.read_csv("C:\\Users\\hp\\Downloads\\Rainfall.csv")
df.columns = df.columns.str.strip()

st.write("""
# Welcome To This AI-Based RainFall Prediction System       

This app uses daily weather conditions to predict the occurence of Rainfall  
""")

st.write("***")

scaler = joblib.load("C:\\Users\\hp\\Downloads\\scaler_rainfall.joblib")
model = joblib.load("C:\\Users\\hp\\Downloads\\model_rainfall.joblib")

def rainfall_pred(x,model,scaler,pressure ,dewpoint,humidity,cloud,sunshine,winddirection,windspeed):
    fv = np.zeros(len(x.columns))
    pressure,dewpoint,humidity,cloud = float(pressure),float(dewpoint),float(humidity),float(cloud)
    sunshine,winddirection,windspeed = float(sunshine) ,float(winddirection),float(windspeed)
    scaled_values = scaler.transform(np.array([[pressure, dewpoint,
                                                humidity,cloud,sunshine,winddirection,windspeed]]))[0]
    fv[0:] = scaled_values
    fv = fv.reshape(1,-1)
    prediction = model.predict(fv)
    probab_pred = model.predict_proba(fv)
    if prediction == 0:
        return (f"No Rainfall🌧️⚙️ is predicted with {(probab_pred[0][0]*100).round()}% probability")
    else:
        return (f"Rainfall🌧️⚙️ is detected to occur with {(probab_pred[0][1]*100).round()}% probability")
    
pressure_ch = st.number_input("What is the current atmospheric pressure (in hPa or mb) in your region?",
                           min_value = 980.0,
                           max_value = 1040.0
                           )

dewpoint_ch = st.number_input("What is the dew point temperature (in °C or °F) right now?",
                           min_value = df['dewpoint'].min(),
                           max_value = df['dewpoint'].max()
                           )
humidity_ch = st.number_input("What is the relative humidity percentage (%) in your area?",
                           min_value = df['humidity'].min(),
                           max_value = df['humidity'].max()
                           )
cloud_ch = st.number_input("How much of the sky is covered by clouds (in %)",
                        min_value = df['cloud'].min(),
                        max_value = df['cloud'].max()
                        )
sunshine_ch = st.number_input("How many hours of sunshine did your region receive today?",
                           min_value = df['sunshine'].min(),
                           max_value = df['sunshine'].max()
                           )
winddirection_ch = st.number_input("What direction is the wind coming from(degrees)",
                                min_value = df['winddirection'].min(),
                                max_value = 360.0
                                )
windspeed_ch = st.number_input("What is the current wind speed (in km/h or mph)",
min_value = df['windspeed'].min(),
max_value = df['windspeed'].max()
)

st.sidebar.header("🌧️ About Rainfall")

st.sidebar.write("""
**Key Weather Indicators:**  
- **Low Pressure** (<1010 hPa): Higher rain likelihood.  
- **High Dew Point** (Close to air temp): Saturated air → rain.  
- **Humidity** (>70%): Moisture-rich air supports precipitation.  
- **Cloud Cover** (>50%): Overcast skies often precede rain.  
- **Sunshine** (<3 hours): Cloudy days reduce evaporation.  
- **Wind Direction**: Westerly/Southerly winds may carry moisture.  
- **Wind Speed**: Strong winds (≥20 km/h) can signal storms.  
""")

st.sidebar.header("🔍 How It Works")
st.sidebar.write("""
This Machine Learning model predicts rainfall using real-time weather inputs:  
1. **Atmospheric Pressure** (hPa/mbar)  
2. **Dew Point** (°C/°F)  
3. **Relative Humidity** (%)  
4. **Cloud Cover** (%)  
5. **Sunshine Duration** (hours)  
6. **Wind Direction** (degrees)  
7. **Wind Speed** (km/h/mph)  

*Model trained on historical weather data for accuracy.*  
""")

st.sidebar.header("💡 Why It Matters")

st.sidebar.write("""
Predicting rainfall helps:  
- Farmers plan irrigation/harvests.  
- Cities prepare for floods/storms.  
- Travelers avoid disrupted plans.  
- Renewable energy grids optimize power.  
""")

if st.button("RainFall Occurence"):
    result = rainfall_pred(x,model,scaler,pressure_ch ,dewpoint_ch,humidity_ch,cloud_ch,sunshine_ch,winddirection_ch,windspeed_ch)
    st.success(result)