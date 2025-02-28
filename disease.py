import google.generativeai as genai
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from prophet import Prophet
from datetime import timedelta
import logging
import sys

# Configure logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Load the model and configure GenAI
classification_model = load_model('mobilenet_crop_disease_best.h5')
index_to_class = {
    0: "Corn__Common_Rust",
    1: "Corn__Gray_Leaf_Spot",
    2: "Corn__healthy",
    3: "Corn__Northern_Leaf_Blight",
    4: "Potato___Early_blight",
    5: "Potato___healthy",
    6: "Potato___Late_blight",
    7: "Rice__Healthy",
    8: "Rice__Leaf_Blast",
    9: "Rice__Neck_Blast",
    10: "Wheat_Brown_Rust",
    11: "Wheat_healthy",
    12: "Wheat_Yellow_Rust"
}

# Configure Gemini AI
genai.configure(api_key="AIzaSyBy0z4UHNPYYgXP9Kkxu0FwaSDecneIOIo")
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 512,
}
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

def predict_image(image_path):
    """Predict disease from image using the classification model."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = classification_model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]
    return index_to_class[predicted_index]

def get_recommendations(disease, soil_type, temperature, rainfall, humidity, n, p, k, ph):
    """Get recommendations from Gemini AI based on disease and conditions."""
    prompt = f"""
    I have the following details for a crop:
    - Disease: {disease}
    - Soil Type: {soil_type}
    - Average Temperature (next 7 days): {temperature}°C 🌡
    - Average Rainfall (next 7 days): {rainfall} mm 🌧
    - Average Humidity (next 7 days): {humidity}% 💧
    - Soil Nutrient Values: N = {n}, P = {p}, K = {k} 🧪
    - Soil pH: {ph}

    Based on these conditions, please provide:
    1. Fertilizer/Manure Recommendations: on the basis of data provide suggest the exact amount and name of fertilizers or manure to be used.
    2. Outbreak Analysis: Do the current soil and weather conditions support the disease? What are the chances of the disease spreading?
    3. Remedies: All possible remedies or treatments to manage or cure the disease.
    4. do not include /n /** these types of things in the output text.
    5. give concise and clear answers.
    6. do not include any unnecessary information.
    """

    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

def get_weather_forecast_averages(file_path='final_dataset.csv'):
    """Get weather forecast averages for the next 7 days."""
    original_stdout = sys.stdout
    sys.stdout = open('nul', 'w')
    
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        current_date = df.index[-1]
        target_date = current_date + timedelta(days=7)
        steps = (target_date - current_date).days
        
        def fit_prophet_and_forecast(df, column, steps):
            series = df[column].interpolate().reset_index()
            series.columns = ['ds', 'y']
            model = Prophet(yearly_seasonality=True)
            model.fit(series)
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            forecast_values = forecast[['ds', 'yhat']].tail(steps)
            return forecast_values['yhat'].values
        
        def fit_prophet_for_rainfall(df, steps):
            rain_data = df.reset_index()[['Date', 'Rainfall_mm']]
            rain_data.columns = ['ds', 'y']
            rain_data['monsoon'] = rain_data['ds'].dt.month.isin([7, 8, 9])
            
            model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='monsoon', period=365.25, fourier_order=10, condition_name='monsoon')
            
            model.fit(rain_data)
            future = model.make_future_dataframe(periods=steps)
            future['monsoon'] = future['ds'].dt.month.isin([7, 8, 9])
            forecast = model.predict(future)
            forecast['yhat'] = np.clip(forecast['yhat'], 0, None)
            forecast['month'] = forecast['ds'].dt.month
            forecast.loc[forecast['month'].isin([7, 8, 9]), 'yhat'] = np.clip(forecast['yhat'], 0, 150)
            forecast.loc[forecast['month'].isin([12, 1, 2]), 'yhat'] = np.clip(forecast['yhat'], 0, 2)
            forecast_values = forecast[['yhat']].tail(steps)
            return forecast_values['yhat'].values
        
        temperature_forecast = fit_prophet_and_forecast(df, 'Temperature_C', steps)
        humidity_forecast = fit_prophet_and_forecast(df, 'Humidity_%', steps)
        rainfall_forecast = fit_prophet_for_rainfall(df, steps)
        
        return {
            'avg_temperature': round(np.mean(temperature_forecast), 2),
            'avg_humidity': round(np.mean(humidity_forecast), 2),
            'avg_rainfall': round(np.mean(rainfall_forecast), 2)
        }
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

def analyze_crop_disease(file_path, soil_type, n, p, k, ph):
    """Main pipeline for crop disease analysis."""
    try:
        # Predict disease from image
        disease = predict_image(file_path)
        
        # Get weather forecasts
        weather_forecast = get_weather_forecast_averages()
        
        # Get recommendations
        recommendations = get_recommendations(
            disease,
            soil_type,
            weather_forecast['avg_temperature'],
            weather_forecast['avg_rainfall'],
            weather_forecast['avg_humidity'],
            n, p, k, ph
        )
        
        return {
            'disease': disease,
            'temperature': weather_forecast['avg_temperature'],
            'humidity': weather_forecast['avg_humidity'],
            'rainfall': weather_forecast['avg_rainfall'],
            'recommendations': recommendations
        }
    except Exception as e:
        raise Exception(f"Error in disease analysis pipeline: {str(e)}")