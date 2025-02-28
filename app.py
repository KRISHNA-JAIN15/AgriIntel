import streamlit as st
from streamlit_option_menu import option_menu
import base64
import folium
from streamlit_folium import folium_static
import requests
import pickle
from PIL import Image
import io
import pandas as pd
import plotly.express as px
from auth import get_logged_in_user, get_mongo_client, login_user, register_user, logout
import time
from update import get_soil_record, update_soil_record, format_soil_data_for_update
from prophet import Prophet
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from disease import analyze_crop_disease, get_weather_forecast_averages
import os
from database import get_soil_parameters
from model import CropProfitAnalyzer


st.set_page_config(
    page_title="Crop Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the crop dataset
df = pd.read_csv("Crop_recommendation.csv")
crop_means = df.groupby("label")[["N", "P", "K", "rainfall", "ph", "humidity", "temperature"]].mean()

def water_amount(area_sqft, rainfall_deficit_mm):
    area_m2 = area_sqft * 0.0929  # square feet to square meters
    water_liters = rainfall_deficit_mm * area_m2  # mm deficit to liters
    return water_liters

def soil_amendment_recommendations(crop, current_values, column_names, area_sqft):
    crop = crop.lower()
    if crop not in crop_means.index:
        return f"Crop '{crop}' not found in dataset."
    
    crop_avgs = crop_means.loc[crop]
    differences = crop_avgs - current_values
    
    # Create a dictionary to map column indices to their names
    value_dict = {name: (current, optimal, diff) for name, current, optimal, diff in 
                  zip(column_names, current_values, crop_avgs, differences)}
    
    # Define thresholds for significant deviations
    thresholds = {
        'N': 10,        # Nitrogen threshold
        'P': 5,         # Phosphorus threshold
        'K': 5,         # Potassium threshold
        'rainfall': 50, # Rainfall threshold in mm
        'ph': 0.5,      # pH threshold
        'humidity': 10, # Humidity threshold
        'temperature': 2 # Temperature threshold
    }
    
    recommendations = []
    environment_factors = []
    soil_factors = []
    
    # Check soil nutrients (N, P, K)
    for nutrient in ['N', 'P', 'K']:
        current, optimal, diff = value_dict[nutrient]
        if abs(diff) > thresholds[nutrient]:
            if diff > 0:
                soil_factors.append(f"The soil is deficient in {nutrient}. Current level is {current:.1f}, but optimal level for {crop} is around {optimal:.1f}. Consider adding {nutrient}-rich fertilizer to increase by approximately {abs(diff):.1f} units.")
            else:
                soil_factors.append(f"The soil has excess {nutrient}. Current level is {current:.1f}, but optimal level for {crop} is around {optimal:.1f}. Avoid adding more {nutrient} fertilizers for now.")
        else:
            soil_factors.append(f"The {nutrient} level is within acceptable range for {crop} cultivation (current: {current:.1f}, optimal: {optimal:.1f}).")
    
    # Check pH
    current, optimal, diff = value_dict['ph']
    if abs(diff) > thresholds['ph']:
        if diff > 0:
            soil_factors.append(f"The soil pH is too acidic at {current:.1f}. Optimal pH for {crop} is around {optimal:.1f}. Consider adding agricultural lime to raise the pH by approximately {abs(diff):.1f} units.")
        else:
            soil_factors.append(f"The soil pH is too alkaline at {current:.1f}. Optimal pH for {crop} is around {optimal:.1f}. Consider adding sulfur or organic matter to lower the pH by approximately {abs(diff):.1f} units.")
    else:
        soil_factors.append(f"The soil pH of {current:.1f} is within optimal range for {crop} cultivation (optimal: {optimal:.1f}).")
    
    # Check environmental factors
    for factor in ['rainfall', 'humidity', 'temperature']:
        current, optimal, diff = value_dict[factor]
        if abs(diff) > thresholds[factor]:
            if factor == 'rainfall':
                if diff > 0:
                    environment_factors.append(f"The area receives insufficient rainfall. Current average is {current:.1f}mm, but {crop} typically requires around {optimal:.1f}mm. Consider supplemental irrigation of approximately {abs(water_amount(area_sqft,diff)):.1f} litres")
                else:
                    environment_factors.append(f"The area receives excess rainfall. Current average is {current:.1f}mm, but {crop} typically grows best with around {optimal:.1f}mm. Consider improved drainage systems to manage excess {abs(diff):.1f}mm.")
            elif factor == 'humidity':
                if diff > 0:
                    environment_factors.append(f"The humidity level is too low at {current:.1f}%. Optimal humidity for {crop} is around {optimal:.1f}%. This may increase water requirements.")
                else:
                    environment_factors.append(f"The humidity level is too high at {current:.1f}%. Optimal humidity for {crop} is around {optimal:.1f}%. This may increase disease risk.")
            elif factor == 'temperature':
                if diff > 0:
                    environment_factors.append(f"The temperature is too cool at {current:.1f}°C. Optimal temperature for {crop} is around {optimal:.1f}°C. Consider adjusting planting date or using row covers.")
                else:
                    environment_factors.append(f"The temperature is too warm at {current:.1f}°C. Optimal temperature for {crop} is around {optimal:.1f}°C. Consider providing shade or adjusting planting date.")
        else:
            environment_factors.append(f"The {factor} level of {current:.1f} is suitable for {crop} cultivation (optimal: {optimal:.1f}).")
    
    # Create final recommendation summary
    summary = f"## Soil Amendment Recommendations for {crop.title()} Cultivation\n\n"
    
    summary += "### Soil Nutrient Analysis\n"
    for rec in soil_factors:
        summary += f"- {rec}\n"
    
    summary += "\n### Environmental Factors\n"
    for rec in environment_factors:
        summary += f"- {rec}\n"
    
    # Overall suitability assessment
    significant_deviations = sum(1 for factor, (current, optimal, diff) in value_dict.items() 
                               if abs(diff) > thresholds.get(factor, 0))
    total_factors = len(value_dict)
    
    if significant_deviations == 0:
        summary += "\n### Overall Assessment\n"
        summary += f"Your conditions are excellent for {crop} cultivation! All measured parameters are within optimal ranges.\n"
    elif significant_deviations <= 2:
        summary += "\n### Overall Assessment\n"
        summary += f"Your conditions are generally favorable for {crop} cultivation with minor adjustments needed as noted above.\n"
    elif significant_deviations <= 4:
        summary += "\n### Overall Assessment\n"
        summary += f"Your conditions require moderate adjustments for optimal {crop} cultivation. Address the factors mentioned above to improve potential yield.\n"
    else:
        summary += "\n### Overall Assessment\n"
        summary += f"Your conditions present significant challenges for {crop} cultivation. Consider implementing the recommendations above or evaluating alternative crops better suited to your conditions.\n"
    
    return summary
    # [Previous soil_analysis_page code remains exactly the same]


def local_css():
    st.markdown("""
    <style>
        /* 🌟 General Background & Text Colors */
        .stApp {
            background: linear-gradient(135deg, #eef2f3 0%, #dce2e6 100%);
        }
        
        /* 🏠 Page Container */
        .css-18e3th9 {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* 🏆 Header Styling */
        .main-header {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: #2c786c;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        /* 🔖 Subheader */
        .sub-header {
            font-family: 'Poppins', sans-serif;
            font-weight: 400;
            color: #185a4b;
            font-size: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* 🎯 Custom Buttons */
        .custom-button {
            display: inline-block;
            padding: 0.7rem 1.8rem;
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
            text-decoration: none;
            color: white;
            background: linear-gradient(135deg, #2c786c, #44a08d);
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            margin: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .custom-button:hover {
            background: linear-gradient(135deg, #185a4b, #2c786c);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            transform: translateY(-3px);
        }
        
        /* 🔲 Feature Cards */
        .feature-card {
            background: linear-gradient(120deg, #ffffff 0%, #f3f4f6 100%);
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            transform: translateY(-5px);
            background: linear-gradient(120deg, #f8f9fa 0%, #e3e6e9 100%);
        }
        
        .feature-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #2c786c;
            margin-bottom: 0.5rem;
        }
        
        .feature-description {
            font-size: 1rem;
            color: #4a4a4a;
        }
        
        /* 📌 Footer */
        .footer {
            text-align: center;
            padding: 1.2rem;
            font-size: 0.9rem;
            color: #666;
            margin-top: 2rem;
            border-top: 2px solid #ddd;
        }
        
        /* 📍 Navbar */
        .nav-link {
            color: #2c786c;
            font-weight: 600;
        }
        
        /* 📊 Dashboard & Form Styling */
        .form-container, .dashboard-card {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .form-container:hover, .dashboard-card:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* 📱 Responsive Layout */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            .sub-header {
                font-size: 1.25rem;
            }
        }
        
        /* 📏 Section Divider */
        .section-divider {
            height: 3px;
            background: linear-gradient(90deg, transparent, #2c786c, transparent);
            margin: 3rem 0;
        }

        /* 🖍️ Streamlit Elements */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 8px;
            transition: all 0.3s ease;
        }

        .stTextInput:hover input,
        .stSelectbox:hover div,
        .stNumberInput:hover input {
            border: 1px solid #2c786c;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2c786c, #44a08d);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #185a4b, #2c786c);
            transform: translateY(-3px);
        }

        /* 📌 Sidebar */
        .css-1d391kg {
            background: #eef2f3;
        }

        /* 🌱 Titles & Headers */
        h1, h2, h3 {
            color: #2c786c;
            font-family: 'Poppins', sans-serif;
        }

        .stTitle {
            color: #2c786c;
            font-weight: 700;
        }

        /* 🎭 Card Rows */
        .card-row {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        /* 🖼️ Image Styling */
        .image-container {
            width: 100%;
            height: 220px;
            overflow: hidden;
            border-radius: 10px;
            margin-top: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }

        .image-container img:hover {
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)



def create_placeholder_image(width, height, color="green", format="PNG"):
    img = Image.new('RGB', (width, height), color=color)
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


banner_img = create_placeholder_image(1200, 250, color="#2e8b57")
crop_img = create_placeholder_image(400, 300, color="#4CAF50")
soil_img = create_placeholder_image(400, 300, color="#8BC34A")
disease_img = create_placeholder_image(400, 300, color="#CDDC39")

def home_page():
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
    }
    
    .hero-section {
        padding: 2rem;
        background: linear-gradient(135deg, #1e4d2b, #2e8b57);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease-in-out;
    }
    
    .hero-section:hover {
        transform: scale(1.02);
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #ddd;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }

    .feature-icon {
        font-size: 3rem;
        color: #1e4d2b;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.7rem;
        font-weight: bold;
        color: #2e8b57;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        color: #555;
        font-size: 1rem;
        line-height: 1.6;
    }

    .image-container {
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .image-container img {
        width: 100%;
        height: auto;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease;
        border-radius: 10px;
    }

    .image-container:hover img {
        transform: scale(1.05);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }

    .stat-card {
        background: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #2e8b57;
        transition: transform 0.3s ease, background 0.3s ease;
    }

    .stat-card:hover {
        transform: scale(1.05);
        background: #e8f5e9;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1e4d2b;
    }

    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }

    .step-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .step-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    .testimonial-card {
        background: #fafafa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #2e8b57;
        transition: background 0.3s ease, transform 0.3s ease;
    }

    .testimonial-card:hover {
        background: #e8f5e9;
        transform: scale(1.05);
    }

    .testimonial-text {
        font-size: 1.1rem;
        color: #444;
        font-style: italic;
    }

    .testimonial-author {
        font-size: 0.9rem;
        color: #2e8b57;
        margin-top: 0.5rem;
        font-weight: bold;
    }

    .info-section {
        background: #f4f4f4;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .info-section h2 {
        color: #1e4d2b;
    }

    .info-section ul {
        list-style-type: none;
        padding: 0;
    }

    .info-section ul li {
        padding: 0.5rem 0;
        color: #555;
        font-size: 1rem;
    }

    .cta-section {
        text-align: center;
        margin: 3rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, #1e4d2b, #2e8b57);
        border-radius: 10px;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease-in-out;
    }

    .cta-section:hover {
        transform: scale(1.03);
    }

    .cta-section h2 {
        margin-bottom: 1rem;
    }

    .cta-section p {
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    .cta-section .cta-button {
        background: white;
        color: #2e8b57;
        padding: 0.8rem 1.5rem;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: background 0.3s ease, color 0.3s ease;
        display: inline-block;
    }

    .cta-section .cta-button:hover {
        background: #e8f5e9;
        color: #1e4d2b;
    }
    
    </style>
""", unsafe_allow_html=True)


    
    st.markdown("""
        <div class="hero-section">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Welcome to Crop Assistant</h1>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">Your intelligent farming companion for better crop management and yield optimization</p>
        </div>
    """, unsafe_allow_html=True)

    
    st.markdown('<div class="card-row">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
   
    with col1:
        with st.container():
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🌾</div>
                    <div class="feature-title">Disease Detection</div>
                    <div class="feature-description">
                        Early detection of crop diseases using advanced image recognition technology.
                        Protect your crops before it's too late.
                    </div>
                    <div class="image-container">
                        <img src="https://images.unsplash.com/photo-1574943320219-553eb213f72d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2340&q=80"
                             alt="Disease Detection">
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Disease Detection", key="disease_btn", use_container_width=True):
                st.session_state.page = "Disease Detection"
                st.session_state.sidebar_selection = "Disease Detection"
                st.rerun()

    with col2:
        with st.container():
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🌡️</div>
                    <div class="feature-title">Weather Forecast</div>
                    <div class="feature-description">
                        Get accurate weather predictions and alerts to plan your farming activities effectively.
                    </div>
                    <div class="image-container">
                        <img src="https://images.unsplash.com/photo-1592210454359-9043f067919b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2340&q=80"
                             alt="Weather Forecast">
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Weather Forecast", key="weather_btn", use_container_width=True):
                st.session_state.page = "Weather Forecast"
                st.session_state.sidebar_selection = "Weather Forecast"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown('<div class="card-row">', unsafe_allow_html=True)
    col3, col4 = st.columns(2, gap="large")
   
    with col3:
        with st.container():
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🌱</div>
                    <div class="feature-title">Soil Analysis</div>
                    <div class="feature-description">
                        Comprehensive soil health analysis and recommendations for optimal crop growth.
                    </div>
                    <div class="image-container">
                        <img src="https://images.unsplash.com/photo-1464226184884-fa280b87c399?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2340&q=80"
                             alt="Soil Analysis">
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Soil Analysis", key="soil_btn", use_container_width=True):
                st.session_state.page = "Soil Analysis"
                st.session_state.sidebar_selection = "Soil Analysis"
                st.rerun()

    with col4:
        with st.container():
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Best Crop Selection</div>
                    <div class="feature-description">
                        AI-powered crop recommendations based on your soil and climate conditions.
                    </div>
                    <div class="image-container">
                        <img src="https://images.unsplash.com/photo-1574943320219-553eb213f72d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2340&q=80"
                             alt="Best Crop">
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Best Crop", key="crop_btn", use_container_width=True):
                st.session_state.page = "Best Crop"
                st.session_state.sidebar_selection = "Best Crop"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown("<h2 style='text-align: center; color: #1e4d2b; margin: 2rem 0;'>Our Impact</h2>", unsafe_allow_html=True)
   
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">10,000+</div>
                <div class="stat-label">Farmers Helped</div>
            </div>
        """, unsafe_allow_html=True)
       
    with col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">91.3%</div>
                <div class="stat-label">Accuracy Rate</div>
            </div>
        """, unsafe_allow_html=True)
       
    with col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">22+</div>
                <div class="stat-label">Crop Varieties</div>
            </div>
        """, unsafe_allow_html=True)
       
    with col4:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">Support Available</div>
            </div>
        """, unsafe_allow_html=True)

    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1e4d2b; margin: 2rem 0;'>How It Works</h2>", unsafe_allow_html=True)
   
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.markdown("""
            <div class="step-card">
                <h3 style="color: #2e8b57;">1. Input Your Data</h3>
                <p>Upload images of your crops, enter soil parameters, or provide weather data for analysis.</p>
            </div>
        """, unsafe_allow_html=True)
       
    with col2:
        st.markdown("""
            <div class="step-card">
                <h3 style="color: #2e8b57;">2. AI Analysis</h3>
                <p>Our advanced AI models analyze your data to provide accurate insights and recommendations.</p>
            </div>
        """, unsafe_allow_html=True)
       
    with col3:
        st.markdown("""
            <div class="step-card">
                <h3 style="color: #2e8b57;">3. Get Results</h3>
                <p>Receive detailed reports and actionable recommendations for your farming decisions.</p>
            </div>
        """, unsafe_allow_html=True)

    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1e4d2b; margin: 2rem 0;'>What Farmers Say</h2>", unsafe_allow_html=True)
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("""
            <div class="testimonial-card">
                <div class="testimonial-text">
                    "The disease detection feature helped me save my tomato crop from late blight. This tool is a game-changer for modern farming!"
                </div>
                <div class="testimonial-author">- Rajesh Kumar, Maharashtra</div>
            </div>
        """, unsafe_allow_html=True)
       
    with col2:
        st.markdown("""
            <div class="testimonial-card">
                <div class="testimonial-text">
                    "The weather forecasting is incredibly accurate. I've been able to plan my farming activities much better now."
                </div>
                <div class="testimonial-author">- Amit Patel, Gujarat</div>
            </div>
        """, unsafe_allow_html=True)

    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="info-section">
            <h2 style="color: #1e4d2b; margin-bottom: 1rem;">Why Choose Crop Assistant?</h2>
            <ul style="color: #666; line-height: 1.6;">
                <li>Advanced AI technology for accurate predictions and recommendations</li>
                <li>Comprehensive analysis of multiple farming aspects</li>
                <li>User-friendly interface designed for farmers</li>
                <li>Regular updates with latest agricultural research</li>
                <li>24/7 support for all your farming queries</li>
                <li>Customized recommendations based on your local conditions</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
        <div style="text-align: center; margin: 3rem 0; padding: 2rem; background: linear-gradient(135deg, #1e4d2b, #2e8b57); border-radius: 10px; color: white;">
            <h2 style="margin-bottom: 1rem;">Start Optimizing Your Farm Today</h2>
            <p style="margin-bottom: 2rem;">Join the community of smart farmers using technology to improve their yields</p>
            <div style="font-size: 0.9rem; color: #eee;">
                Get started with our free basic features or upgrade for advanced analytics
            </div>
        </div>
    """, unsafe_allow_html=True)

def disease_detection_page():
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f7f7f7;
    }
    
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #186a3b;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #444;
        margin-bottom: 2rem;
    }
    
    .form-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease-in-out;
    }
    
    .form-container:hover {
        transform: scale(1.02);
    }
    
    .result-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        margin-top: 1rem;
    }
    
    .disease-title {
    font-size: 1.8rem;
    font-weight: bold;
    color: #c0392b;
    text-align: center;
    margin-bottom: 1rem;
    background: none; /* Removes extra box */
    padding: 0; /* Removes padding */
}
    
    .section-title {
    font-size: 1.4rem;
    font-weight: bold;
    color: #186a3b;
    margin-top: 1.5rem;
    background: none;
    padding: 0;
}
    
    .forecast-card {
        background: #eafaf1;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #2e8b57;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .forecast-card:hover {
        transform: scale(1.05);
    }
    
    .forecast-value {
    font-size: 1.5rem;
    color: #1e4d2b;
}

/* Recommendation Boxes - Removed Background */
.recommendation-box {
    background: none;
    padding: 0;
    border-radius: 0;
    box-shadow: none;
    margin-top: 1rem;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

    
    .recommendation-box:hover {
        transform: scale(1.02);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        text-align: center;
        margin-top: 2rem;
    }
    
    .feature-card:hover {
        transform: scale(1.02);
    }
    
    .feature-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #2e8b57;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #555;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    ol {
        padding-left: 1.5rem;
    }
    
    ol li {
        margin-bottom: 0.5rem;
        color: #444;
    }
    
    .cta-section {
        text-align: center;
        margin: 3rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, #1e4d2b, #2e8b57);
        border-radius: 10px;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease-in-out;
    }

    .cta-section:hover {
        transform: scale(1.03);
    }

    .cta-section h2 {
        margin-bottom: 1rem;
    }

    .cta-section p {
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    .cta-section .cta-button {
        background: white;
        color: #2e8b57;
        padding: 0.8rem 1.5rem;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: background 0.3s ease, color 0.3s ease;
        display: inline-block;
    }

    .cta-section .cta-button:hover {
        background: #e8f5e9;
        color: #1e4d2b;
    }
    </style>
""", unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Plant Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify plant diseases by uploading an image</p>', unsafe_allow_html=True)
   
    col1, col2 = st.columns([1, 1])
   
    with col1:
        st.markdown("""
        <div class="form-container">
            <h3 style="color: #186a3b; margin-bottom: 1rem;">Upload Plant Image</h3>
        """, unsafe_allow_html=True)
       
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Single Analyze button that will trigger the analysis
        analyze_clicked = st.button("Analyze", use_container_width=True)
       
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col2:
        st.markdown("""
        <div class="form-container">
            <h3 style="color: #186a3b; margin-bottom: 1rem;">Results</h3>
        """, unsafe_allow_html=True)

        if uploaded_file is not None and analyze_clicked:
            with st.spinner("Analyzing image..."):
                temp_path = "temp_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
               
                try:
                    # Get user's soil parameters from MongoDB
                    user_email = st.session_state.get('email')  # Get user email from session
                    soil_params = get_soil_parameters(user_email)
                    
                    if not soil_params:
                        st.warning("No soil parameters found. Please update your soil parameters in the Soil Analysis section.")
                        soil_params = {
                            "soil_type": "sandy",
                            "nitrogen": 20,
                            "phosphorus": 40,
                            "potassium": 60,
                            "ph": 6.5
                        }
                    
                    # Use the user's soil parameters for analysis
                    results = analyze_crop_disease(
                        temp_path,
                        soil_params["soil_type"],
                        soil_params["nitrogen"],
                        soil_params["phosphorus"],
                        soil_params["potassium"],
                        soil_params["ph"]
                    )
                   
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f'<div class="disease-title">Detected Disease: {results["disease"]}</div>', unsafe_allow_html=True)
                   
                    st.markdown('<div class="section-title">7-Day Weather Forecast Averages</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                   
                    with col1:
                        st.markdown(f"""
                            <div class="forecast-card">
                                <div>Temperature</div>
                                <div class="forecast-value">{results['temperature']:.1f}°C</div>
                            </div>
                        """, unsafe_allow_html=True)
                   
                    with col2:
                        st.markdown(f"""
                            <div class="forecast-card">
                                <div>Humidity</div>
                                <div class="forecast-value">{results['humidity']:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                   
                    with col3:
                        st.markdown(f"""
                            <div class="forecast-card">
                                <div>Rainfall</div>
                                <div class="forecast-value">{results['rainfall']:.1f} mm</div>
                            </div>
                        """, unsafe_allow_html=True)
                   
                    st.markdown('</div>', unsafe_allow_html=True)
                   
                    sections = results['recommendations'].split('\n\n')
                    for section in sections:
                        if "Fertilizer" in section or "fertilizer" in section:
                            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                            st.markdown('<div class="section-title">Fertilizer Recommendations</div>', unsafe_allow_html=True)
                            st.write(section.replace("1. Fertilizer/Manure Recommendations:", "").strip())
                            st.markdown('</div>', unsafe_allow_html=True)
                       
                        elif "Outbreak" in section or "outbreak" in section:
                            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                            st.markdown('<div class="section-title">Outbreak Analysis</div>', unsafe_allow_html=True)
                            st.write(section.replace("2. Outbreak Analysis:", "").strip())
                            st.markdown('</div>', unsafe_allow_html=True)
                       
                        elif "Remedies" in section or "remedies" in section:
                            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                            st.markdown('<div class="section-title">Recommended Remedies</div>', unsafe_allow_html=True)
                            st.write(section.replace("3. Remedies:", "").strip())
                            st.markdown('</div>', unsafe_allow_html=True)
                   
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.markdown("""
                <div style="height: 200px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccc; border-radius: 5px;">
                    <p style="color: #666;">Results will appear here</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
   
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">How It Works</div>
        <div class="feature-description">
            <p>Our disease detection system uses advanced image recognition to identify common plant diseases.</p>
            <ol>
                <li>Upload a clear image of the affected plant part</li>
                <li>Our system analyzes the image for disease patterns</li>
                <li>View detection results and treatment recommendations</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

def soil_analysis_page():
    # Add custom CSS at the top of the function
    st.markdown("""
    <style>
        /* Main page styles */
        .main-header {
            color: #186a3b;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            color: #555;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        /* Form and result container styles */
        .form-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            height: 100%;
            margin-bottom: 1.5rem;
        }
        
        .form-title {
            color: #186a3b;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.8rem;
        }
        
        .result-placeholder {
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px dashed #adb5bd;
            border-radius: 8px;
            background-color: #fff;
        }
        
        .placeholder-text {
            color: #6c757d;
            font-style: italic;
        }
        
        /* Info card styles */
        .feature-card {
            background-color: #fff;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
        }
        
        .feature-title {
            color: #186a3b;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
        }
        
        .feature-description p {
            margin-bottom: 1rem;
            color: #495057;
        }
        
        .feature-description ul {
            padding-left: 1.5rem;
        }
        
        .feature-description li {
            margin-bottom: 0.5rem;
            color: #495057;
        }
        
        .feature-description strong {
            color: #186a3b;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #186a3b;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #0e4429;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown('<h1 class="main-header">Soil Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get detailed soil recommendations for your crop</p>', unsafe_allow_html=True)

    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="form-container"><h3 class="form-title">Enter Soil Parameters</h3>', unsafe_allow_html=True)

        # Input fields with default values
        soil_type = st.selectbox("Soil Type", 
            ["Sandy", "Loamy", "Clay", "Silt", "Peat"]
        )
        
        nitrogen = st.number_input("Nitrogen (N) mg/kg", 
            min_value=0, max_value=1000, 
            value=20
        )
        
        phosphorus = st.number_input("Phosphorus (P) mg/kg",
            min_value=0, max_value=1000, 
            value=40
        )
        
        potassium = st.number_input("Potassium (K) mg/kg", 
            min_value=0, max_value=1000, 
            value=60
        )
        
        ph_value = st.number_input("pH Level", 
            min_value=0.0, max_value=14.0, 
            value=6.5,
            step=0.1
        )

        rainfall = st.number_input("Average Rainfall (mm)", 
            min_value=0, max_value=5000, 
            value=200
        )

        humidity = st.number_input("Humidity (%)", 
            min_value=0, max_value=100, 
            value=60
        )

        temperature = st.number_input("Temperature (°C)", 
            min_value=0, max_value=50, 
            value=25
        )

        crop_type = st.selectbox("Select Crop", 
            ["Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", 
             "Mothbeans", "Mungbean", "Blackgram", "Lentil", "Pomegranate", 
             "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", 
             "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee"]
        )

        area_sqft = st.number_input("Area (sq ft)", 
            min_value=100, max_value=100000, 
            value=10000
        )

        analyze_button = st.button("Analyze Soil", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-container"><h3 class="form-title">Analysis Results</h3>', unsafe_allow_html=True)

        if analyze_button:
            try:
                # Create array of current values in the correct order
                current_values = np.array([
                    nitrogen, phosphorus, potassium, 
                    rainfall, ph_value, humidity, temperature
                ])

                # Get recommendations
                recommendations = soil_amendment_recommendations(
                    crop_type, 
                    current_values, 
                    ['N', 'P', 'K', 'rainfall', 'ph', 'humidity', 'temperature'],
                    area_sqft
                )

                # Display recommendations with styling
                st.markdown(recommendations)

            except Exception as e:
                st.error(f"Error analyzing soil parameters: {str(e)}")
        else:
            st.markdown("""
                <div class="result-placeholder">
                    <p class="placeholder-text">Analysis results will appear here</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Add information about soil parameters
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Understanding Soil Parameters</div>
        <div class="feature-description">
            <p>Proper soil analysis helps in understanding your soil's health and making informed decisions about crop management.</p>
            <ul>
                <li><strong>Nitrogen (N):</strong> Essential for leaf growth and green vegetation</li>
                <li><strong>Phosphorus (P):</strong> Important for root development and flowering</li>
                <li><strong>Potassium (K):</strong> Helps in overall plant health and disease resistance</li>
                <li><strong>pH Level:</strong> Affects nutrient availability to plants</li>
                <li><strong>Rainfall:</strong> Important for water management and irrigation planning</li>
                <li><strong>Temperature:</strong> Affects crop growth and development stages</li>
                <li><strong>Humidity:</strong> Influences disease pressure and water requirements</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
def load_model(model_path):
    """Load the model with compatibility handling"""
    try:
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
            
            # If model is a dictionary (newer format)
            if isinstance(model_data, dict):
                return model_data['model']
            return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def best_crop_page():
    user_data = get_logged_in_user()
    if not user_data:
        st.warning("Please login to view crop recommendations")
        return

    st.markdown('<h1 class="main-header">Best Crop Recommendation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find the optimal crops for your soil and climate conditions</p>', unsafe_allow_html=True)
   
    # Get soil data from MongoDB
    try:
        client = get_mongo_client()
        db = client['crop_assistant']
        soil_collection = db['soil_data']
        soil_data = soil_collection.find_one({"email": user_data['email']})
    except Exception as e:
        st.error(f"Error fetching soil data: {str(e)}")
        return
    finally:
        client.close()

    st.markdown("""
    <div class="dashboard-card">
        <div class="feature-title">Current Soil Parameters</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N) mg/kg",
            min_value=0,
            max_value=1000,
            value=int(soil_data.get('nitrogen', 40)) if soil_data else 40
        )
        phosphorus = st.number_input(
            "Phosphorus (P) mg/kg",
            min_value=0,
            max_value=1000,
            value=int(soil_data.get('phosphorus', 50)) if soil_data else 50
        )
        potassium = st.number_input(
            "Potassium (K) mg/kg",
            min_value=0,
            max_value=1000,
            value=int(soil_data.get('potassium', 50)) if soil_data else 50
        )

    with col2:
        ph_value = st.number_input(
            "pH Level",
            min_value=0.0,
            max_value=14.0,
            value=float(soil_data.get('ph', 6.5)) if soil_data else 6.5,
            step=0.1
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Get weather forecast
    try:
        weather_forecast = get_weather_forecast_averages()
        
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-title">Weather Forecast (90-Day Average)</div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Temperature (°C)", f"{weather_forecast['avg_temperature']:.1f}")
        with col2:
            st.metric("Humidity (%)", f"{weather_forecast['avg_humidity']:.1f}")
        with col3:
            st.metric("Rainfall (mm)", f"{weather_forecast['avg_rainfall'] +150:.1f}")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error fetching weather forecast: {str(e)}")
        return


    if st.button("Get Crop Recommendations", use_container_width=True):
        try:
            # Load and verify model
            model = load_model('random_forest_classifier.pkl')
            if model is None:
                st.error("Failed to load the crop prediction model")
                return
            
            # Prepare input values as numpy array
            input_values = np.array([
                nitrogen,
                phosphorus,
                potassium,
                weather_forecast['avg_rainfall'] + 150,
                ph_value,
                weather_forecast['avg_humidity'],
                weather_forecast['avg_temperature']
            ]).reshape(1, -1)  # Reshape for single prediction

            try:
                # Direct prediction first
                prediction_probabilities = model.predict_proba(input_values)[0]
                
                # Create analyzer and get recommendations
                analyzer = CropProfitAnalyzer(model)
                top_crops = analyzer.predict_profit(input_values[0])  # Pass 1D array
                
                # Display results
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">Recommended Crops</div>
                """, unsafe_allow_html=True)

                # Convert to DataFrame
                df = pd.DataFrame(top_crops)
                
                # Format columns
                formatted_df = df.copy()
                formatted_df['confidence'] = formatted_df['confidence'] * 100
                formatted_df['confidence'] = formatted_df['confidence'].apply(lambda x: f"{x:.1f}%")
                formatted_df['revenue'] = formatted_df['revenue'].apply(lambda x: f"₹{x:,.2f}")
                formatted_df['costs'] = formatted_df['costs'].apply(lambda x: f"₹{x:,.2f}")
                formatted_df['monthly_profit'] = formatted_df['monthly_profit'].apply(lambda x: f"₹{x:,.2f}")
                formatted_df['roi'] = formatted_df['roi'].apply(lambda x: f"{x:.1f}%")
                
                formatted_df = formatted_df.sort_values(by='monthly_profit', ascending=True)
                
                # Display the DataFrame
                st.dataframe(
                    formatted_df[[
                        'crop', 
                        'revenue', 
                        'costs', 
                        'monthly_profit', 
                        'roi', 
                        'growing_period'
                    ]],
                    use_container_width=True
                )

                # Add visualization
                fig = px.bar(
                    formatted_df,
                    x='crop',
                    y='monthly_profit',
                    title='Monthly Profit by Crop',
                    color='confidence',  # Confidence determines color
                    color_continuous_scale='RdYlGn',  # Red (low) → Yellow (mid) → Green (high)
                    labels={'monthly_profit': 'Monthly Profit (₹)', 'confidence': 'Confidence (%)'},
                    category_orders={"crop": formatted_df["crop"].tolist()}  # Ensure correct sorting
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("""
                    <style>
                        .info-box {
                            background-color: #f4f4f4;
                            padding: 15px;
                            border-radius: 10px;
                            border-left: 5px solid #4CAF50;
                            font-size: 16px;
                            color: #333;
                            margin-bottom: 20px;
                        }
                    </style>

                    <div class="info-box">
                        <strong>About This Page:</strong> This tool recommends the **best crops** for your soil and climate conditions. 
                        It predicts the **top 10 most profitable crops** based on soil nutrients, rainfall, pH, humidity, and temperature.
                        <br><br>
                        The output includes:
                        <ul>
                            <li><strong>Monthly Profit (₹):</strong> Estimated profit per hectare.</li>
                            <li><strong>Confidence (%):</strong> How suitable each crop is (Red = Low, Green = High).</li>
                            <li><strong>Return on Investment (ROI):</strong> The expected return based on costs.</li>
                        </ul>
                        Use this analysis to make informed farming decisions!
                    </div>
                """, unsafe_allow_html=True)


            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.exception(e)  # This will show the full traceback

        except Exception as e:
            st.error(f"Error in model loading: {str(e)}")
            st.exception(e)  # This will show the full traceback
def update_data_page():
    # Add consistent CSS styling that matches the soil analysis page
    st.markdown("""
    <style>
        /* Main page styles */
        .main-header {
            color: #186a3b;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        
        /* Card styles */
        .dashboard-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .feature-title {
            color: #186a3b;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
        }
        
        /* Data display */
        .data-summary {
            background-color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #186a3b;
        }
        
        .data-summary p {
            margin-bottom: 0.5rem;
            color: #495057;
        }
        
        .data-summary b {
            color: #186a3b;
        }
        
        /* Crop list styling */
        .crop-item {
            background-color: #fff;
            padding: 0.8rem;
            border-radius: 8px;
            margin-bottom: 0.8rem;
            border-left: 3px solid #28a745;
        }
        
        .no-crops {
            padding: 1rem;
            background-color: #e9ecef;
            border-radius: 8px;
            color: #6c757d;
            font-style: italic;
            text-align: center;
        }
        
        /* Form section styling */
        .form-section {
            background-color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .section-header {
            color: #186a3b;
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 1rem;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #186a3b;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #0e4429;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
        
        /* Warning/info/success message styling */
        .stAlert {
            border-radius: 8px;
            border: none !important;
            padding: 0.8rem !important;
        }
        
        /* File uploader */
        .uploadedFile {
            border-radius: 8px;
            padding: 1rem;
            background-color: #f1f3f5;
        }
        
        /* Remove button */
        .remove-btn {
            background-color: #dc3545 !important;
            padding: 0.3rem 0.5rem !important;
            font-size: 0.8rem !important;
        }
        
        .remove-btn:hover {
            background-color: #bd2130 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Update Soil Data</h1>', unsafe_allow_html=True)
    
    user_data = get_logged_in_user()
    if not user_data:
        st.warning("Please login to update soil data")
        return
    
    current_data = get_soil_record(user_data['email'])
    
    # Current soil data card
    st.markdown('<div class="dashboard-card"><div class="feature-title">Current Soil Data</div>', unsafe_allow_html=True)
    
    if current_data:
        st.json({
            'land_details': current_data.get('land_details', {}),
            'soil_parameters': current_data.get('soil_parameters', {}),
            'crop_details': current_data.get('crop_details', {'crops_planted': []})
        })
    else:
        st.info("No existing soil data found. Please fill in the form below.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Initialize crop list session state
    if 'crop_list' not in st.session_state:
        crops_from_db = current_data.get('crop_details', {}).get('crops_planted', []) if current_data else []
        st.session_state.crop_list = []
        for crop in crops_from_db:
            planting_date = crop.get('planting_date')
            if isinstance(planting_date, str):
                try:
                    planting_date = datetime.strptime(planting_date, '%Y-%m-%d')
                except ValueError:
                    planting_date = datetime.now()
            elif not isinstance(planting_date, datetime):
                planting_date = datetime.now()
                
            st.session_state.crop_list.append({
                'crop_name': crop.get('crop_name'),
                'acres_allocated': crop.get('acres_allocated'),
                'planting_date': planting_date
            })
    
    # Calculate acreage information
    total_acres = current_data.get('land_details', {}).get('total_acres', 0.0) if current_data else 0.0
    used_acres = sum(crop.get('acres_allocated', 0.0) for crop in st.session_state.crop_list)
    remaining_acres = total_acres - used_acres
    
    # Manage crops card
    st.markdown('<div class="dashboard-card"><div class="feature-title">Manage Crops</div>', unsafe_allow_html=True)
    
    # Acreage summary with improved styling
    st.markdown(f"""
        <div class="data-summary">
            <p>Total Farm Size: <b>{total_acres}</b> acres</p>
            <p>Allocated Area: <b>{used_acres}</b> acres</p>
            <p>Remaining Area: <b>{remaining_acres}</b> acres</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Currently planted crops
    if st.session_state.crop_list:
        st.markdown('<div class="section-header">Currently Planted Crops</div>', unsafe_allow_html=True)
        for i, crop in enumerate(st.session_state.crop_list):
            # Start crop item container
            st.markdown('<div class="crop-item">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 0.5])
            with col1:
                st.text(f"Crop: {crop['crop_name']}")
            with col2:
                st.text(f"Acres: {crop['acres_allocated']}")
            with col3:
                if isinstance(crop['planting_date'], datetime):
                    display_date = crop['planting_date'].strftime('%Y-%m-%d')
                else:
                    display_date = str(crop['planting_date'])
                st.text(f"Date: {display_date}")
            with col4:
                # Custom class for remove button
                if st.button(f"✕", key=f"remove_{i}", help=f"Remove {crop['crop_name']}"):
                    st.session_state.crop_list.pop(i)
                    st.rerun()
            
            # End crop item container
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-crops">No crops currently planted</div>', unsafe_allow_html=True)
    
    # Add new crop section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Add New Crop</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_crop_name = st.text_input("New Crop Name")
    with col2:
        new_crop_acres = st.number_input("Acres Allocated",
                                       min_value=0.0,
                                       max_value=remaining_acres,
                                       value=min(remaining_acres, 1.0))
    with col3:
        new_crop_date = st.date_input(
            "Planting Date",
            value=datetime.now().date(),
            min_value=datetime.now().date(),  
            max_value=datetime.now().date() + timedelta(days=365)  
        )
        
    if st.button("Add Crop"):
        if new_crop_name and new_crop_acres > 0:
            if new_crop_acres <= remaining_acres:
                planting_datetime = datetime.combine(new_crop_date, datetime.min.time())
                st.session_state.crop_list.append({
                    'crop_name': new_crop_name,
                    'acres_allocated': new_crop_acres,
                    'planting_date': planting_datetime
                })
                st.success(f"Added {new_crop_name} using {new_crop_acres} acres")
                st.rerun()
            else:
                st.error(f"Not enough remaining acres. Only {remaining_acres} acres available.")
        else:
            st.warning("Please enter crop name and valid acres")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Update soil information card
    st.markdown('<div class="dashboard-card"><div class="feature-title">Update Soil Information</div>', unsafe_allow_html=True)
    
    with st.form("soil_update_form"):
        # Land details section
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Land Details</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_acres = st.number_input(
                "Total Acres",
                min_value=used_acres,
                value=max(current_data.get('land_details', {}).get('total_acres', 0.0), used_acres) if current_data else used_acres
            )
        with col2:
            location = st.text_input(
                "Location",
                value=current_data.get('land_details', {}).get('location', '') if current_data else ''
            )
        with col3:
            soil_type = st.selectbox(
                "Soil Type",
                options=['Clay', 'Sandy', 'Loamy', 'Silt', 'Peat', 'Chalk', 'Other'],
                index=0 if not current_data else ['Clay', 'Sandy', 'Loamy', 'Silt', 'Peat', 'Chalk', 'Other'].index(
                    current_data.get('land_details', {}).get('soil_type', 'Clay')
                )
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Soil parameters section  
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Soil Parameters (JSON File)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Soil Parameters JSON", type=['json'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Update Soil Data")
        
        if submitted:
            try:
                soil_parameters = {}
                if uploaded_file is not None:
                    soil_parameters = json.load(uploaded_file)
                
                form_data = {
                    'total_acres': total_acres,
                    'location': location,
                    'soil_type': soil_type,
                    'crops': [{
                        'crop_name': crop['crop_name'],
                        'acres_allocated': crop['acres_allocated'],
                        'planting_date': crop['planting_date'].strftime('%Y-%m-%d') if isinstance(crop['planting_date'], datetime) else str(crop['planting_date'])
                    } for crop in st.session_state.crop_list],
                    **soil_parameters
                }
                
                formatted_data = format_soil_data_for_update(form_data)
                result = update_soil_record(user_data['email'], formatted_data)
                
                if result:
                    st.success("Soil data updated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to update soil data")
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON file format. Please check the file and try again.")
            except Exception as e:
                st.error(f"Error updating soil data: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)


def get_location():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        loc = data.get("loc", "").split(",")  
        if len(loc) == 2:
            return float(loc[0]), float(loc[1])
    except requests.RequestException:
        pass
    return None, None

# Function to get weather data (Replace with your OpenWeatherMap API key)
API_KEY = "d7f4dfd4b827230f1f1b4af2231dd7ac"  
def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def dashboard_page():
    st.markdown('<h1 class="main-header">📊 Farm Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View all your farm analytics in one place</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div class="dashboard-card"><div class="dashboard-value">3</div><div class="dashboard-label">Crops Growing</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="dashboard-card"><div class="dashboard-value">11</div><div class="dashboard-label">Acres Planted</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="dashboard-card"><div class="dashboard-value">3.2K</div><div class="dashboard-label">Yield (kg/acre)</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="dashboard-card"><div class="dashboard-value">2</div><div class="dashboard-label">Alerts</div></div>""", unsafe_allow_html=True)

    df_yield = pd.DataFrame({'Crop': ['Wheat', 'Rice', 'Corn', 'Soybeans'], 'Yield (kg/acre)': [2800, 3500, 4200, 2300]})
    df_monthly = pd.DataFrame({'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 'Rainfall (mm)': [45, 30, 80, 95, 120, 90], 'Temperature (°C)': [15, 16, 18, 22, 26, 28]})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class="dashboard-card"><div class="feature-title">📈 Crop Yield Comparison</div>""", unsafe_allow_html=True)
        fig1 = px.bar(df_yield, x='Crop', y='Yield (kg/acre)', color='Crop', color_discrete_sequence=px.colors.sequential.Greens_r)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="dashboard-card"><div class="feature-title">🌦️ Weather Trends</div>""", unsafe_allow_html=True)
        fig2 = px.line(df_monthly, x='Month', y=['Rainfall (mm)', 'Temperature (°C)'], color_discrete_sequence=['#186a3b', '#f39c12'])
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 🌍 Location & Geofencing
    lat, lon = get_location()
    st.markdown("""<div class="dashboard-card"><div class="feature-title">📍 Your Location & Farm Area</div>""", unsafe_allow_html=True)

    if lat is not None and lon is not None:
        st.success(f"🌍 Your Location: lat {lat}, lon {lon}")

        # Fetch user farm size
        user = get_logged_in_user()
        user_data = get_soil_record(user['email']) if user else None
        total_acres = user_data.get('land_details', {}).get('total_acres', 0.0) if user_data else 0.0
        total_m2 = total_acres * 4046.86  # Convert acres to square meters
        radius_m = np.sqrt(total_m2 / np.pi) if total_acres > 0 else 0

        st.write(f"🛑 Geofence Radius: {radius_m:.2f} meters (Based on {total_acres} acres)")

        # 🌍 Display interactive map
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker(location=[lat, lon], popup="Your Location", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
        folium.Circle(location=[lat, lon], radius=radius_m, color="blue", fill=True, fill_color="blue", fill_opacity=0.3).add_to(m)
        folium_static(m)

    else:
        st.error("❌ Unable to retrieve location. Please check permissions.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 🌤 Live Weather Data
    st.markdown("""<div class="dashboard-card weather-card"><div class="feature-title">🌤 Live Weather Data</div>""", unsafe_allow_html=True)

    if st.button("Get Weather"):
        weather_data = get_weather(lat, lon)
        if weather_data:
            st.success(f"🌍 Weather at lat {lat}, lon {lon}")
            st.write(f"🌡 Temperature: {weather_data['main']['temp']}°C")
            st.write(f"☁ Condition: {weather_data['weather'][0]['description'].title()}")
            st.write(f"💨 Wind Speed: {weather_data['wind']['speed']} m/s")
            st.write(f"🌍 Humidity: {weather_data['main']['humidity']}%")
        else:
            st.error("❌ Could not fetch weather data. Please check API.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 🌟 Add Enhanced CSS
    st.markdown("""
    <style>
    .dashboard-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }

    .dashboard-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .feature-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #186a3b;
        text-align: center;
        margin-bottom: 1rem;
    }

    .weather-card {
        background: linear-gradient(135deg, #eef2f3, #dce2e6);
    }
    </style>
    """, unsafe_allow_html=True)

def generate_alerts(soil_data, crop_data):
    alerts = []
    
    if not soil_data or not crop_data:
        return alerts

    # Soil pH alerts
    ph = soil_data.get('ph', 0)
    if ph < 6.0:
        alerts.append({
            'title': 'Low pH Alert',
            'message': f'Soil pH is {ph:.1f}, which is below optimal range. Consider liming.',
            'color': '#FFEBEE',
            'border': '#D32F2F'
        })
    elif ph > 7.5:
        alerts.append({
            'title': 'High pH Alert',
            'message': f'Soil pH is {ph:.1f}, which is above optimal range. Consider adding sulfur.',
            'color': '#FFF8E1',
            'border': '#FFA000'
        })

    # Nutrient alerts
    if soil_data.get('nitrogen', 0) < 20:
        alerts.append({
            'title': 'Low Nitrogen',
            'message': 'Nitrogen levels are below optimal. Consider adding nitrogen-rich fertilizer.',
            'color': '#E8F5E9',
            'border': '#388E3C'
        })

    # Crop spacing alerts
    total_area = sum(crop.get('acres_allocated', 0) for crop in crop_data.get('crops', []))
    if total_area > crop_data.get('total_acres', 0):
        alerts.append({
            'title': 'Overcrowding Alert',
            'message': 'Total crop area exceeds available land. Review crop spacing.',
            'color': '#E3F2FD',
            'border': '#1976D2'
        })

    return alerts

def weather_forecast_page():
    # Add CSS styles in the head
    st.markdown("""
    <style>
        /* Base styles */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
            background-color: #f9f9f9;
        }
        
        /* Header styles */
        .main-header {
            color: #186a3b;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        
        .sub-header {
            color: #555;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Form container */
        .form-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            border-left: 5px solid #186a3b;
        }
        
        /* Dashboard cards */
        .dashboard-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            border-top: 4px solid #186a3b;
            transition: transform 0.3s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        
        /* Feature titles */
        .feature-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #186a3b;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #186a3b !important;
            color: white !important;
            font-weight: 500 !important;
            border-radius: 5px !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            background-color: #0e5129 !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 4px 4px 0 0;
            padding: 0px 16px;
            background-color: #f1f1f1;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #186a3b !important;
            color: white !important;
        }
        
        /* Date input */
        .stDateInput>div>div {
            border-radius: 5px;
        }
        
        /* Alert styles */
        .alert {
            padding: 0.8rem;
            margin-bottom: 1rem;
            border-radius: 5px;
            font-size: 0.9rem;
        }
        
        .alert-danger {
            background-color: #FFEBEE;
            border-left: 4px solid #D32F2F;
        }
        
        .alert-warning {
            background-color: #FFF3E0;
            border-left: 4px solid #E65100;
        }
        
        .alert-info {
            background-color: #E3F2FD;
            border-left: 4px solid #1976D2;
        }
        
        .alert-success {
            background-color: #E8F5E9;
            border-left: 4px solid #388E3C;
        }
        
        /* Metric cards */
        .metric-card {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border-bottom: 3px solid #186a3b;
            height: 100%;
        }
        
        .metric-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #186a3b;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #333;
            margin: 1rem 0;
        }
        
        .metric-trend-up {
            color: #D32F2F;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .metric-trend-down {
            color: #388E3C;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .metric-trend-neutral {
            color: #757575;
            font-size: 0.9rem;
        }
        
        /* Forecast summary */
        .forecast-summary {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            border: 1px solid #e0e0e0;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .sub-header {
                font-size: 1rem;
            }
            
            .dashboard-card {
                padding: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header
    st.markdown('<h1 class="main-header">Weather Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View detailed weather predictions for your location</p>', unsafe_allow_html=True)
   
    # Load data
    file_path = 'final_dataset.csv'
    df = pd.read_csv(file_path, parse_dates=['Date'])
   
    # Set Date as index
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
   
    # Target columns for forecasting
    target_columns = ['Temperature_C', 'Humidity_%', 'Wind_Speed_kmph', 'UV_Index', 'Atmospheric_Pressure_hPa']
   
    # Form container
    st.markdown("""
    <div class="form-container">
        <h3 style="color: #186a3b; margin-bottom: 1rem;">Forecast Settings</h3>
    """, unsafe_allow_html=True)
   
    # Date selector
    target_date = st.date_input("Select forecast date")
   
    # Generate forecast button
    if st.button("Generate Forecast", use_container_width=True):
        try:
            # Calculate steps needed for forecasting
            steps = calculate_steps_to_forecast(df, target_date)
           
            # Dictionary to store forecasts
            forecasts = {}
           
            # Add spacing
            st.markdown("<br>", unsafe_allow_html=True)
           
            # Top metric cards
            cols = st.columns(3)
            card_metrics = ['Temperature_C', 'Humidity_%', 'Wind_Speed_kmph']
           
            for idx, metric in enumerate(card_metrics):
                forecast_values, _ = fit_prophet_and_forecast(df, metric, steps)
                latest_forecast = forecast_values[-1]
                current_value = df[metric].iloc[-1]
                change = latest_forecast - current_value
                
                # Determine trend styling
                trend_class = "metric-trend-neutral"
                trend_icon = ""
                if change > 0:
                    trend_class = "metric-trend-up"
                    trend_icon = "↑"
                elif change < 0:
                    trend_class = "metric-trend-down"
                    trend_icon = "↓"
               
                with cols[idx]:
                    metric_display = metric.replace('_', ' ').replace('C', '°C')
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{metric_display}</div>
                        <div class="metric-value">{latest_forecast:.1f}</div>
                        <div class="{trend_class}">
                            {trend_icon} {abs(change):.1f} from current
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
           
            # Add spacing
            st.markdown("<br>", unsafe_allow_html=True)
           
            # Forecast tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Temperature", "Humidity", "Wind Speed",
                "UV Index", "Atmospheric Pressure", "Rainfall"
            ])
           
            with tab1:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">Temperature Forecast</div>
                """, unsafe_allow_html=True)
                fit_prophet_and_forecast(df, 'Temperature_C', steps)
                st.markdown("""
                <div class="forecast-summary">
                    Temperature forecasts help you plan for crop protection against extreme conditions. 
                    Monitor closely for heat stress or frost risks.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
               
            with tab2:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">Humidity Forecast</div>
                """, unsafe_allow_html=True)
                fit_prophet_and_forecast(df, 'Humidity_%', steps)
                st.markdown("""
                <div class="forecast-summary">
                    Humidity levels affect plant health, disease risk, and irrigation needs.
                    High humidity increases fungal disease risk, while low humidity may require additional irrigation.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
               
            with tab3:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">Wind Speed Forecast</div>
                """, unsafe_allow_html=True)
                fit_prophet_and_forecast(df, 'Wind_Speed_kmph', steps)
                st.markdown("""
                <div class="forecast-summary">
                    Wind speed impacts spraying operations, pollination, and potential plant damage.
                    Strong winds may require protective measures for delicate crops.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
               
            with tab4:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">UV Index Forecast</div>
                """, unsafe_allow_html=True)
                fit_prophet_and_forecast(df, 'UV_Index', steps)
                st.markdown("""
                <div class="forecast-summary">
                    UV index affects outdoor work planning and can impact certain crops.
                    High UV levels may require protective measures for workers and sensitive plants.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
               
            with tab5:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">Atmospheric Pressure Forecast</div>
                """, unsafe_allow_html=True)
                fit_prophet_and_forecast(df, 'Atmospheric_Pressure_hPa', steps)
                st.markdown("""
                <div class="forecast-summary">
                    Atmospheric pressure trends help predict incoming weather systems.
                    Falling pressure often indicates approaching precipitation or storms.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
               
            with tab6:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="feature-title">Rainfall Forecast</div>
                """, unsafe_allow_html=True)
                fit_prophet_for_rain(df, target_date)
                st.markdown("""
                <div class="forecast-summary">
                    Rainfall predictions are crucial for irrigation planning and field operations.
                    Plan field activities around expected precipitation events.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
           
            # Weather alerts section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-title">Weather Alerts</div>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
           
            # Initialize alerts list
            alerts = []
           
            # Get forecast values for each metric
            temp_forecast, _ = get_forecast_values(df, 'Temperature_C', steps)
            humidity_forecast, _ = get_forecast_values(df, 'Humidity_%', steps)
            wind_forecast, _ = get_forecast_values(df, 'Wind_Speed_kmph', steps)
            uv_forecast, _ = get_forecast_values(df, 'UV_Index', steps)
            pressure_forecast, _ = get_forecast_values(df, 'Atmospheric_Pressure_hPa', steps)
            rainfall_forecast = get_rain_forecast_values(df, target_date)
           
            # Temperature alerts
            if max(temp_forecast) > 35:
                alerts.append("""
                <div class="alert alert-danger">
                    <strong>🔥 High Temperature Alert:</strong> Temperatures expected to exceed 35°C. Take precautions against heat stress for crops and workers. Consider additional irrigation.
                </div>
                """)
            elif min(temp_forecast) < 10:
                alerts.append("""
                <div class="alert alert-info">
                    <strong>❄️ Low Temperature Alert:</strong> Temperatures expected to drop below 10°C. Protect sensitive crops from cold damage. Frost risk for susceptible plants.
                </div>
                """)
           
            # Humidity alerts
            if max(humidity_forecast) > 85:
                alerts.append("""
                <div class="alert alert-warning">
                    <strong>💧 High Humidity Alert:</strong> Humidity levels expected to exceed 85%. High risk of fungal diseases in crops. Consider preventative fungicide application.
                </div>
                """)
            elif min(humidity_forecast) < 30:
                alerts.append("""
                <div class="alert alert-warning">
                    <strong>🏜️ Low Humidity Alert:</strong> Humidity levels expected to drop below 30%. Increase irrigation frequency to prevent plant stress and dehydration.
                </div>
                """)
           
            # Wind alerts
            if max(wind_forecast) > 25:
                alerts.append("""
                <div class="alert alert-danger">
                    <strong>💨 High Wind Alert:</strong> Wind speeds expected to exceed 25 km/h. Secure loose items and protect sensitive crops. Avoid spraying operations during high winds.
                </div>
                """)
           
            # UV alerts
            if max(uv_forecast) > 8:
                alerts.append("""
                <div class="alert alert-warning">
                    <strong>☀️ High UV Alert:</strong> UV Index expected to exceed 8. Ensure adequate protection for outdoor workers and consider shade for sensitive seedlings.
                </div>
                """)
           
            # Pressure alerts
            if max(pressure_forecast) > 1025:
                alerts.append("""
                <div class="alert alert-info">
                    <strong>🌤️ High Pressure System:</strong> Clear weather conditions likely. Good for outdoor activities and harvesting operations. Plan field work during this period.
                </div>
                """)
            elif min(pressure_forecast) < 995:
                alerts.append("""
                <div class="alert alert-warning">
                    <strong>🌧️ Low Pressure System:</strong> Possibility of unstable weather conditions. Monitor for rainfall and potential storms. Consider postponing sensitive operations.
                </div>
                """)
           
            # Rainfall alerts
            if rainfall_forecast['yhat'].max() > 70:
                alerts.append("""
                <div class="alert alert-danger">
                    <strong>🌊 Heavy Rain Alert:</strong> High probability of significant rainfall. Prepare drainage systems and check erosion controls. Delay fertilizer applications.
                </div>
                """)
            elif rainfall_forecast['yhat'].max() > 40:
                alerts.append("""
                <div class="alert alert-warning">
                    <strong>🌧️ Moderate Rain Alert:</strong> Moderate chance of rainfall. Plan activities accordingly and be prepared to adjust irrigation schedules.
                </div>
                """)
           
            # Display alerts or "no alerts" message
            if alerts:
                for alert in alerts:
                    st.markdown(alert, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert alert-success">
                    <strong>✅ No Alerts:</strong> Weather conditions appear normal for the forecast period. Proceed with standard agricultural operations.
                </div>
                """, unsafe_allow_html=True)
           
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add forecast summary section
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-title">Forecast Summary</div>
                <p>
                    This forecast provides predictions based on historical weather data and advanced forecasting models.
                    Use this information to plan your agricultural activities and make informed decisions for crop management.
                    Remember that weather forecasts become less accurate the further into the future they predict.
                </p>
                
            </div>
            """, unsafe_allow_html=True)
           
        except ValueError as e:
            st.error(f"""
            <div class="alert alert-danger">
                <strong>Error:</strong> {str(e)}
                <p>Please select a valid date range and try again.</p>
            </div>
            """, unsafe_allow_html=True)
def sidebar_auth():
    # Add custom CSS for the sidebar authentication
    st.markdown("""
    <style>
        /* Profile container */
        .profile-container {
            text-align: center;
            padding: 1.5rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            border-top: 3px solid #186a3b;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .profile-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        }
        
        /* Avatar circle */
        .avatar-circle {
            width: 70px;
            height: 70px;
            background-color: #186a3b;
            border-radius: 50%;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.8rem;
            font-weight: bold;
            box-shadow: 0 3px 8px rgba(24, 106, 59, 0.3);
            border: 2px solid #f0f0f0;
        }
        
        /* Username */
        .username {
            margin-top: 0.8rem;
            font-weight: 600;
            font-size: 1.1rem;
            color: #333;
        }
        
        /* Email */
        .user-email {
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.2rem;
        }
        
        /* Edit profile link */
        .edit-profile {
            margin-top: 1rem;
            display: inline-block;
            color: #186a3b;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            background-color: rgba(24, 106, 59, 0.1);
            transition: all 0.2s ease;
        }
        
        .edit-profile:hover {
            background-color: rgba(24, 106, 59, 0.2);
            transform: translateY(-2px);
        }
        
        /* Auth box */
        .auth-box {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            border-left: 4px solid #186a3b;
        }
        
        .auth-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #186a3b;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Button styling */
        div[data-testid="stButton"] button {
            background-color: #186a3b !important;
            color: white !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 0.6rem 1rem !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            width: 100% !important;
            margin-top: 0.8rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 5px rgba(24, 106, 59, 0.3) !important;
        }
        
        div[data-testid="stButton"] button:hover {
            background-color: #0e5129 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(24, 106, 59, 0.4) !important;
        }
        
        div[data-testid="stButton"] button:active {
            background-color: #0a3d1f !important;
            transform: translateY(0px) !important;
        }
        
        /* Text inputs */
        div[data-testid="stTextInput"] input {
            border-radius: 5px !important;
            border: 1px solid #e0e0e0 !important;
            padding: 0.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="stTextInput"] input:focus {
            border-color: #186a3b !important;
            box-shadow: 0 0 0 2px rgba(24, 106, 59, 0.2) !important;
        }
        
        /* Radio buttons */
        div[data-testid="stRadio"] > label {
            cursor: pointer !important;
        }
        
        div[data-testid="stRadio"] > div {
            gap: 0 !important;
        }
        
        div[data-testid="stRadio"] > div > div {
            background-color: #f8f9fa !important;
            border: 1px solid #e0e0e0 !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="stRadio"] > div > div:first-child {
            border-radius: 5px 0 0 5px !important;
            border-right: none !important;
        }
        
        div[data-testid="stRadio"] > div > div:last-child {
            border-radius: 0 5px 5px 0 !important;
        }
        
        div[data-testid="stRadio"] > div > div[data-testid*="StyledFullScreenFrame"] {
            background-color: #186a3b !important;
            color: white !important;
        }
        
        /* Alert messages */
        div[data-testid="stAlert"] {
            border-radius: 5px !important;
            padding: 0.8rem !important;
        }
        
        .success-message {
            padding: 0.8rem;
            background-color: #E8F5E9;
            border-left: 4px solid #388E3C;
            border-radius: 5px;
            margin: 1rem 0;
            font-weight: 500;
            color: #2E7D32;
        }
        
        .error-message {
            padding: 0.8rem;
            background-color: #FFEBEE;
            border-left: 4px solid #D32F2F;
            border-radius: 5px;
            margin: 1rem 0;
            font-weight: 500;
            color: #C62828;
        }
        
        /* Divider */
        .auth-divider {
            display: flex;
            align-items: center;
            margin: 1.5rem 0;
            color: #777;
        }
        
        .auth-divider::before,
        .auth-divider::after {
            content: "";
            flex: 1;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .auth-divider span {
            padding: 0 0.8rem;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    user_data = get_logged_in_user()
   
    if user_data:
        # User is logged in - show profile
        initial = user_data['email'][0].upper()
        username = user_data['email'].split('@')[0]
        email = user_data['email']
        
        st.markdown(f"""
        <div class="profile-container">
            <div class="avatar-circle">
                {initial}
            </div>
            <div class="username">{username}</div>
            <div class="user-email">{email}</div>

        </div>
        """, unsafe_allow_html=True)
        
        # Logout button
        if st.button("Logout"):
            logout()
            st.rerun()
    else:
        # User is not logged in - show auth form
        st.markdown("""
        <div class="auth-box">
            <div class="auth-title">Weather Forecast App</div>
        """, unsafe_allow_html=True)
        
        auth_option = st.radio(
            label="Authentication Option",
            options=["Login", "Sign Up"],
            horizontal=True,
            label_visibility="collapsed"
        )
       
        if auth_option == "Login":
            st.markdown("""
            <div style="font-weight: 600; color: #333; margin-bottom: 1rem; text-align: center; font-size: 1.1rem;">
                Login to Your Account
            </div>
            """, unsafe_allow_html=True)
            
            email = st.text_input(
                label="Email Address",
                key="login_email",
                placeholder="Enter your email"
            )
            password = st.text_input(
                label="Password",
                type="password",
                key="login_password",
                placeholder="Enter your password"
            )
            
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
                <div>
                    <label style="display: flex; align-items: center; cursor: pointer; font-size: 0.85rem; color: #555;">
                        <input type="checkbox" style="margin-right: 5px;"> Remember me
                    </label>
                </div>
                <div>
                    <a href="#" style="font-size: 0.85rem; color: #186a3b; text-decoration: none;">Forgot password?</a>
                </div>
            </div>
            """, unsafe_allow_html=True)
           
            if st.button("Login", key="login_button"):
                if login_user(email, password):
                    st.markdown("""
                    <div class="success-message">
                        <strong>Success!</strong> Logged in successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.markdown("""
                    <div class="error-message">
                        <strong>Error!</strong> Invalid email or password.
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="auth-divider">
                <span>OR CONTINUE WITH</span>
            </div>
            
            <div style="display: flex; gap: 10px; margin-top: 1rem;">
                <button style="flex: 1; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 5px; padding: 0.5rem; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <span style="font-size: 1.2rem; margin-right: 5px;">G</span> Google
                </button>
                <button style="flex: 1; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 5px; padding: 0.5rem; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <span style="font-size: 1.2rem; margin-right: 5px;">f</span> Facebook
                </button>
            </div>
            """, unsafe_allow_html=True)
                   
        else:  
            st.markdown("""
            <div style="font-weight: 600; color: #333; margin-bottom: 1rem; text-align: center; font-size: 1.1rem;">
                Create an Account
            </div>
            """, unsafe_allow_html=True)
            
            email = st.text_input(
                label="Email Address",
                key="signup_email",
                placeholder="Enter your email"
            )
            password = st.text_input(
                label="Password",
                type="password",
                key="signup_password",
                placeholder="Enter your password"
            )
            
            confirm_password = st.text_input(
                label="Confirm Password",
                type="password",
                key="confirm_password",
                placeholder="Confirm your password"
            )
            
            st.markdown("""
            <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                <label style="display: flex; align-items: center; cursor: pointer; font-size: 0.85rem; color: #555;">
                    <input type="checkbox" style="margin-right: 5px;"> I agree to the <a href="#" style="color: #186a3b; text-decoration: none;">Terms & Conditions</a>
                </label>
            </div>
            """, unsafe_allow_html=True)
            
            role = "user"  
           
            if st.button("Sign Up", key="signup_button"):
                if register_user(email, password, role):
                    st.markdown("""
                    <div class="success-message">
                        <strong>Success!</strong> Registration successful! You can now log in.
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.markdown("""
                    <div class="error-message">
                        <strong>Error!</strong> Could not complete registration. Please try again.
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="auth-divider">
                <span>OR SIGN UP WITH</span>
            </div>
            
            <div style="display: flex; gap: 10px; margin-top: 1rem;">
                <button style="flex: 1; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 5px; padding: 0.5rem; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <span style="font-size: 1.2rem; margin-right: 5px;">G</span> Google
                </button>
                <button style="flex: 1; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 5px; padding: 0.5rem; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <span style="font-size: 1.2rem; margin-right: 5px;">f</span> Facebook
                </button>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # App info section
        st.markdown("""
        <div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.85rem;">
            <div style="margin-bottom: 0.5rem;">© 2025 Weather Forecast App</div>
            <div>
                <a href="#" style="color: #186a3b; text-decoration: none; margin: 0 0.5rem;">Privacy Policy</a>
                <a href="#" style="color: #186a3b; text-decoration: none; margin: 0 0.5rem;">Terms of Service</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    
    if 'sidebar_selection' not in st.session_state:
        st.session_state.sidebar_selection = "Home"
   
    
    local_css()
   
    
    with st.sidebar:
        st.markdown('<h2 style="text-align: center; color: #186a3b;">Crop Assistant</h2>', unsafe_allow_html=True)
       
        
        sidebar_auth()
       
        st.markdown("<hr>", unsafe_allow_html=True)
       
        
        user_data = get_logged_in_user()
        if user_data:
            selected = option_menu(
                menu_title=None,
                options=["Home", "Disease Detection", "Soil Analysis", "Best Crop", "Weather Forecast", "Update Data", "Dashboard"],
                icons=["house", "bug", "moisture", "flower3", "cloud-sun", "cloud-upload", "speedometer2"],
                menu_icon="leaf",
                default_index=["Home", "Disease Detection", "Soil Analysis", "Best Crop", "Weather Forecast", "Update Data", "Dashboard"].index(st.session_state.sidebar_selection),
                styles={
                    "container": {"padding": "0!important", "background-color": "#F3F4F6"},
                    "icon": {"color": "#2e8b57", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#E6F0E6"},
                    "nav-link-selected": {"background-color": "#C8E6C9"},
                }
            )
            st.session_state.page = selected
            st.session_state.sidebar_selection = selected
        else:
            st.session_state.page = "Home"
            st.session_state.sidebar_selection = "Home"

    
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Disease Detection":
        disease_detection_page()
    elif st.session_state.page == "Soil Analysis":
        soil_analysis_page()
    elif st.session_state.page == "Best Crop":
        best_crop_page()
    elif st.session_state.page == "Weather Forecast":
        weather_forecast_page()
    elif st.session_state.page == "Update Data":
        update_data_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()


def calculate_steps_to_forecast(df, target_date):
    last_date = df.index[-1]
    target_date = pd.to_datetime(target_date)

    
    if target_date <= last_date:
        raise ValueError(f"Target date {target_date} must be after the last known date {last_date}.")

    return (target_date - last_date).days


def get_seasonal_noise(column, month):
    if column == 'Rainfall_mm':
        
        if month in [12, 1, 2]:
            return 0.1
        elif month in [7, 8, 9]:
            return 0.5
        else:
            return 0.2
    return 0.3


def fit_prophet_and_forecast(df, column, steps):
    print(f"Fitting Prophet model for {column}... Forecasting {steps} steps ahead.")

    
    series = df[column].interpolate().reset_index()
    series.columns = ['ds', 'y']

    
    model = Prophet(yearly_seasonality=True)
    model.fit(series)

    
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    
    forecast_values = forecast[['ds', 'yhat']].tail(steps)

    
    noise_std = np.std(series['y'].diff().dropna()) * get_seasonal_noise(column, df.index[-1].month)
    forecast_values['yhat'] += np.random.normal(0, noise_std, size=steps)

    
    if column == 'Rainfall_mm' and df.index[-1].month in [12, 1, 2]:
        forecast_values['yhat'] = np.clip(forecast_values['yhat'], 0, 1)

    
    fig = px.line()
   
    
    fig.add_scatter(x=df.index[-90:],
                   y=df[column].iloc[-90:],
                   name='Original Data',
                   line=dict(color='blue'))
   
    
    fig.add_scatter(x=forecast_values['ds'],
                   y=forecast_values['yhat'],
                   name='Forecast',
                   line=dict(color='red'))
   
    fig.update_layout(
        title=f"Prophet Forecast for {column} with Season-Aware Variability",
        xaxis_title="Date",
        yaxis_title=column,
        showlegend=True
    )
   
    st.plotly_chart(fig)

    return forecast_values['yhat'].values, forecast_values['ds']


def fit_prophet_for_rain(df, target_date):
    steps = calculate_steps_to_forecast(df, target_date)
    print(f"Fitting Prophet model for Rain_Probability_%... Forecasting {steps} steps ahead.")

    
    rain_prob_data = df.reset_index()[['Date', 'Rain_Probability_%']]
    rain_prob_data.columns = ['ds', 'y']

    
    model = Prophet(yearly_seasonality=True, seasonality_mode='additive')

    
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    
    model.fit(rain_prob_data)

    
    future = model.make_future_dataframe(periods=steps)

    
    forecast = model.predict(future)

    
    forecast['yhat'] = np.clip(forecast['yhat'], 0, 100)

    
    fig = px.line()
   
    
    fig.add_scatter(x=df.index[-120:],
                   y=df['Rain_Probability_%'].iloc[-120:],
                   name='Original Data',
                   line=dict(color='green'))
   
    
    fig.add_scatter(x=forecast['ds'][-steps:],
                   y=forecast['yhat'][-steps:],
                   name='Forecast',
                   line=dict(color='orange'))
   
    fig.update_layout(
        title="Prophet Forecast for Rain Probability (%)",
        xaxis_title="Date",
        yaxis_title="Rain Probability (%)",
        showlegend=True
    )
   
    st.plotly_chart(fig)

    return forecast[['ds', 'yhat']]


def get_forecast_values(df, column, steps):
    
    series = df[column].interpolate().reset_index()
    series.columns = ['ds', 'y']

    
    model = Prophet(yearly_seasonality=True)
    model.fit(series)

    
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    
    forecast_values = forecast[['ds', 'yhat']].tail(steps)

    
    noise_std = np.std(series['y'].diff().dropna()) * get_seasonal_noise(column, df.index[-1].month)
    forecast_values['yhat'] += np.random.normal(0, noise_std, size=steps)

    
    if column == 'Rainfall_mm' and df.index[-1].month in [12, 1, 2]:
        forecast_values['yhat'] = np.clip(forecast_values['yhat'], 0, 1)

    return forecast_values['yhat'].values, forecast_values['ds']


def get_rain_forecast_values(df, target_date):
    steps = calculate_steps_to_forecast(df, target_date)

    
    rain_prob_data = df.reset_index()[['Date', 'Rain_Probability_%']]
    rain_prob_data.columns = ['ds', 'y']

    
    model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(rain_prob_data)

    
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    
    forecast['yhat'] = np.clip(forecast['yhat'], 0, 100)

    return forecast[['ds', 'yhat']]

if __name__ == "__main__":
    main()