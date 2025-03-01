# AgriIntel

# 🌱 AI-Powered Farming Assistant  

An intelligent system designed to **empower farmers** with **AI-driven insights**, **real-time data**, and **smart recommendations** for **better decision-making**.  

![1](https://github.com/user-attachments/assets/38bb9bc7-cd1f-4a0e-b69b-0b7c4e5b0245)
![2](https://github.com/user-attachments/assets/48983a36-11a8-4a50-be55-1d6d6f47a9ad)
![3](https://github.com/user-attachments/assets/d5b3cbe7-043d-4936-8559-b139c19f9d43)
![4](https://github.com/user-attachments/assets/bed8ab1a-50b8-4ca4-a605-e289bb7b22a5)


---

## 🚀 Features  

### 🌦️ Weather Forecasting  
- Predicts **temperature, humidity, rainfall, wind speed, UV index, and atmospheric pressure**.  
- Uses **ARIMA & Prophet models** trained on **15 years of data**.  
- **Real-time alerts** for extreme weather conditions.  

### 🌱 Crop Disease Prediction & Recommendations  
- Farmers can **upload a leaf image** to detect diseases.  
- **Mobilenet V2 model** classifies **13 crop diseases** with **91.41% accuracy**.  
- Recommends solutions based on **weather, soil data, and historical trends**.  

### 🧪 Soil Analysis & Best Crop Recommendation  
- **Real-time sensor data** (N, P, K, pH, rainfall, temperature, humidity).  
- **Random Forest model** predicts the best crops with **95%+ accuracy**.  
- **Profit estimations** for different crop options.  

### 📝 Dynamic Data Updation  
- Farmers can **update land details** (crop types, area).  
- The **dashboard reflects changes automatically**.  

### 📊 Smart Dashboard & Geofencing  
- Displays **land area, crops grown, alerts, and weather conditions**.  
- Uses **Google Maps API** for **geofencing & location tracking**.  
- **Live weather updates** for precise farm monitoring.  

---
## 📌 Installation  

### 1️⃣ Clone the Repository  
First, clone the repository and navigate into the project folder:  
```sh
git clone https://github.com/KRISHNA-JAIN15/AgriIntel.git
cd AgriIntel
```

### 2️⃣ Install Dependencies
Ensure Python 3.8+ is installed, then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ Running the streamlit app
Start the application using:
```sh
streamlit run app.py
```


### 🎮 Usage
- 1️⃣ Upload a leaf image for disease detection.
- 2️⃣ Enter soil data (N, P, K, pH, etc.) to get crop recommendations.
- 3️⃣ View weather forecasts and receive alerts for extreme conditions.
- 4️⃣ Monitor farm data through an interactive dashboard.


🛠️ Tech Stack
- Frontend: Streamlit, Plotly, Folium, CSS
- Backend: Python, FastAPI, MongoDB
- Machine Learning Models:
- Prophet (Weather Forecasting)
- Mobilenet V2 (Crop Disease Detection)
- Random Forest (Crop Recommendation)
- APIs: Openweathermap API
