import numpy as np
import pickle

average_yields = {
    'rice': 4.0, 'maize': 3.5, 'chickpea': 1.2, 'kidneybeans': 1.1, 'pigeonpeas': 1.0,
    'mothbeans': 0.9, 'mungbean': 1.0, 'blackgram': 0.9, 'lentil': 1.0, 'pomegranate': 15.0,
    'banana': 35.0, 'mango': 10.0, 'grapes': 18.0, 'watermelon': 30.0, 'muskmelon': 25.0,
    'apple': 20.0, 'orange': 15.0, 'papaya': 40.0, 'coconut': 12000, 'cotton': 2.0,
    'jute': 2.5, 'coffee': 1.0
}

market_prices = {
    'rice': 20, 'maize': 15, 'chickpea': 60, 'kidneybeans': 80, 'pigeonpeas': 70,
    'mothbeans': 60, 'mungbean': 65, 'blackgram': 70, 'lentil': 75, 'pomegranate': 80,
    'banana': 25, 'mango': 50, 'grapes': 60, 'watermelon': 15, 'muskmelon': 20,
    'apple': 70, 'orange': 40, 'papaya': 30, 'coconut': 15, 'cotton': 60,
    'jute': 40, 'coffee': 300
}

seed_costs = {
    'rice': 5000, 'maize': 4500, 'chickpea': 6000, 'kidneybeans': 7000, 'pigeonpeas': 5500,
    'mothbeans': 5000, 'mungbean': 5200, 'blackgram': 5000, 'lentil': 6500, 'pomegranate': 50000,
    'banana': 35000, 'mango': 40000, 'grapes': 80000, 'watermelon': 8000, 'muskmelon': 7000,
    'apple': 100000, 'orange': 50000, 'papaya': 15000, 'coconut': 25000, 'cotton': 10000,
    'jute': 6000, 'coffee': 70000
}

maintenance_costs = {
    'rice': 15000, 'maize': 12000, 'chickpea': 10000, 'kidneybeans': 12000, 'pigeonpeas': 10000,
    'mothbeans': 8000, 'mungbean': 9000, 'blackgram': 9000, 'lentil': 9500, 'pomegranate': 80000,
    'banana': 60000, 'mango': 50000, 'grapes': 120000, 'watermelon': 25000, 'muskmelon': 20000,
    'apple': 150000, 'orange': 60000, 'papaya': 40000, 'coconut': 30000, 'cotton': 25000,
    'jute': 15000, 'coffee': 100000
}

fertilizer_costs = {
    'N': 20,  # Cost per kg of Nitrogen
    'P': 30,  # Cost per kg of Phosphorous
    'K': 25   # Cost per kg of Potassium
}

growing_period = {
    'rice': 4, 'maize': 4, 'chickpea': 4, 'kidneybeans': 3, 'pigeonpeas': 5,
    'mothbeans': 3, 'mungbean': 3, 'blackgram': 4, 'lentil': 5, 'pomegranate': 36,
    'banana': 12, 'mango': 60, 'grapes': 24, 'watermelon': 3, 'muskmelon': 3,
    'apple': 60, 'orange': 36, 'papaya': 12, 'coconut': 72, 'cotton': 6,
    'jute': 4, 'coffee': 36
}

# Optimal conditions for each crop (mean values)
optimal_conditions = {
    'rice': [78.89, 47.58, 38.97, 236.18, 6.42, 82.27, 23.68],
    'maize': [77.76, 48.44, 19.79, 84.77, 6.24, 65.09, 22.38],
    'apple': [20.80, 134.22, 199.89, 112.65, 5.93, 92.33, 22.63],
    'banana': [100.23, 82.01, 50.05, 104.62, 5.98, 80.35, 27.37],
    'blackgram': [40.02, 67.47, 19.24, 67.88, 7.13, 65.11, 29.97],
    'chickpea': [40.09, 67.79, 79.92, 80.05, 7.33, 16.86, 18.87],
    'coconut': [21.98, 16.93, 30.59, 175.68, 5.97, 94.84, 27.40],
    'coffee': [101.20, 28.74, 29.94, 158.07, 6.79, 58.87, 25.54],
    'cotton': [117.77, 46.24, 19.56, 80.39, 6.91, 79.84, 23.99],
    'grapes': [23.18, 132.53, 200.11, 69.61, 6.02, 81.87, 23.85],
    'jute': [78.40, 46.86, 39.99, 174.79, 6.73, 79.63, 24.95],
    'kidneybeans': [20.75, 67.54, 20.05, 105.92, 5.74, 21.60, 20.11],
    'lentil': [18.77, 68.36, 19.41, 45.68, 6.92, 64.80, 24.50],
    'mango': [20.07, 27.18, 29.92, 94.70, 5.76, 50.16, 31.20],
    'mothbeans': [21.44, 48.01, 25.19, 51.19, 6.83, 53.16, 18.19],
    'mungbean': [20.09, 47.28, 19.87, 48.40, 6.72, 85.50, 28.52],
    'muskmelon': [100.32, 17.72, 50.08, 24.68, 6.53, 92.34, 28.66],
    'orange': [19.58, 16.55, 10.01, 110.47, 7.01, 92.17, 22.76],
    'papaya': [49.88, 59.05, 50.04, 142.62, 6.74, 92.40, 33.72],
    'pigeonpeas': [20.73, 67.73, 20.29, 149.45, 5.79, 48.06, 27.74],
    'pomegranate': [18.87, 17.85, 30.91, 107.52, 6.42, 90.12, 21.83],
    'watermelon': [99.42, 17.00, 50.22, 50.78, 6.49, 85.16, 25.59]
}

# Define mapping from numeric labels to crop names
# Using the order from label encoder's inverse_transform
crop_index_mapping = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

class CropProfitAnalyzer:
    def __init__(self, model):
        self.model = model
        self.crop_list = list(average_yields.keys())
        
    def load_model(model_path):
        try:
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
                
                if isinstance(model_data, dict):
                    return model_data['model']
                return model_data
        except Exception as e:
            return None    
        
    def predict_profit(self, input_values):
        """
        Predict the top 10 most profitable crops based on soil conditions
        
        input_values: list of [N, P, K, rainfall, pH, humidity, temperature]
        """
        # Get model prediction confidence for each crop
        input_array = np.array([input_values])
        prediction_probabilities = self.model.predict_proba(input_array)[0]
        
        # Get the crop classes from the model
        crop_classes = self.model.classes_
        
        # Create a mapping of crop confidences
        crop_confidences = {crop: 0.01 for crop in self.crop_list}  # Default to 0.01 as minimum
        
        # Map the numeric class predictions to crop names
        for i, class_idx in enumerate(crop_classes):
            crop_name = crop_index_mapping.get(class_idx)
            if crop_name in crop_confidences:
                crop_confidences[crop_name] = max(prediction_probabilities[i], 0.01)  # Ensure minimum confidence
        
        # Calculate fertilizer costs for current input
        n_cost = input_values[0] * fertilizer_costs['N']
        p_cost = input_values[1] * fertilizer_costs['P']
        k_cost = input_values[2] * fertilizer_costs['K']
        current_fertilizer_cost = n_cost + p_cost + k_cost
        
        # Calculate profit for each crop
        crop_profits = []
        for crop in self.crop_list:
            # Basic revenue calculation
            revenue = average_yields[crop] * market_prices[crop] * 10000  # Convert to per hectare
            
            # For coconut, revenue is calculated differently (per nut)
            if crop == 'coconut':
                revenue = average_yields[crop] * market_prices[crop]  # Already per hectare
            
            # Calculate costs
            costs = seed_costs[crop] + maintenance_costs[crop] + current_fertilizer_cost
            
            # Basic profit calculation
            basic_profit = revenue - costs
            
            # Calculate monthly profit
            monthly_profit = basic_profit / growing_period[crop]
            
            # Calculate ROI (Return on Investment)
            roi = (basic_profit / costs) * 100 if costs > 0 else 0
            
            # Calculate monthly ROI
            monthly_roi = roi / growing_period[crop]
            
            # Adjust profit based on confidence
            confidence = crop_confidences[crop]
            adjusted_profit = monthly_profit * confidence
            adjusted_roi = monthly_roi * confidence
            
            crop_profits.append({
                'crop': crop,
                'confidence': confidence,
                'revenue': revenue,
                'costs': costs,
                'basic_profit': basic_profit,
                'monthly_profit': monthly_profit,
                'adjusted_profit': adjusted_profit,
                'roi': roi,
                'monthly_roi': monthly_roi,
                'adjusted_roi': adjusted_roi,
                'growing_period': growing_period[crop]
            })
        
        # Sort by adjusted profit in descending order
        sorted_profits = sorted(crop_profits, key=lambda x: x['adjusted_profit'], reverse=True)
        
        # Return top 10
        return sorted_profits[:10]

# Example usage
def analyze_crop_profits(model_path, input_values):
    model = load_model(model_path)
    analyzer = CropProfitAnalyzer(model)
    top_crops = analyzer.predict_profit(input_values)
    
    # Format and return results
    result_df = pd.DataFrame(top_crops)
    result_df = result_df[['crop', 'confidence', 'revenue', 'costs', 
                           'basic_profit', 'monthly_profit', 'adjusted_profit', 
                           'roi', 'monthly_roi', 'adjusted_roi', 'growing_period']]
    
    # Format percentage columns for better readability
    for col in ['confidence', 'roi', 'monthly_roi', 'adjusted_roi']:
        result_df[col] = result_df[col].apply(lambda x: f"{x:.2f}%")
    
    # Format monetary values for better readability
    for col in ['revenue', 'costs', 'basic_profit', 'monthly_profit', 'adjusted_profit']:
        result_df[col] = result_df[col].apply(lambda x: f"₹{x:.2f}")
    
    return result_df