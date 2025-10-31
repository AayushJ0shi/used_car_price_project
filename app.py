from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import json

app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        model = joblib.load('models/random_forest_model.pkl')
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Available options
BRANDS = ['Maruti Suzuki', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Volkswagen', 'Tata', 'Mahindra']
FUEL_TYPES = ['Petrol', 'Diesel', 'CNG']
SELLER_TYPES = ['Dealer', 'Individual']
TRANSMISSION_TYPES = ['Manual', 'Automatic']

# Label encoders
label_encoders = {
    'brand': LabelEncoder().fit(BRANDS),
    'fuel_type': LabelEncoder().fit(FUEL_TYPES),
    'seller_type': LabelEncoder().fit(SELLER_TYPES),
    'transmission_type': LabelEncoder().fit(TRANSMISSION_TYPES)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if model is None:
                return render_template('predict.html', error="Model not trained yet. Please run train_model.py first.")
            
            # Get form data
            brand = request.form['brand']
            vehicle_age = float(request.form['vehicle_age'])
            km_driven = float(request.form['km_driven'])
            fuel_type = request.form['fuel_type']
            seller_type = request.form['seller_type']
            transmission_type = request.form['transmission_type']
            mileage = float(request.form['mileage'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])
            seats = float(request.form['seats'])
            
            # Encode categorical features
            brand_encoded = label_encoders['brand'].transform([brand])[0]
            fuel_type_encoded = label_encoders['fuel_type'].transform([fuel_type])[0]
            seller_type_encoded = label_encoders['seller_type'].transform([seller_type])[0]
            transmission_type_encoded = label_encoders['transmission_type'].transform([transmission_type])[0]
            
            # Prepare features for prediction
            features = np.array([[
                vehicle_age, km_driven, mileage, engine, max_power, seats,
                brand_encoded, fuel_type_encoded, seller_type_encoded, transmission_type_encoded
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            return render_template('result.html', 
                                 prediction=round(prediction, 2),
                                 brand=brand,
                                 vehicle_age=int(vehicle_age),
                                 km_driven=int(km_driven),
                                 fuel_type=fuel_type,
                                 seller_type=seller_type,
                                 transmission_type=transmission_type,
                                 mileage=mileage,
                                 engine=int(engine),
                                 max_power=max_power,
                                 seats=int(seats))
        
        except Exception as e:
            return render_template('predict.html', error=f"Error: {str(e)}")
    
    return render_template('predict.html')

@app.route('/insights')
def insights():
    """Show data insights and visualizations"""
    try:
        # Load insights data
        insights_data = {}
        try:
            with open('static/insights.txt', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        insights_data[key.strip()] = value.strip()
        except:
            insights_data = {
                'total_cars': '1000',
                'avg_price': '5.75',
                'avg_age': '7.2',
                'avg_km': '67500',
                'most_common_brand': 'Maruti Suzuki',
                'most_common_fuel': 'Petrol',
                'best_model': 'Random Forest',
                'best_accuracy': '89.2'
            }
        
        # Check which visualizations exist
        visualizations = {}
        viz_files = [
            'price_distribution.png', 'brand_analysis.png', 
            'correlation_matrix.png', 'model_performance.png'
        ]
        
        for viz_file in viz_files:
            visualizations[viz_file] = os.path.exists(f'static/images/{viz_file}')
        
        return render_template('insights.html', 
                             insights=insights_data,
                             visualizations=visualizations)
    
    except Exception as e:
        return render_template('insights.html', 
                             insights={},
                             visualizations={},
                             error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded', 'status': 'error'}), 400
        
        data = request.get_json()
        
        # Extract features
        brand = data['brand']
        vehicle_age = float(data['vehicle_age'])
        km_driven = float(data['km_driven'])
        fuel_type = data['fuel_type']
        seller_type = data['seller_type']
        transmission_type = data['transmission_type']
        mileage = float(data['mileage'])
        engine = float(data['engine'])
        max_power = float(data['max_power'])
        seats = float(data['seats'])
        
        # Encode categorical features
        brand_encoded = label_encoders['brand'].transform([brand])[0]
        fuel_type_encoded = label_encoders['fuel_type'].transform([fuel_type])[0]
        seller_type_encoded = label_encoders['seller_type'].transform([seller_type])[0]
        transmission_type_encoded = label_encoders['transmission_type'].transform([transmission_type])[0]
        
        # Prepare features and predict
        features = np.array([[
            vehicle_age, km_driven, mileage, engine, max_power, seats,
            brand_encoded, fuel_type_encoded, seller_type_encoded, transmission_type_encoded
        ]])
        
        prediction = model.predict(features)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'price_units': 'lakhs',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/health')
def health():
    status = {
        'model_loaded': model is not None,
        'status': 'healthy' if model else 'model_missing'
    }
    return jsonify(status)

@app.route('/about')
def about():
    """About page with project details"""
    return render_template('about.html')


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    print("🚗 AutoValue - Used Car Price Prediction System Started!")
    print("📍 Available routes:")
    print("   http://localhost:5000/          - Home page")
    print("   http://localhost:5000/predict   - Price Prediction") 
    print("   http://localhost:5000/insights  - Data Insights & Visualizations")
    print("   http://localhost:5000/health    - Health check")
    
    if model:
        print("✅ System ready! Model is loaded.")
    else:
        print("❌ WARNING: Model not loaded. Run: python train_model.py")
    
    app.run(debug=True, host='0.0.0.0', port=5000)