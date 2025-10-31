import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

plt.style.use('default')
sns.set_palette("husl")

def generate_sample_data():
    """Generate sample dataset with realistic patterns"""
    np.random.seed(42)
    n_samples = 1000
    
    brands = ['Maruti Suzuki', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Volkswagen', 'Tata', 'Mahindra']
    fuel_types = ['Petrol', 'Diesel', 'CNG']
    seller_types = ['Dealer', 'Individual']
    transmission_types = ['Manual', 'Automatic']
    
    data = []
    
    for _ in range(n_samples):
        brand = np.random.choice(brands, p=[0.25, 0.20, 0.15, 0.12, 0.08, 0.07, 0.08, 0.05])
        vehicle_age = np.random.randint(1, 15)
        km_driven = np.random.randint(1000, 150000)
        fuel_type = np.random.choice(fuel_types, p=[0.6, 0.35, 0.05])
        seller_type = np.random.choice(seller_types, p=[0.6, 0.4])
        transmission_type = np.random.choice(transmission_types, p=[0.75, 0.25])
        mileage = np.random.uniform(12, 25)
        engine = np.random.choice([998, 1197, 1498, 1598, 1998, 2198])
        max_power = np.random.uniform(70, 200)
        seats = np.random.choice([5, 7, 8], p=[0.7, 0.2, 0.1])
        
        # Realistic price calculation with brand premium
        brand_premium = {
            'Maruti Suzuki': 0.9, 'Hyundai': 0.95, 'Honda': 1.1, 
            'Toyota': 1.2, 'Ford': 1.0, 'Volkswagen': 1.05, 
            'Tata': 0.85, 'Mahindra': 0.95
        }
        
        base_price = (engine * 0.015 + max_power * 0.8 + mileage * 0.5) * brand_premium[brand]
        
        # Adjust for age and km
        selling_price = (base_price * 
                        (1 - vehicle_age * 0.07) * 
                        (1 - km_driven * 0.0000008) + 
                        np.random.normal(0, 0.3))
        
        # Ensure price is reasonable and apply fuel type adjustments
        if fuel_type == 'Diesel':
            selling_price *= 1.1
        elif fuel_type == 'CNG':
            selling_price *= 0.9
            
        if transmission_type == 'Automatic':
            selling_price *= 1.15
            
        selling_price = max(selling_price, 1.0)
        
        data.append({
            'brand': brand,
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'fuel_type': fuel_type,
            'seller_type': seller_type,
            'transmission_type': transmission_type,
            'mileage': round(mileage, 1),
            'engine': engine,
            'max_power': round(max_power, 1),
            'seats': seats,
            'selling_price': round(selling_price, 2)
        })
    
    df = pd.DataFrame(data)
    return df

def load_and_preprocess_data():
    """Load and preprocess the car data"""
    try:
        # Try to load existing data
        df = pd.read_csv('data/car_data.csv')
        print("✅ Loaded existing dataset")
    except:
        # Generate sample data if file doesn't exist
        print("📊 Generating sample dataset...")
        df = generate_sample_data()
        df.to_csv('data/car_data.csv', index=False)
        print("✅ Sample dataset created and saved")
    
    # Data cleaning
    print(f"📈 Initial data shape: {df.shape}")
    
    # Handle missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"📊 Processed data shape: {df.shape}")
    return df

def create_visualizations(df, results, X_test, y_test, model):
    """Create comprehensive visualizations"""
    os.makedirs('static/images', exist_ok=True)
    
    # 1. Price Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['selling_price'], kde=True, color='#f97316')
    plt.title('Distribution of Car Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Price (Lakhs ₹)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['selling_price'], color='#fdba74')
    plt.title('Price Distribution Box Plot', fontsize=14, fontweight='bold')
    plt.ylabel('Price (Lakhs ₹)')
    plt.tight_layout()
    plt.savefig('static/images/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Brand-wise Analysis
    plt.figure(figsize=(14, 8))
    brand_stats = df.groupby('brand').agg({
        'selling_price': ['mean', 'count']
    }).round(2)
    brand_stats.columns = ['avg_price', 'count']
    brand_stats = brand_stats.sort_values('avg_price', ascending=False)
    
    plt.subplot(2, 2, 1)
    sns.barplot(x=brand_stats.index, y=brand_stats['avg_price'], palette='Oranges_r')
    plt.title('Average Price by Brand', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Average Price (Lakhs ₹)')
    
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='vehicle_age', y='selling_price', hue='brand', 
                   palette='Set2', alpha=0.7)
    plt.title('Price vs Age by Brand', fontsize=12, fontweight='bold')
    plt.xlabel('Vehicle Age (Years)')
    plt.ylabel('Price (Lakhs ₹)')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='fuel_type', y='selling_price', palette='YlOrBr')
    plt.title('Price Distribution by Fuel Type', fontsize=12, fontweight='bold')
    plt.xlabel('Fuel Type')
    plt.ylabel('Price (Lakhs ₹)')
    
    plt.subplot(2, 2, 4)
    transmission_price = df.groupby('transmission_type')['selling_price'].mean()
    plt.pie(transmission_price.values, labels=transmission_price.index, 
            autopct='%1.1f%%', colors=['#fdba74', '#f97316'])
    plt.title('Average Price by Transmission', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/images/brand_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Correlations
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='Oranges', 
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/images/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance
    plt.figure(figsize=(12, 6))
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Vehicle Age', 'KM Driven', 'Mileage', 'Engine', 'Max Power', 'Seats',
                        'Brand', 'Fuel Type', 'Seller Type', 'Transmission']
        feature_imp = pd.Series(model.feature_importances_, index=feature_names)
        feature_imp = feature_imp.sort_values(ascending=True)
        
        plt.subplot(1, 2, 1)
        feature_imp.plot(kind='barh', color='#f97316')
        plt.title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
        plt.xlabel('Importance Score')
        
    # 5. Actual vs Predicted
    y_pred = model.predict(X_test)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.6, color='#f97316')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (Lakhs ₹)')
    plt.ylabel('Predicted Price (Lakhs ₹)')
    plt.title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/images/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Create interactive plotly charts
    create_interactive_charts(df)

def create_interactive_charts(df):
    """Create interactive Plotly charts"""
    # Interactive scatter plot
    fig = px.scatter(df, x='vehicle_age', y='selling_price', color='brand',
                     size='engine', hover_data=['km_driven', 'fuel_type'],
                     title='Car Price vs Age by Brand',
                     labels={'vehicle_age': 'Vehicle Age (Years)', 
                            'selling_price': 'Price (Lakhs ₹)'})
    fig.write_html('static/images/interactive_scatter.html')
    
    # Brand comparison
    brand_avg = df.groupby('brand').agg({
        'selling_price': 'mean',
        'vehicle_age': 'mean',
        'km_driven': 'mean'
    }).reset_index()
    
    fig = px.bar(brand_avg, x='brand', y='selling_price',
                 title='Average Car Prices by Brand',
                 labels={'selling_price': 'Average Price (Lakhs ₹)', 'brand': 'Brand'})
    fig.write_html('static/images/brand_prices.html')
    
    # 3D scatter plot
    fig = px.scatter_3d(df.head(200), x='vehicle_age', y='km_driven', z='selling_price',
                        color='brand', size='engine',
                        title='3D View: Age vs KM vs Price')
    fig.write_html('static/images/3d_scatter.html')

def train_models(df):
    """Train and evaluate multiple models"""
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['brand', 'fuel_type', 'seller_type', 'transmission_type']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Features and target
    features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
                'brand_encoded', 'fuel_type_encoded', 'seller_type_encoded', 'transmission_type_encoded']
    
    X = df[features]
    y = df['selling_price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }
        
        print(f"\n{name} Results:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    return results, X_train, X_test, y_train, y_test, features, label_encoders

def save_best_model(results):
    """Save the best performing model"""
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
    best_model = results[best_model_name]['model']
    
    joblib.dump(best_model, 'models/random_forest_model.pkl')
    print(f"\n✅ Best model saved: {best_model_name}")
    
    return best_model_name, best_model

def generate_insights_report(df, results):
    """Generate insights report"""
    insights = {
        'total_cars': len(df),
        'avg_price': round(df['selling_price'].mean(), 2),
        'avg_age': round(df['vehicle_age'].mean(), 1),
        'avg_km': round(df['km_driven'].mean()),
        'most_common_brand': df['brand'].mode()[0],
        'most_common_fuel': df['fuel_type'].mode()[0],
        'best_model': max(results.keys(), key=lambda x: results[x]['r2_score']),
        'best_accuracy': round(max(results[x]['r2_score'] for x in results) * 100, 1)
    }
    
    # Save insights to file
    with open('static/insights.txt', 'w') as f:
        for key, value in insights.items():
            f.write(f"{key}: {value}\n")
    
    return insights

if __name__ == "__main__":
    print("🚗 Starting Car Price Prediction Model Training...")
    
    print("\n📊 Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("\n🤖 Training models...")
    results, X_train, X_test, y_train, y_test, features, label_encoders = train_models(df)
    
    print("\n💾 Saving best model...")
    best_model_name, best_model = save_best_model(results)
    
    print("\n📈 Generating visualizations...")
    create_visualizations(df, results, X_test, y_test, best_model)
    
    print("\n🔍 Generating insights report...")
    insights = generate_insights_report(df, results)
    
    print("\n✅ Training completed successfully!")
    print(f"🎯 Best model: {best_model_name}")
    print(f"📊 Dataset insights:")
    print(f"   - Total cars analyzed: {insights['total_cars']}")
    print(f"   - Average price: ₹{insights['avg_price']} lakhs")
    print(f"   - Average vehicle age: {insights['avg_age']} years")
    print(f"   - Model accuracy: {insights['best_accuracy']}%")
    
    print(f"\n📁 Generated visualizations:")
    print("   - static/images/price_distribution.png")
    print("   - static/images/brand_analysis.png") 
    print("   - static/images/correlation_matrix.png")
    print("   - static/images/model_performance.png")
    print("   - static/images/interactive_scatter.html")