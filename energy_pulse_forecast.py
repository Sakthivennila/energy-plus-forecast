import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def generate_dummy_data(n_samples=1000):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate dummy data
    # Occupancy: 0 to 100 people
    occupancy = np.random.randint(0, 100, n_samples)
    
    # Temperature: 10 to 35 degrees Celsius
    temperature = np.random.uniform(10, 35, n_samples)
    
    # Electricity consumption (kWh): base load + occupancy effect + temperature effect (e.g. AC) + noise
    # Assuming baseline of 50 kWh, 0.5 kWh per person, and non-linear temp effect (more AC when hot, heating when cold)
    # For simplicity in linear regression, let's use a linear relation: baseline + 2*occupancy + 5*(temp-20) + noise
    electricity = 50 + (1.5 * occupancy) + (3.0 * temperature) + np.random.normal(0, 10, n_samples)
    
    df = pd.DataFrame({
        'Occupancy': occupancy,
        'Temperature': temperature,
        'Electricity_Consumption': electricity
    })
    
    return df

def train_and_evaluate_model(df):
    # Features (X) and Target (y)
    X = df[['Occupancy', 'Temperature']]
    y = df['Electricity_Consumption']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Coefficients:")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Occupancy Coef: {model.coef_[0]:.2f}")
    print(f"Temperature Coef: {model.coef_[1]:.2f}")
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    return model

def predict_consumption(model, occupancy, temperature):
    input_data = pd.DataFrame([[occupancy, temperature]], columns=['Occupancy', 'Temperature'])
    prediction = model.predict(input_data)[0]
    print(f"\nPrediction for Occupancy={occupancy}, Temperature={temperature}C : {prediction:.2f} kWh")
    return prediction

if __name__ == "__main__":
    print("Generating dummy data for Energy Pulse Forecast...")
    df = generate_dummy_data(1000)
    
    print("Training Linear Regression Model...")
    model = train_and_evaluate_model(df)
    
    # Test predictions
    predict_consumption(model, occupancy=50, temperature=25)
    predict_consumption(model, occupancy=10, temperature=15)
