import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def prepare_features(df):
    """
    Extracts time-based features from the Timestamp column.
    """
    # Ensure Timestamp is a datetime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Feature engineering
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['IsWeekday'] = np.where(df['DayOfWeek'] < 5, 1, 0)
    
    return df

def train_forecaster():
    print("Loading building energy data...")
    df = pd.read_csv("building_energy_data.csv")
    
    # Extract time features
    df = prepare_features(df)
    
    # Define our features (X) and target variable (y)
    feature_cols = ['Temperature', 'Hour', 'DayOfWeek', 'Month', 'IsWeekday']
    X = df[feature_cols]
    y = df['Total_Load']
    
    # Split the dataset into 80% Training and 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    # Random Forest is highly accurate for tabular regression and doesn't require feature scaling
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Training Complete!")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kWh")
    print(f"R² Score (Accuracy): {r2 * 100:.2f}%")
    
    # Save the trained model to disk so we can use it in our Streamlit dashboard
    joblib.dump(model, "energy_forecast_model.pkl")
    print("Model saved successfully as 'energy_forecast_model.pkl'!")

if __name__ == "__main__":
    train_forecaster()