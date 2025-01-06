import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

class BikeRentalModel:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        # Convert 'datetime' to datetime type and extract features
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['year'] = data['datetime'].dt.year
        data['month'] = data['datetime'].dt.month
        data['day'] = data['datetime'].dt.day
        data['hour'] = data['datetime'].dt.hour
        data['weekday'] = data['datetime'].dt.weekday
        
        # Drop unnecessary columns
        features_to_drop = ['datetime', 'casual', 'registered']
        for col in features_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
                
        return data
        
    def train(self, train_data):
        # Preprocess the data
        processed_data = self.preprocess_data(train_data.copy())
        
        # Split features and target
        X = processed_data.drop('count', axis=1)
        y = processed_data['count']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        
    def predict(self, input_data):
        # Preprocess the input data
        processed_data = self.preprocess_data(input_data.copy())
        
        # Scale the features
        X_scaled = self.scaler.transform(processed_data)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        return predictions
        
    def save_model(self, model_path='model.joblib'):
        joblib.dump((self.model, self.scaler), model_path)
        
    def load_model(self, model_path='model.joblib'):
        self.model, self.scaler = joblib.load(model_path)
