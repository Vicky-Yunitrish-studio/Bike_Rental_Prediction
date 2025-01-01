# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the data
data_path = '/home/yuchi/Bike_Rental_Prediction/SeoulBikeData.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
#print(data.head())

# Step 2: Preprocess data
# Ensure the target and feature columns are appropriately selected and handle missing values if any
data = data.dropna()
data['Rented Bike Count'] = data['Rented Bike Count'].astype(float)

# One-hot encode categorical variables
ohe = OneHotEncoder(sparse_output=False, drop='first')
categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
categorical_encoded = ohe.fit_transform(data[categorical_features])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out(categorical_features))

# Combine numerical and categorical features
numerical_features = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                      'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']
X = pd.concat([data[numerical_features], categorical_encoded_df], axis=1)
y = data['Rented Bike Count']

# Step 3: Analyze correlation
correlation_matrix = data[numerical_features + ['Rented Bike Count']].corr()
#print("Correlation matrix:\n", correlation_matrix)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 6: Function for prediction
def predict_rented_bike_count(hour, temperature, humidity, wind_speed, visibility, dew_point_temp,
                              solar_radiation, rainfall, snowfall, season, holiday, functioning_day):
    # Prepare input data
    input_data = {
        'Hour': hour,
        'Temperature(C)': temperature,
        'Humidity(%)': humidity,
        'Wind speed (m/s)': wind_speed,
        'Visibility (10m)': visibility,
        'Dew point temperature(C)': dew_point_temp,
        'Solar Radiation (MJ/m2)': solar_radiation,
        'Rainfall(mm)': rainfall,
        'Snowfall (cm)': snowfall,
        'Seasons': season,
        'Holiday': holiday,
        'Functioning Day': functioning_day
    }
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    categorical_input = ohe.transform(input_df[categorical_features])
    categorical_input_df = pd.DataFrame(categorical_input, columns=ohe.get_feature_names_out(categorical_features))

    # Combine numerical and categorical features
    input_combined = pd.concat([input_df[numerical_features], categorical_input_df], axis=1)

    # Predict rented bike count
    prediction = model.predict(input_combined)
    return prediction[0]

'''
# Example usage
hour = 4  # Example Hour
temp = -6  # Example Temperature in Celsius
humidity = 36  # Example Humidity in %
wind_speed = 2.3  # Example Wind Speed in m/s
visibility = 2000  # Example Visibility in 10m
dew_point_temp = -18.6  # Example Dew Point Temperature in Celsius
solar_radiation = 0  # Example Solar Radiation in MJ/m2
rainfall = 0  # Example Rainfall in mm
snowfall = 0  # Example Snowfall in cm
season = 'Winter'  # Example Season
holiday = 'No Holiday'  # Example Holiday
functioning_day = 'Yes'  # Example Functioning Day

predicted_count = predict_rented_bike_count(hour, temp, humidity, wind_speed, visibility, dew_point_temp,
                                            solar_radiation, rainfall, snowfall, season, holiday, functioning_day)
print(f"Predicted Rented Bike Count: {predicted_count}")
'''