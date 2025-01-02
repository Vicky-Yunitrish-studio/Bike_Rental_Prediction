import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

class BikeRentalPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Bike Rental Prediction System")
        self.window.geometry("700x800")
        
        # Initialize model and encoder
        self.model = None
        self.ohe = None
        self.load_and_train_model()
        
        # Create input fields
        self.create_input_fields()
        
        # Create buttons
        self.predict_button = ttk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)
        
        self.weather_button = ttk.Button(self.window, text="Get Current Weather", command=self.get_weather)
        self.weather_button.pack(pady=5)
        
        self.result_label = ttk.Label(self.window, text="")
        self.result_label.pack(pady=10)

    def load_and_train_model(self):
        try:
            # Load data
            data_path = './SeoulBikeData.csv'
            data = pd.read_csv(data_path)
            data = data.dropna()
            data['Rented Bike Count'] = data['Rented Bike Count'].astype(float)

            # One-hot encode categorical variables
            self.ohe = OneHotEncoder(sparse=False, drop='first')
            categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
            categorical_encoded = self.ohe.fit_transform(data[categorical_features])
            categorical_encoded_df = pd.DataFrame(
                categorical_encoded, 
                columns=self.ohe.get_feature_names_out(categorical_features)
            )

            # Prepare features
            numerical_features = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 
                                'Visibility (10m)', 'Dew point temperature(C)', 
                                'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']
            X = pd.concat([data[numerical_features], categorical_encoded_df], axis=1)
            y = data['Rented Bike Count']

            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
        except Exception as e:
            print(f"Error loading/training model: {e}")

    def create_input_fields(self):
        # Extended input fields
        input_fields = [
            ("Hour (0-23):", "hour_entry", "0-23"),
            ("Temperature (°C):", "temp_entry", ""),
            ("Humidity (%):", "humidity_entry", ""),
            ("Wind Speed (m/s):", "windspeed_entry", ""),
            ("Visibility (10m):", "visibility_entry", "0-2000"),
            ("Dew Point Temperature (°C):", "dewpoint_entry", ""),
            ("Solar Radiation (MJ/m2):", "radiation_entry", "0-10"),
            ("Rainfall (mm):", "rainfall_entry", "0-100"),
            ("Snowfall (cm):", "snowfall_entry", "0-100")
        ]

        for label_text, entry_name, placeholder in input_fields:
            ttk.Label(self.window, text=label_text).pack()
            entry = ttk.Entry(self.window)
            entry.insert(0, placeholder)
            entry.pack()
            setattr(self, entry_name, entry)

        # Dropdown for categorical variables
        ttk.Label(self.window, text="Season:").pack()
        self.season_var = tk.StringVar()
        self.season_combo = ttk.Combobox(self.window, textvariable=self.season_var, 
                                        values=['Spring', 'Summer', 'Autumn', 'Winter'])
        self.season_combo.pack()

        ttk.Label(self.window, text="Holiday Status:").pack()
        self.holiday_var = tk.StringVar()
        self.holiday_combo = ttk.Combobox(self.window, textvariable=self.holiday_var, 
                                         values=['Holiday', 'No Holiday'])
        self.holiday_combo.pack()

        ttk.Label(self.window, text="Functioning Day:").pack()
        self.functioning_var = tk.StringVar()
        self.functioning_combo = ttk.Combobox(self.window, textvariable=self.functioning_var, 
                                            values=['Yes', 'No'])
        self.functioning_combo.pack()

    def get_weather(self):
        try:
            API_KEY = "CWA-15F1DACE-AFC5-444F-B7D7-5CFBC6218CEF"
            location = "彰化縣"
            url = "https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0003-001"
        
            response = requests.get(url)
            params = {
                "Authorization": API_KEY,
                "locationName": location
            }
            response = requests.get(url, params=params, timeout=10, verify=True)
            if response.status_code == 200:
                data = response.json()
                weather_data = data['records']['location'][0]['weatherElement']
                
                # 解析資料
                temp = next(item['elementValue'] for item in weather_data if item['elementName'] == 'TEMP')
                humidity = next(item['elementValue'] for item in weather_data if item['elementName'] == 'HUMD')
                windspeed = next(item['elementValue'] for item in weather_data if item['elementName'] == 'WDSD')
                
                # 更新輸入欄位
                self.temp_entry.delete(0, tk.END)
                self.temp_entry.insert(0, temp)
                
                self.humidity_entry.delete(0, tk.END)
                self.humidity_entry.insert(0, float(humidity) * 100)  # 轉換為百分比
                
                self.windspeed_entry.delete(0, tk.END)
                self.windspeed_entry.insert(0, windspeed)
                
            else:
                self.result_label.config(text="Unable to fetch weather data")
                
        except requests.exceptions.ConnectionError:
            self.result_label.config(text="Connection failed: Please check your internet connection")
        except requests.exceptions.Timeout:
            self.result_label.config(text="Request timeout: Server response took too long") 
        except Exception as e:
            print(e)
            self.result_label.config(text="Error occurred, please try again later")
    
    def predict(self):
        try:
            # Get all input values
            input_data = {
                'Hour': float(self.hour_entry.get()),
                'Temperature(C)': float(self.temp_entry.get()),
                'Humidity(%)': float(self.humidity_entry.get()),
                'Wind speed (m/s)': float(self.windspeed_entry.get()),
                'Visibility (10m)': float(self.visibility_entry.get()),
                'Dew point temperature(C)': float(self.dewpoint_entry.get()),
                'Solar Radiation (MJ/m2)': float(self.radiation_entry.get()),
                'Rainfall(mm)': float(self.rainfall_entry.get()),
                'Snowfall (cm)': float(self.snowfall_entry.get()),
                'Seasons': self.season_var.get(),
                'Holiday': self.holiday_var.get(),
                'Functioning Day': self.functioning_var.get()
            }

            # Prepare input for prediction
            input_df = pd.DataFrame([input_data])
            categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
            categorical_input = self.ohe.transform(input_df[categorical_features])
            categorical_input_df = pd.DataFrame(
                categorical_input, 
                columns=self.ohe.get_feature_names_out(categorical_features)
            )

            numerical_features = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 
                                'Visibility (10m)', 'Dew point temperature(C)', 
                                'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']
            input_combined = pd.concat([input_df[numerical_features], categorical_input_df], axis=1)

            # Make prediction
            prediction = self.model.predict(input_combined)
            predicted_count = max(0, int(prediction[0]))
            
            self.result_label.config(text=f"Predicted Rental Count: {predicted_count} bikes")
            
        except ValueError:
            self.result_label.config(text="Please enter valid numerical values")
        except Exception as e:
            self.result_label.config(text=f"Prediction error: {str(e)}")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = BikeRentalPredictor()
    app.run()
