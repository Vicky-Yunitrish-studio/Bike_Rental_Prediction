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
import math
from datetime import datetime

class BikeRentalPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Bike Rental Prediction System")
        self.window.geometry("1200x800")
        self.window.configure(bg='#f0f0f0')  # 設置背景色
        
        # Create main containers
        self.left_frame = ttk.Frame(self.window)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.right_frame = ttk.Frame(self.window)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Initialize model and encoder
        self.model = None
        self.ohe = None
        self.load_and_train_model()
        
        # Create UI sections
        self.create_clock_section()
        self.create_weather_section()
        self.create_input_section()
        self.create_control_section()

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

    def create_clock_section(self):
        clock_frame = ttk.LabelFrame(self.left_frame, text="Time Settings", padding=10)
        clock_frame.pack(fill=tk.X, pady=10)
        
        self.clock_canvas = tk.Canvas(clock_frame, width=200, height=200, bg='white')
        self.clock_canvas.pack(pady=10)
        
        self.hour_var = tk.IntVar(value=0)
        self.hour_slider = ttk.Scale(
            clock_frame,
            from_=0,
            to=23,
            orient='horizontal',
            variable=self.hour_var,
            command=self.update_clock
        )
        self.hour_slider.pack(fill=tk.X, pady=5)
        
        self.hour_label = ttk.Label(clock_frame, text="Hour: 0")
        self.hour_label.pack()
        
        self.draw_clock()

    def create_weather_section(self):
        weather_frame = ttk.LabelFrame(self.left_frame, text="Weather Control", padding=10)
        weather_frame.pack(fill=tk.X, pady=10)
        
        # Auto update settings
        self.auto_frame = ttk.Frame(weather_frame)
        self.auto_frame.pack(fill=tk.X, pady=5)
        
        self.auto_update_time = tk.BooleanVar(value=False)
        self.auto_update_weather = tk.BooleanVar(value=False)
        
        update_controls = ttk.Frame(self.auto_frame)
        update_controls.pack(fill=tk.X, pady=5)
        
        self.time_check = ttk.Checkbutton(
            update_controls,
            text="Auto update time",
            variable=self.auto_update_time,
            command=self.toggle_auto_update
        )
        self.time_check.pack(side=tk.LEFT, padx=5)
        
        self.weather_check = ttk.Checkbutton(
            update_controls,
            text="Auto update weather",
            variable=self.auto_update_weather,
            command=self.toggle_auto_update
        )
        self.weather_check.pack(side=tk.LEFT, padx=5)
        
        interval_frame = ttk.Frame(weather_frame)
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="Update interval:").pack(side=tk.LEFT)
        self.update_interval = ttk.Entry(interval_frame, width=5)
        self.update_interval.insert(0, "5")
        self.update_interval.pack(side=tk.LEFT, padx=5)
        ttk.Label(interval_frame, text="minutes").pack(side=tk.LEFT)
        
        button_frame = ttk.Frame(weather_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.time_button = ttk.Button(button_frame, text="Get Current Time", command=self.get_current_time)
        self.time_button.pack(side=tk.LEFT, padx=5)
        
        self.weather_button = ttk.Button(button_frame, text="Get Current Weather", command=self.get_weather)
        self.weather_button.pack(side=tk.LEFT, padx=5)

    def create_input_section(self):
        input_frame = ttk.LabelFrame(self.right_frame, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for inputs
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Numerical inputs
        input_fields = [
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
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label_text, width=20).pack(side=tk.LEFT)
            entry = ttk.Entry(frame)
            entry.insert(0, placeholder)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            setattr(self, entry_name, entry)

        # Categorical inputs
        categories_frame = ttk.Frame(scrollable_frame)
        categories_frame.pack(fill=tk.X, pady=10)
        
        # Season
        season_frame = ttk.Frame(categories_frame)
        season_frame.pack(fill=tk.X, pady=2)
        ttk.Label(season_frame, text="Season:", width=20).pack(side=tk.LEFT)
        self.season_var = tk.StringVar()
        self.season_combo = ttk.Combobox(season_frame, textvariable=self.season_var, 
                                        values=['Spring', 'Summer', 'Autumn', 'Winter'])
        self.season_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Holiday
        holiday_frame = ttk.Frame(categories_frame)
        holiday_frame.pack(fill=tk.X, pady=2)
        ttk.Label(holiday_frame, text="Holiday Status:", width=20).pack(side=tk.LEFT)
        self.holiday_var = tk.StringVar()
        self.holiday_combo = ttk.Combobox(holiday_frame, textvariable=self.holiday_var, 
                                         values=['Holiday', 'No Holiday'])
        self.holiday_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Functioning Day
        functioning_frame = ttk.Frame(categories_frame)
        functioning_frame.pack(fill=tk.X, pady=2)
        ttk.Label(functioning_frame, text="Functioning Day:", width=20).pack(side=tk.LEFT)
        self.functioning_var = tk.StringVar()
        self.functioning_combo = ttk.Combobox(functioning_frame, textvariable=self.functioning_var, 
                                            values=['Yes', 'No'])
        self.functioning_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_control_section(self):
        control_frame = ttk.LabelFrame(self.right_frame, text="Prediction Control", padding=10)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.predict_button = ttk.Button(control_frame, text="Predict", command=self.predict)
        self.predict_button.pack(fill=tk.X, pady=5)
        
        self.result_label = ttk.Label(control_frame, text="", wraplength=400)
        self.result_label.pack(pady=10)

    def draw_clock(self):
        # Clear canvas
        self.clock_canvas.delete("all")
        
        # Clock face parameters
        center_x = 100
        center_y = 100
        radius = 80
        
        # Draw clock circle
        self.clock_canvas.create_oval(
            center_x-radius, center_y-radius,
            center_x+radius, center_y+radius,
            width=2
        )
        
        # Draw hour markers
        for hour in range(24):
            angle = math.radians(hour * 360 / 24 - 90)
            outer_x = center_x + radius * math.cos(angle)
            outer_y = center_y + radius * math.sin(angle)
            inner_x = center_x + (radius-10) * math.cos(angle)
            inner_y = center_y + (radius-10) * math.sin(angle)
            
            # Draw hour marker line
            self.clock_canvas.create_line(inner_x, inner_y, outer_x, outer_y)
            
            # Draw hour numbers
            text_x = center_x + (radius-25) * math.cos(angle)
            text_y = center_y + (radius-25) * math.sin(angle)
            self.clock_canvas.create_text(text_x, text_y, text=str(hour))
        
        # Draw hour hand
        current_hour = self.hour_var.get()
        angle = math.radians(current_hour * 360 / 24 - 90)
        hand_length = radius - 20
        hand_x = center_x + hand_length * math.cos(angle)
        hand_y = center_y + hand_length * math.sin(angle)
        self.clock_canvas.create_line(
            center_x, center_y, hand_x, hand_y,
            width=3, fill='red', arrow=tk.LAST
        )

    def update_clock(self, *args):
        self.hour_label.config(text=f"Hour: {self.hour_var.get()}")
        self.draw_clock()

    def get_current_time(self):
        current_hour = datetime.now().hour
        self.hour_var.set(current_hour)
        self.update_clock()
        self.result_label.config(text=f"Time updated to current hour: {current_hour}")

    def toggle_auto_update(self):
        """Handle both time and weather auto-updates"""
        if hasattr(self, '_after_id'):
            self.window.after_cancel(self._after_id)
            
        if self.auto_update_time.get() or self.auto_update_weather.get():
            self.update_continuously()

    def update_continuously(self):
        """Update both time and weather based on checkbox states"""
        try:
            if self.auto_update_time.get():
                self.get_current_time()
            
            if self.auto_update_weather.get():
                self.get_weather()
            
            # Calculate interval in milliseconds
            interval = int(self.update_interval.get()) * 60 * 1000  # Convert minutes to milliseconds
            
            # Schedule next update if either auto-update is enabled
            if self.auto_update_time.get() or self.auto_update_weather.get():
                self._after_id = self.window.after(interval, self.update_continuously)
                
        except ValueError:
            self.result_label.config(text="Please enter a valid update interval")
            self.auto_update_time.set(False)
            self.auto_update_weather.set(False)

    def get_weather(self):
        try:
            API_KEY = "CWA-15F1DACE-AFC5-444F-B7D7-5CFBC6218CEF"
            url = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001?Authorization=' + API_KEY
            
            response = requests.get(url)
            data_json = response.json()
            weather_data = data_json['records']
            
            # Find data for 彰化縣
            for station in weather_data['Station']:
                if station['GeoInfo']['CountyName'] == "彰化縣":
                    # Get weather elements
                    wind_speed = station['WeatherElement']['WindSpeed']
                    temperature = station['WeatherElement']['AirTemperature']
                    humidity = station['WeatherElement']['RelativeHumidity']
                    
                    # Update input fields
                    self.temp_entry.delete(0, tk.END)
                    self.temp_entry.insert(0, temperature)
                    
                    self.humidity_entry.delete(0, tk.END)
                    self.humidity_entry.insert(0, float(humidity))
                    
                    self.windspeed_entry.delete(0, tk.END)
                    self.windspeed_entry.insert(0, wind_speed)
                    
                    self.result_label.config(text="Weather data updated successfully")
                    break
            else:
                self.result_label.config(text="No data found for 彰化縣")
            
            # Don't automatically update time unless auto time update is enabled
            if self.auto_update_time.get():
                self.get_current_time()
                
        except requests.exceptions.ConnectionError:
            self.result_label.config(text="Connection failed: Please check your internet connection")
        except requests.exceptions.Timeout:
            self.result_label.config(text="Request timeout: Server response took too long") 
        except Exception as e:
            print(e)
            self.result_label.config(text="Error occurred, please try again later")
    
    def predict(self):
        try:
            # Get all input values (modified to use hour_var instead of hour_entry)
            input_data = {
                'Hour': float(self.hour_var.get()),
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
