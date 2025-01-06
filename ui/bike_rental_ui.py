import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime

class BikeRentalUI:
    def __init__(self, predict_callback):
        self.predict_callback = predict_callback
        self.window = tk.Tk()
        self.window.title("Bike Rental Prediction")
        self.window.geometry("600x800")
        self.create_widgets()

    def create_widgets(self):
        # Create input fields
        input_frame = ttk.LabelFrame(self.window, text="Input Parameters", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)

        # DateTime
        ttk.Label(input_frame, text="Date and Time:").grid(row=0, column=0, sticky="w")
        self.datetime_entry = ttk.Entry(input_frame)
        self.datetime_entry.grid(row=0, column=1, padx=5, pady=5)
        self.datetime_entry.insert(0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Season
        ttk.Label(input_frame, text="Season (1-4):").grid(row=1, column=0, sticky="w")
        self.season_entry = ttk.Entry(input_frame)
        self.season_entry.grid(row=1, column=1, padx=5, pady=5)

        # Weather
        ttk.Label(input_frame, text="Weather (1-4):").grid(row=2, column=0, sticky="w")
        self.weather_entry = ttk.Entry(input_frame)
        self.weather_entry.grid(row=2, column=1, padx=5, pady=5)

        # Temperature
        ttk.Label(input_frame, text="Temperature (Celsius):").grid(row=3, column=0, sticky="w")
        self.temp_entry = ttk.Entry(input_frame)
        self.temp_entry.grid(row=3, column=1, padx=5, pady=5)

        # Humidity
        ttk.Label(input_frame, text="Humidity (%):").grid(row=4, column=0, sticky="w")
        self.humidity_entry = ttk.Entry(input_frame)
        self.humidity_entry.grid(row=4, column=1, padx=5, pady=5)

        # Windspeed
        ttk.Label(input_frame, text="Windspeed:").grid(row=5, column=0, sticky="w")
        self.windspeed_entry = ttk.Entry(input_frame)
        self.windspeed_entry.grid(row=5, column=1, padx=5, pady=5)

        # Predict Button
        self.predict_button = ttk.Button(input_frame, text="Predict", command=self.on_predict)
        self.predict_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Result
        result_frame = ttk.LabelFrame(self.window, text="Prediction Result", padding="10")
        result_frame.pack(fill="x", padx=10, pady=5)

        self.result_label = ttk.Label(result_frame, text="")
        self.result_label.pack(pady=5)

    def on_predict(self):
        try:
            input_data = pd.DataFrame({
                'datetime': [self.datetime_entry.get()],
                'season': [float(self.season_entry.get())],
                'weather': [float(self.weather_entry.get())],
                'temp': [float(self.temp_entry.get())],
                'humidity': [float(self.humidity_entry.get())],
                'windspeed': [float(self.windspeed_entry.get())]
            })
            
            prediction = self.predict_callback(input_data)
            self.result_label.config(text=f"Predicted number of rentals: {int(prediction[0])}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.window.mainloop()
