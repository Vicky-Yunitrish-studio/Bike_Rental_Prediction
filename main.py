import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

class BikeRentalPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Bike Rental Prediction System")
        self.window.geometry("400x500")
        
        # 創建輸入欄位
        self.create_input_fields()
        
        # 創建預測按鈕
        self.predict_button = ttk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)
        
        # 顯示結果的標籤
        self.result_label = ttk.Label(self.window, text="")
        self.result_label.pack(pady=10)
        
        # 新增自動獲取天氣按鈕
        self.weather_button = ttk.Button(self.window, text="Get Current Weather", command=self.get_weather)
        self.weather_button.pack(pady=5)
        
    def create_input_fields(self):
        # 輸入欄位
        ttk.Label(self.window, text="Temperature (°C):").pack()
        self.temp_entry = ttk.Entry(self.window)
        self.temp_entry.pack()
        
        ttk.Label(self.window, text="Humidity (%):").pack()
        self.humidity_entry = ttk.Entry(self.window)
        self.humidity_entry.pack()
        
        ttk.Label(self.window, text="Wind Speed (m/s):").pack()
        self.windspeed_entry = ttk.Entry(self.window)
        self.windspeed_entry.pack()
        
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
            self.result_label.config(text="An error occurred, please try again later")
    
    def predict(self):
        try:
            # 獲取輸入值
            temp = float(self.temp_entry.get())
            humidity = float(self.humidity_entry.get())
            windspeed = float(self.windspeed_entry.get())
            
            # 這裡應該載入訓練好的模型
            # model = joblib.load('bike_rental_model.pkl')
            
            # 暫時使用簡單計算方式（實際應使用訓練好的模型）
            predicted_count = int((temp * 10 + (100 - humidity) * 0.5 - windspeed * 5))
            predicted_count = max(0, predicted_count)  # 確保不會出現負數
            
            self.result_label.config(text=f"Predicted rental count: {predicted_count} bikes")
            
        except ValueError:
            self.result_label.config(text="Please enter valid numbers")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = BikeRentalPredictor()
    app.run()
