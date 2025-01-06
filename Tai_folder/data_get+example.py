import requests

url = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001?Authorization=CWA-15F1DACE-AFC5-444F-B7D7-5CFBC6218CEF'
data = requests.get(url)   # 取得 JSON 檔案的內容為文字
data_json = data.json()    # 轉換成 JSON 格式
weather_data = data_json['records'] # 取得天氣資料
for i in weather_data['Station']:
    if (i['GeoInfo']['CountyName']) == "彰化縣":
        print(i['StationName'])
        print(i['WeatherElement']['WindSpeed'])
        print(i['WeatherElement']['AirTemperature'])
        print(i['WeatherElement']['RelativeHumidity'])
        print("--------------------")
        #break