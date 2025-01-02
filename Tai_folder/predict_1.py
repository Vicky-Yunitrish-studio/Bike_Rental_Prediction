# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 讀取數據
df = pd.read_csv('./../SeoulBikeData.csv')

# 數據預處理
# 將連續型變量離散化
def discretize_column(df, column, bins):
    labels = [f'{i}' for i in range(len(bins)-1)]
    df[f'{column}_cat'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

# 對主要特徵進行離散化
df = discretize_column(df, 'Temperature(C)', bins=[-20, 0, 10, 20, 30, 40])
df = discretize_column(df, 'Humidity(%)', bins=[0, 20, 40, 60, 80, 100])
df = discretize_column(df, 'Wind speed (m/s)', bins=[0, 2, 4, 6, 8, 10])
df = discretize_column(df, 'Rented Bike Count', bins=[0, 500, 1000, 1500, 2000, 3000])
#print(df)

# 創建 One-Hot 編碼
columns_to_encode = ['Temperature(C)_cat', 'Humidity(%)_cat', 'Wind speed (m/s)_cat', 'Rented Bike Count_cat']
df_encoded = pd.get_dummies(df[columns_to_encode])
#print(df_encoded)

# 使用 Apriori 算法找出頻繁項集
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
#print(frequent_itemsets.head(5))

# 生成關聯規則
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))

def predict_bike_count(temp, humidity, wind_speed):
    # 將輸入值轉換為對應的類別
    temp_cat = pd.cut([temp], bins=[-20, 0, 10, 20, 30, 40],
                      labels=['Temperature(C)_cat_0', 'Temperature(C)_cat_1', 'Temperature(C)_cat_2', 'Temperature(C)_cat_3', 'Temperature(C)_cat_4'])[0]
    humidity_cat = pd.cut([humidity], bins=[0, 20, 40, 60, 80, 100],
                          labels=['Humidity(%)_cat_0', 'Humidity(%)_cat_1', 'Humidity(%)_cat_2', 'Humidity(%)_cat_3', 'Humidity(%)_cat_4'])[0]
    wind_cat = pd.cut([wind_speed], bins=[0, 2, 4, 6, 8, 10],
                      labels=['Wind speed (m/s)_cat_0', 'Wind speed (m/s)_cat_1', 'Wind speed (m/s)_cat_2', 'Wind speed (m/s)_cat_3', 'Wind speed (m/s)_cat_4'])[0]

    print(f"溫度類別: {temp_cat}" + f"\n濕度類別: {humidity_cat}" + f"\n風速類別: {wind_cat}")

    # 構建條件集合
    conditions = {temp_cat, humidity_cat, wind_cat}
    #print(conditions)

    # 查找相關規則
    relevant_rules = rules[rules['antecedents'].apply(lambda x: not conditions.isdisjoint(x))]

    # 過濾出與租借數量相關的規則
    relevant_rules = relevant_rules[relevant_rules['consequents'].apply(lambda x: any('Rented Bike Count' in str(i) for i in x))]

    if not relevant_rules.empty:
        # 根據置信度最高的規則進行預測
        best_rule = relevant_rules.sort_values('confidence', ascending=False).iloc[0]
        predicted_range = list(best_rule['consequents'])[0]
        print(predicted_range)
        if predicted_range == "Rented Bike Count_cat_0" :
          #return f"預測的租借數量範圍: 0-500"
            return f"0-500"
        else :
            return f"test!"
    else:
        return "無法找到足夠的關聯規則進行預測"


'''
# 測試預測函數
print(predict_bike_count(-5.0, 35, 2.5))

# 顯示一些重要的關聯規則
print("\n重要關聯規則:")
important_rules = rules[rules['lift'] > 1.0].sort_values('confidence', ascending=False)
print(important_rules[['antecedents', 'consequents', 'confidence', 'lift']].head())
'''