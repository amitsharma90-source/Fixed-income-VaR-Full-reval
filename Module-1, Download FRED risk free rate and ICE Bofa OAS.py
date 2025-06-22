# -*- coding: utf-8 -*-
"""
Created on Sun May 11 10:16:22 2025

@author: amits
"""

import pandas_datareader.data as web
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Define FRED series
yield_series = {
    "1Y_Treasury": "GS1",
    "2Y_Treasury": "GS2",
    "3Y_Treasury": "GS3",
    "5Y_Treasury": "GS5"
}

oas_series = {
    "AAA_OAS": "BAMLC0A1CAAA",
    "AA_OAS": "BAMLC0A2CAA",
    "A_OAS": "BAMLC0A3CA",
    "BBB_OAS": "BAMLC0A4CBBB",
    "BB_OAS": "BAMLH0A1HYBB"
}

# Fetch ~2 years of data to ensure 500 trading days
end = datetime.today()
start = end - timedelta(days=1000)

# Download and combine
data = pd.DataFrame()
for label, series in {**yield_series, **oas_series}.items():
    df = web.DataReader(series, "fred", start, end)
    df.columns = [label]
    data = pd.concat([data, df], axis=1)

# Drop missing values and keep last 500 rows
# data = data.dropna().tail(500)
data["4Y_Treasury"] = (data["3Y_Treasury"] + data["5Y_Treasury"]) / 2
data = data.ffill().dropna().tail(500)

data.to_csv("C:\\Users\\amits\\Desktop\\Market Risk\\Fixed income values at Risk\\data\\Yield curves and OAS\\risk_free_and_oas_500d.csv")
print(data.tail())

# data2 = data(['1Y_Treasury',	'2Y_Treasury',	'3Y_Treasury',	'5Y_Treasury'])

# # Plot
# data2.plot(figsize=(12, 6), title="1Y to 5Y Treasury Yields (Past 2 Years)")
# plt.ylabel("Yield (%)")
# plt.xlabel("Date")
# plt.grid(True)
# plt.show()
