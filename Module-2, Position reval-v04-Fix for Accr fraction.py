# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:09:32 2025

@author: amits
"""
# Bond VaR Engine: Full Revaluation with Time-Weighted Historical Simulation (Spot + OAS)

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import requests
import io
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2, norm

# --- Configuration ---
lookback_days = 250
test_days = 249
confidence_level = 0.99

# --- Load Data ---
RF_OAS = pd.read_csv("C:\\Users\\amits\\Desktop\\Market Risk\\Fixed income values at Risk\\data\\Yield curves and OAS\\risk_free_and_oas_500d.csv", index_col=0, parse_dates=True)
RF_OAS = RF_OAS.ffill()
positions = pd.read_excel("C:\\Users\\amits\\Desktop\\Market Risk\\Fixed income values at Risk\\data\\Positions\\holdings.xlsx")  # Contains bond info including type, notional, maturity, coupon, tenor

RF_OAS['PV'] = None

# --- Bond Valuation Using Date-Based Year Loop ---
def price_spot_oas(position, rf_oas_row, rate = None):
    today = date.today()
    total_pv = 0
    if rate is None:
        coupon_rate_decimal = position['Coupon'] * 0.01
    else:
        coupon_rate_decimal = position['Coupon'] * 0.01 + rate*0.01
    
    discount_year = 1
    total_accr_frac = 0
    for i, year in enumerate(range(today.year, position['Maturity'].year + 1)):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        accr_frac = 0                                
        if i == 0:
            if position['Maturity'] <= year_end:
                accr_frac = (position['Maturity'] - today).days / 360
                Notional = position['Notional exposure$']
            elif position['Maturity'] > year_end:
                accr_frac = (year_end - pd.Timestamp(today)).days / 360
                Notional = 0
        else:
            # Year_increment+=1
            if position['Maturity'] <= year_end:
                accr_frac = (position['Maturity'] - year_start).days / 360
                Notional = position['Notional exposure$']
            else:
                accr_frac = 1
                Notional = 0
        
        total_accr_frac+=accr_frac        
        accr_coupon = position['Notional exposure$'] * coupon_rate_decimal * accr_frac
        # tenor = discount_year + "Y_Treasury"
        tenor = str(discount_year) + "Y_Treasury"
        pv = (Notional + accr_coupon) / (1 + (rf_oas_row[tenor] + rf_oas_row[position['OAS curve']])/100)**(total_accr_frac)
        total_pv += pv
        discount_year+=1

    return total_pv

# --- PnL Simulation Loop (daily) would go here (same as before) ---
# --- Simulate Daily P&L ---

for date_index in RF_OAS.index:
    Total_pv = 0
    pv  = 0
    for _, row in positions.iterrows():
        pv  = 0
        if row['Position Type'] == "Bond":
            pv = price_spot_oas(row,RF_OAS.loc[date_index])            
        elif row['Position Type'] == "Repo":
            pv = price_spot_oas(row,RF_OAS.loc[date_index], rate = RF_OAS.loc[date_index]['SOFR'] + 0.07)
            
        elif row['Position Type'] == "Bond TRS":
            pv = price_spot_oas(row,RF_OAS.loc[date_index], rate = -0.75)
            
        Total_pv = Total_pv+pv
        RF_OAS.loc[date_index, 'PV'] = Total_pv
       
pnl_series = RF_OAS["PV"].diff().dropna()

# --- Configuration ---
lookback_days = 250
test_days = 249
results = []
for i in range(lookback_days, len(pnl_series) - 1):
    hist = pnl_series.iloc[i - lookback_days:i]
    sorted_idx = np.argsort(hist)
    sorted_returns = hist.values[sorted_idx]
    # sorted_weights = time_weights[sorted_idx]
    # cum_weights = np.cumsum(sorted_weights)
    index_99 = int(0.01 * len(sorted_returns))
    VaR_99 = sorted_returns[index_99]
    # VaR_idx = np.searchsorted(cum_weights, 1 - confidence_level)
    # VaR = -sorted_returns[VaR_idx]
    realized_pnl = pnl_series.iloc[i + 1]
    exception = realized_pnl < VaR_99
    tail_losses = sorted_returns[sorted_returns < VaR_99]
    # tail_weights = sorted_weights[sorted_returns < -VaR]
    CVaR = np.mean(tail_losses) #/ #np.sum(tail_weights)
    results.append({
        'Date': pnl_series.index[i + 1],
        'VaR_99': VaR_99,
        'CVaR_99': CVaR,
        'Realized_PnL': realized_pnl,
        'Exception': exception
    })

backtest_df = pd.DataFrame(results)

n_obs = len(backtest_df)
n_exceptions = backtest_df['Exception'].sum()
expected_exceptions = (1 - confidence_level) * n_obs
fail_ratio = n_exceptions / n_obs
p_hat = 1 - confidence_level
LR_uc = -2 * (np.log((1 - p_hat)**(n_obs - n_exceptions) * (p_hat)**n_exceptions) -
              np.log((1 - fail_ratio)**(n_obs - n_exceptions) * (fail_ratio)**n_exceptions))
p_value = 1 - chi2.cdf(LR_uc, df=1)

exceptions = backtest_df['Exception'].astype(int)
transitions = list(zip(exceptions.shift(1, fill_value=0), exceptions))
n00 = transitions.count((0, 0))
n01 = transitions.count((0, 1))
n10 = transitions.count((1, 0))
n11 = transitions.count((1, 1))
total_0 = n00 + n01
total_1 = n10 + n11
p01 = n01 / total_0 if total_0 > 0 else 0.0001
p11 = n11 / total_1 if total_1 > 0 else 0.0001
p_total = (n01 + n11) / (total_0 + total_1)
LR_indep = -2 * (np.log((1 - p_total)**(n00 + n10) * (p_total)**(n01 + n11)) -
                 (np.log((1 - p01)**n00 * (p01)**n01 * (1 - p11)**n10 * (p11)**n11)))
p_value_indep = 1 - chi2.cdf(LR_indep, df=1)

# Output Results
print("Backtesting Results:\n")
print(f"Total Observations: {n_obs}")
print(f"Number of Exceptions: {n_exceptions}")
print(f"Expected Exceptions (1%): {expected_exceptions:.0f}")
print(f"\nKupiec Unconditional Coverage Test:")
print(f"Kupiec Test LR: {LR_uc:.3f}")
print(f"Kupiec Test p-value: {p_value:.3f}")
if p_value > 0.05:
    print("\u2705 Model Passed Kupiec Test (Good calibration)")
else:
    print("\u274C Model Failed Kupiec Test (Poor calibration)")
print(f"\nChristoffersen Independence Test:")
print(f"Christoffersen Test LR: {LR_indep:.3f}")
print(f"Christoffersen Test p-value: {p_value_indep:.3f}")
if p_value_indep > 0.05:
    print("\u2705 Exceptions are independent (Good)")
else:
    print("\u274C Exceptions are not independent")

# Exception summary statistics
exception_df = backtest_df[backtest_df['Exception'] == True].copy()
if not exception_df.empty:
    max_loss = exception_df['Realized_PnL'].min()
    avg_loss = exception_df['Realized_PnL'].mean()
    print(f"\nException Summary:")
    print(f"Worst Exception (Max Loss): ${max_loss:,.2f}")
    print(f"Average Exception Loss: ${avg_loss:,.2f}")


# Plotting
plt.figure(figsize=(14,6))
plt.plot(backtest_df['Date'], backtest_df['Realized_PnL'], label='Realized PnL')
plt.plot(backtest_df['Date'], backtest_df['VaR_99'], label='VaR 99%', linestyle='--')
plt.plot(backtest_df['Date'], backtest_df['CVaR_99'], label='CVaR 99%', linestyle=':')
plt.scatter(backtest_df[backtest_df['Exception']]['Date'],
            backtest_df[backtest_df['Exception']]['Realized_PnL'],
            color='red', label='Exceptions')
plt.title('Combined VaR Backtest - Stocks + Short Options')
plt.xlabel('Date')
plt.ylabel('P&L ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Export results
backtest_df.to_excel("C:\\Users\\amits\\Desktop\\Market Risk\\Combined, Stock, Cash, Short Option, GARCH, Time weighted\\1 year look back, 1 year VaR, tight time weighing, GARCH for short option\\combined_var_backtest_output.xlsx", index=False)
exception_df.to_excel("C:\\Users\\amits\\Desktop\\Market Risk\\Combined, Stock, Cash, Short Option, GARCH, Time weighted\\1 year look back, 1 year VaR, tight time weighing, GARCH for short option\\exceptions_only.xlsx", index=False)
print("Backtest results exported to combined_var_backtest_output.xlsx")
print("Exception days exported to exceptions_only.xlsx")



# Assuming `pnl_series` is your PnL series
plt.figure(figsize=(12, 6))

# KDE + Histogram
sns.histplot(pnl_series, kde=True, stat='density', bins=50, color='skyblue', label='PnL Distribution')

# Overlay normal distribution
mean = pnl_series.mean()
std = pnl_series.std()
x_vals = np.linspace(pnl_series.min(), pnl_series.max(), 1000)
plt.plot(x_vals, stats.norm.pdf(x_vals, mean, std), 'r--', label='Normal Distribution')

plt.title('Combined PnL Distribution vs Normal Distribution')
plt.xlabel('PnL ($)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Skewness: {pnl_series.skew():.3f}")
print(f"Kurtosis: {pnl_series.kurtosis():.3f}")

