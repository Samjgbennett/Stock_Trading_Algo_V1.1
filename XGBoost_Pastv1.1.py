import pandas as pd
import numpy as np
import scipy.optimize as optimize

#Get data

import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries


api_key = ''
symbol = 'MSFT'

ts = TimeSeries(api_key, output_format='pandas')
data, meta = ts.get_intraday(symbol, interval='1min', outputsize='full')


#I have imported this data just for a train journey, so i can use it offline
#data = pd.read_csv("InitialTestData.csv") 

data.sort_values(by='date', ascending = True, inplace = True)

data = data.rename({'1. open': 'open', '2. high': 'high','3. low':'low', '4. close':'close', '5. volume':'volume'}, axis=1)



"""
import yfinance as yf
import pandas as pd

data = yf.download("ES", start="2023-01-27", end="2023-02-3", interval='1m')
data = data.iloc[:,:]

data = data.rename({'Open': 'open', 'High': 'high','Low':'low', 'Close':'close', 'Volume':'volume','Datetime':'date'}, axis=1)
data.index.names = ['date']
"""


data = data.dropna(axis=0)

print(data)

import copy
datax = copy.copy(data)


data['open'] = data['open'].shift(-1)
data['close'] = data['close'].shift(-1)
data['high'] = data['high'].shift(-1)
data['low'] = data['low'].shift(-1)
data['volume'] = data['volume'].shift(-1)



data = data.dropna(axis=0)


data.sort_values(by='date', ascending = False, inplace = True)
datax.sort_values(by='date', ascending = False, inplace = True)


data['lag_1'] = data['close'].shift(1)
data['lag_2'] = data['close'].shift(2)

datax['lag_1'] = datax['close'].shift(1)
datax['lag_2'] = datax['close'].shift(2)


#save this one
data2 = data

#data.to_csv("test_data_lagged.csv")

merged = pd.concat([data, data[['lag_1','lag_2']]], axis=1)

#data = data.dropna(axis=0)

# Split data into train and test sets
test_frac = 0.1

train_start = int(data.shape[0] * test_frac)
test_data = data.iloc[:train_start]
train_data = data.iloc[train_start:]
test_datax = datax.iloc[:train_start]
train_datax = datax.iloc[:train_data.shape[0]]

# Define target and features
X_train = train_data[['lag_1','lag_2']]
y_train = train_datax['close']
X_test = test_data[['lag_1','lag_2']]
y_test = test_datax['close']    

#Standard Scaler


X_train = X_train.apply(lambda x: (x - x.mean()) / (x.std()))
X_test = X_test.apply(lambda x: (x - x.mean()) / (x.std()))

import xgboost as xgb
# Define classifier
classifier = xgb.XGBRegressor(random_state=30)

# Train classifier
classifier.fit(X_train, y_train)

# Test classifier
y_pred = classifier.predict(X_test)
y_pred = y_pred.flatten()

combined = pd.DataFrame(dict(actual=y_test, XGBoost=y_pred))

merged = pd.concat([combined, X_test], axis=1)
merged = merged[['actual', 'XGBoost']]
merged = pd.concat([merged, data2], axis=1)
merged = merged.dropna(subset=['XGBoost'],axis=0)

merged["Diff"] = merged['close'] - merged["XGBoost"]
merged["Diff%"] = (merged["Diff"]/merged["close"])*100


accuracy_diff = merged["Diff"].mean()
accuracy_percentage = merged["Diff%"].mean()


data = merged


from scipy.signal import savgol_filter

data['smooth'] = savgol_filter(data['XGBoost'],window_length=3, polyorder=2)

#data['smooth'] = savgol_filter(data['smooth'],window_length=5, polyorder=3)
import numpy as np
import numpy as np
from pykalman import KalmanFilter


# Define the observation matrix, which is taken as an identity matrix in this example
observation_matrix = np.identity(1)

# Estimate the initial state mean and initial state covariance based on historical data
initial_state_mean = np.mean(data['smooth'])
initial_state_covariance = np.cov(data['smooth'])

# Define the transition matrix, which assumes a linear relationship between the state at time t and t-1
transition_matrix = np.array([[1]])

# Define the process noise covariance and observation noise covariance, which are assumed to be diagonal matrices with small values in this example
process_noise_covariance = np.array([[1e-5]])
observation_noise_covariance = np.array([[1e-3]])

# Create a KalmanFilter object
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    #process_noise_covariance=process_noise_covariance
)

#Fit the Kalman filter to the financial data
filtered_state_means, filtered_state_covariances = kf.filter(data['smooth'])
data['Kalman'] = pd.DataFrame(filtered_state_means, index=data['smooth'].index)


# Create a figure and an axis
fig, ax = plt.subplots()

# Plot the first dataset
ax.plot(data['close'] , label='actual')

ax2 = ax.twinx()

#ax2.plot(data['ma3XGBoost'],'k',label='ma3XGBoost')
ax2.plot(data['smooth'],'c',label='smooth')

ax3 = ax.twinx()
ax3.plot(data['Kalman'],'m', label='Kalman filter')
#ax2.plot(data['fourier'],'m',label='fourier')

plt.title('Stock Movement')
ax.legend()
ax2.legend(loc='upper left')
ax3.legend(loc='lower left')
plt.show()



data['bb_upper'] = data['smooth'].rolling(window=20).mean() + 2*data['smooth'].rolling(window=20).std()
data['bb_middle'] = data['smooth'].rolling(window=20).mean()
data['bb_lower'] = data['smooth'].rolling(window=20).mean() - 2*data['smooth'].rolling(window=20).std()



#Backtesting

#data['Gradient'] = data['XGBoost'] - data['XGBoost'].shift(1)
data['Gradient'] = data['Kalman'] - data['Kalman'].shift(1)
data['2Gradient'] = data['Kalman'] - data['Kalman'].shift(2)
data['SecondDeriv'] = data['Gradient'] - data['Gradient'].shift(1)


data['hour'] = data.index.hour
data['Trading'] = np.where(np.logical_and(data['hour'] >= 10,data['hour'] < 16),1,0)
#buy when gradient > 0.2, sell if gradient < 0 and buy = True


#Long Trades
#data['trades_L'] = np.where(np.logical_and(0.13<data['Gradient'],data['Gradient']<0.18),1,0)

"""Current best strategies:

#1 - ES
data['trades_Buy_L'] = np.where(np.logical_and(data['SecondDeriv']<-0.03,data['Gradient']>0.04),1,0)
data['trades_Sell_L'] = np.where(np.logical_and(data['SecondDeriv']>-0.01,data['Gradient']<0.03),-1,0)

"""


data['trades_Buy_L'] = np.where(data['Trading']==1, np.where(np.logical_and(data['SecondDeriv']<-0.03, data['Gradient']>0.02),1,0), 0)
data['trades_Sell_L'] = np.where(data['Trading']==1, np.where(np.logical_and(data['SecondDeriv']>-0.01, data['Gradient']<0),-1,0), 0)

#Short Trades

data['trades_Buy_S'] = np.where(data['Trading']==1, np.where(np.logical_and(data['SecondDeriv']>0.03, data['Gradient']<-0.04),1,0), 0)
data['trades_Sell_S'] = np.where(data['Trading']==1, np.where(np.logical_and(data['SecondDeriv']<0.01, data['Gradient']>-0.03),-1,0), 0)



#Bet where we have RF is greater than previous point


data['Holding_L'] = np.where(data['trades_Buy_L'] == 1, 1, np.where(data['trades_Sell_L'] == -1, 0, np.nan))
data['Holding_L'].fillna(method='ffill', inplace=True)
data['prev_holding_L'] = data['Holding_L'].shift(1)


data['Holding_S'] = np.where(data['trades_Buy_S'] == 1, 1, np.where(data['trades_Sell_S'] == -1, 0, np.nan))
data['Holding_S'].fillna(method='ffill', inplace=True)
data['prev_holding_S'] = data['Holding_S'].shift(1)



#Calculating where trades are made
data['change_L'] = np.where((data['Holding_L'] == 1) & (data['prev_holding_L'] == 0), 1, np.where((data['Holding_L'] == 0) & (data['prev_holding_L'] == 1), -1, 0))
data['change_S'] = np.where((data['Holding_S'] == 1) & (data['prev_holding_S'] == 0), 1, np.where((data['Holding_S'] == 0) & (data['prev_holding_S'] == 1), -1, 0))


# Generate trades, we trade if over a sufficient number

hold_mask_L = data['change_L'] == 1
hold_mask_S = data['change_S'] == 1


# Create a boolean array for when the holding is 0
not_hold_mask_L = data['change_L'] == -1
not_hold_mask_S = data['change_S'] == -1




""" Plotting """


# Plot the Close values in green when holding is 1
plt.plot(data[hold_mask_L].index, data[hold_mask_L]['close'], 'g.', label='Bought_L',markersize=10)

plt.plot(data[hold_mask_S].index, data[hold_mask_S]['close'], 'k.', label='Bought_S',markersize=10)


# Plot the Close values in red when holding is 0
plt.plot(data[not_hold_mask_L].index, data[not_hold_mask_L]['close'], 'r.', label='Sold_L',markersize=10)

plt.plot(data[not_hold_mask_S].index, data[not_hold_mask_S]['close'], 'm.', label='Sold_S',markersize=10)

plt.plot(data['close'], label= 'Close', dashes=[3, 1])


# Add a legend to the plot
plt.legend()




#Calculation of profit
data['profit_L'] = data['Holding_L'] * (data['close'] - data['close'].shift(1))
profit_L = data['profit_L'].sum()
   
data['profit_S'] = data['Holding_S'] * (data['close'].shift(1)-data['close'])
profit_S = data['profit_S'].sum()
data['cumprofit'] = data['profit_S'].cumsum() + data['profit_L'].cumsum()


Long_profit = data['profit_L'].sum()*50
Short_profit = data['profit_S'].sum()*50

print("Long Profit multiplier",data['profit_L'].sum())
print("Short Profit multiplier",data['profit_S'].sum())
print("Long Profit",Long_profit)
print("Short Profit",Short_profit)



#data.to_csv("UnFunctioned_V1.1.csv")
Number_of_trades_L = (data['trades_Buy_L'].sum())
Number_of_trades_S = (data['trades_Buy_S'].sum())
print("Number of trades (Long)",Number_of_trades_L)
print("Number of trades (Short)",Number_of_trades_S)

Trading_costs_micro = (Number_of_trades_L+Number_of_trades_S)*0.25

Total_profit = (Long_profit + Short_profit) - Trading_costs_micro
print("Profit",Total_profit)


# Plot cumulative returns

plt.figure(figsize=(10,5))
plt.plot(data['cumprofit'])
plt.title('Cumulative XGBoost Profit')
plt.show()


final_cols = ['close','XGBoost','Holding_L','Holding_S','cumprofit']
data = data[final_cols]

print(data)

#data.to_csv("UnFunctioned_V1.10.csv")





