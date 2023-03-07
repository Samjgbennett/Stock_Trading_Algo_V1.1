from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest
from oanda_candles import Pair, Gran, CandleClient
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.instruments as instruments


import pandas as pd
import numpy as np
import scipy.optimize as optimize
import matplotlib as plt

def signal_generator(data):

    data = data.rename({'Open': 'open', 'High': 'high','Low':'low', 'Close':'close', 'Volume':'volume'}, axis=1)
    data.index.names = ['date']

    data = data.dropna(axis=0)

    import copy
    datax = copy.copy(data)


    data['open'] = data['open'].shift(-1)
    data['close'] = data['close'].shift(-1)
    data['high'] = data['high'].shift(-1)
    data['low'] = data['low'].shift(-1)



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


    data = merged

    from scipy.signal import savgol_filter

    
    if len(data['XGBoost']) <= 3:
        window_length = len(data['XGBoost'])
        polyorder = window_length - 1
    else:
        window_length = 3
        polyorder = 2

    data['smooth'] = savgol_filter(data['XGBoost'], window_length=window_length, polyorder=polyorder)

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


    #Backtesting


    #data['Gradient'] = data['XGBoost'] - data['XGBoost'].shift(1)
    data['Gradient'] = data['Kalman'] - data['Kalman'].shift(1)
    data['2Gradient'] = data['Kalman'] - data['Kalman'].shift(2)
    data['SecondDeriv'] = data['Gradient'] - data['Gradient'].shift(1)


    data['date'] = pd.to_datetime(data.index)
    data['hour'] = data['date'].dt.hour
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

    # Sell Pattern
    if data['trades_Sell_L'].iloc[-1] == -1:
        return 1

    # Buy Pattern
    elif data['trades_Buy_L'].iloc[-1] == 1:
        return 2
    
    # No clear pattern
    else:
        return 0



accountID = ''
access_token=''


def get_candles(n):
    client = API(access_token=access_token, environment="practice")
    params = {
        "granularity": "M1", # 1 minute candles , S5 for 5 seconds
        "count": n, # retrieve n candles
    }
    r = instruments.InstrumentsCandles(instrument="SPX500_USD", params=params)
    client.request(r)
    # Access the candles data
    candles = r.response["candles"]
    return candles

candles = get_candles(3893)

def trading_job():
    candles = get_candles(3893)
    dfstream = pd.DataFrame(columns=['Open','Close','High','Low'])
    dfstream = dfstream.rename({'Open': 'open', 'High': 'high','Low':'low', 'Close':'close', 'Volume':'volume'}, axis=1)
    dfstream.index.names = ['date']

    i=0

    """
    for candle in candles:
        dfstream.loc[i, ['open']] = float(str(candle.bid.o))
        dfstream.loc[i, ['close']] = float(str(candle.bid.c))
        dfstream.loc[i, ['high']] = float(str(candle.bid.h))
        dfstream.loc[i, ['low']] = float(str(candle.bid.l))
        i=i+1
    """

   
    for candle in candles:
        dfstream.loc[i, 'open'] = float(str(candle["mid"]["o"]))
        dfstream.loc[i, 'close'] = float(str(candle["mid"]["c"]))
        dfstream.loc[i, 'high'] = float(str(candle["mid"]["h"]))
        dfstream.loc[i, 'low'] = float(str(candle["mid"]["l"]))
        i = i + 1

    dfstream['open'] = dfstream['open'].astype(float)
    dfstream['close'] = dfstream['close'].astype(float)
    dfstream['high'] = dfstream['high'].astype(float)
    dfstream['low'] = dfstream['low'].astype(float)

    signal = signal_generator(dfstream.iloc[:-1,:]) #selceting all rows apart from the last one
    


    # EXECUTING ORDERS
    client = API(access_token)

    #SL = Stop Loss
    #TP = Take Profit
    SLTPRatio = 2.

    #previous candles range
    previous_candleR = abs(dfstream['high'].iloc[-2]-dfstream['low'].iloc[-2])
    
    #float(str(candle.bid.o)) is opening price, i've changed all the bid to mid

    SLBuy = float(str(candle["mid"]["o"]))-previous_candleR
    SLSell = float(str(candle["mid"]["o"]))+previous_candleR

    TPBuy = float(str(candle["mid"]["o"]))+previous_candleR*SLTPRatio
    TPSell = float(str(candle["mid"]["o"]))-previous_candleR*SLTPRatio
    
    print(dfstream.iloc[:-1,:])
    print(TPBuy, "  ", SLBuy, "  ", TPSell, "  ", SLSell)
    signal = 6
    #Sell
    if signal == 1:
        #mo is market order
        mo = MarketOrderRequest(instrument="SPX500_USD", units=-1, takeProfitOnFill=TakeProfitDetails(price=TPSell).data, stopLossOnFill=StopLossDetails(price=SLSell).data)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print(rv) #just to see that order has passed
    #Buy
    elif signal == 2:
        mo = MarketOrderRequest(instrument="SPX500_USD", units=1, takeProfitOnFill=TakeProfitDetails(price=TPBuy).data, stopLossOnFill=StopLossDetails(price=SLBuy).data)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print(rv)

#trading_job()

scheduler = BlockingScheduler()
scheduler.add_job(trading_job, 'cron', day_of_week='mon-fri', hour='10-17',start_date='2022-02-13 10:00:00', timezone='Europe/London')#minute='1,16,31,46'
scheduler.start()

