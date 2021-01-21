import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# Import data
# data = pd.read_csv('data/bydaym2.csv',index_col='date', parse_dates=True)
# data.head()

code = pd.read_csv('data/code2U.csv')


for i in range(0, 199):
    data = pd.read_csv('data/bydaym2.csv', index_col='date', parse_dates=True)
    # print("'"+code['stock'][i]+"'")

    data = data.query('stock=="'+code['stock'][i]+'"')
    # data = data.query('stock=="AK홀딩스"')
    print(data.head())

    # query
    data_s = data.query('date>=20080403 & date<=20180403')
    print(data.head())

    # Drop date variable
    # data_s = data_s.drop(['stock','date'], 1)
    data_s = data_s.drop(['stock', 'open', 'high', 'low', 'diff', 'vol', 'f', 'i'], 1)

    data_s.plot()


    plt.title(code['stock'][i])
    plt.savefig('D:/빅데이터/pic/'+code['stock'][i]+'1'+'.png')



    from fbprophet import Prophet

    df = data_s.reset_index().rename(columns={'date': 'ds', 'close': 'y'})
    df['y'] = np.log(df['y'])
    model = Prophet(daily_seasonality=True)

    model.fit(df)
    future = model.make_future_dataframe(periods=365)  # forecasting for 1 year from now.

    forecast = model.predict(future)

    figure = model.plot(forecast)

    plt.title(code['stock'][i])
    plt.savefig('D:/빅데이터/pic/' + code['stock'][i] + '2' + '.png')


    # Last 3 years of Actuals (orange) vs Forecast (blue – listed as yhat)
    three_years = forecast.set_index('ds').join(data_s)
    three_years = three_years[['close', 'yhat', 'yhat_upper', 'yhat_lower']].dropna().tail(800)
    three_years['yhat'] = np.exp(three_years.yhat)
    three_years['yhat_upper'] = np.exp(three_years.yhat_upper)
    three_years['yhat_lower'] = np.exp(three_years.yhat_lower)
    three_years[['close', 'yhat']].plot()

    plt.title(code['stock'][i])
    plt.savefig('D:/빅데이터/pic/' + code['stock'][i] + '3' + '.png')


    #########

    three_years_AE = (three_years.yhat - three_years.close)
    print(three_years_AE.describe())

    # close to 1
    from sklearn.metrics import r2_score

    r2_score(three_years.close, three_years.yhat)

    # MSE, closer to zero is better
    from sklearn.metrics import *

    mean_squared_error(three_years.close, three_years.yhat)
    mean_absolute_error(three_years.close, three_years.yhat)

    ##
    fig, ax1 = plt.subplots()
    ax1.plot(three_years.close)
    ax1.plot(three_years.yhat)
    ax1.plot(three_years.yhat_upper, color='black', linestyle=':', alpha=0.5)
    ax1.plot(three_years.yhat_lower, color='black', linestyle=':', alpha=0.5)

    ax1.set_title('"'+code['stock'][i]+'" Actual(Orange) vs Forecasted Upper & Lower Confidence (Black)')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    plt.savefig('D:/빅데이터/pic/' + code['stock'][i] + '4' + '.png')


    full_df = forecast.set_index('ds').join(data_s)
    full_df['yhat'] = np.exp(full_df['yhat'])

    #####
    fig, ax1 = plt.subplots()
    ax1.plot(full_df.close)
    ax1.plot(full_df.yhat, color='black', linestyle=':')
    ax1.fill_between(full_df.index, np.exp(full_df['yhat_upper']), np.exp(full_df['yhat_lower']), alpha=0.5,
                     color='darkgray')
    ax1.set_title('"'+code['stock'][i]+'" Actual(Orange) vs Forecasted (Black) with Confidence Bands')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')

    L = ax1.legend()  # get the legend
    L.get_texts()[0].set_text('close Actual')  # change the legend text for 1st plot
    L.get_texts()[1].set_text('close Forecasted')  # change the legend text for 2nd plot

    plt.savefig('D:/빅데이터/pic/' + code['stock'][i] + '5' + '.png')

    ###########
    full_df[['close', 'yhat']].plot()
    # plt.show()
    plt.savefig('D:/빅데이터/pic/' + code['stock'][i] + '6' + '.png')
