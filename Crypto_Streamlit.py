import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import requests
# import tensorflow
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import streamlit as st
# import tensorflow as tf
import warnings
from keras.models import model_from_json
from keras.models import load_model
import altair as alt


# warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


# st.line_chart(data)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=5):
    fig, ax = plt.subplots(1, figsize=(14, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_xlabel('Crypto', fontsize=14)
    ax.set_ylabel('Currency', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show()

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def extract_window_data(df, window_len=10, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test

    
def display_data(data):
    st.write("""
        # Model Data and Information
        """
         )
    train, test = train_test_split(data)
    
    st.write("""
        # Training Data
        """
         )
    st.line_chart(train['close'])
    
    st.write("""
        # Testing Data
        """
         )
    st.line_chart(test['close'])
    
def forecast_data(data,days):
    
    st.write("""
        # Forecasting values

        """
         )
    
    
def main():
    output = st.empty()
    output1 = st.empty()
    output.write("""
        # Crypto Forecast

        """
         )
    
    crypto=st.sidebar.selectbox('Select Choice of Crypto using their abbreviation',('BTC','ETH','DASH','DOGE','LTC','USDT'))
    fiat=st.sidebar.selectbox('Select Choice of currency using their abbreviation',('INR','USD','CAD','EUR'))
    limit = st.sidebar.number_input('Insert a timeframe between 0 and 2000', min_value=1,max_value=2000)
    model = load_model('/Users/adityamaniar/Desktop/CryptoForecast/Crypto/cryptomodel.h5')
    
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym='+crypto+'&tsym='+fiat+'&limit='+str(limit))
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'
    hist=hist[hist['close']!=0]
    hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)
    
    train, test = train_test_split(hist)
    
    output1.write("All Data")
    output1.line_chart(hist['close'])
    
    container = st.sidebar.container()

    row1 = container.columns(3)
        
    with row1[0]:
        button1 = st.sidebar.button("Model Data")
    with row1[1]:
        button2 = st.sidebar.button("Forecast Values")
    if button1:
        output.empty()
        output1.empty()
        display_data(hist)
    if button2:
        days = st.sidebar.number_input('Insert number of Days to forecast Crypto Price', min_value=1,max_value=10)
        fore_data = hist
        output.empty()
        output1.empty()
        for i in range(10):
            input_value = (np.array(fore_data[-10:]).reshape(1,10,6))
            
            fore_data_scaled = input_value/input_value[0][0] - 1 #scaling
            
            fore_pred = model.predict(fore_data_scaled)
            
            fore_pred = (fore_pred + 1) * input_value[0][0]#reverse scaling
            
            first_valid_index = fore_data.apply(lambda row: row.first_valid_index(), axis=1)
            new_index = pd.DatetimeIndex([(first_valid_index.index[-1] + pd.Timedelta(days=1))])
            new_row_df = pd.DataFrame(data=fore_pred, index=new_index, columns=fore_data.columns)
            fore_data = pd.concat([fore_data, new_row_df], axis=0)
        
        st.write("Crypto close price for the next "+str(days)+" days:")
        st.line_chart(fore_data['close'][-10:])
    
    

    # train, test, X_train, X_test, y_train, y_test = prepare_data(hist, target_col)

if __name__ == "__main__":
    main()