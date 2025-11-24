import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title("Stock Price Predictor App")

    stock = st.text_input("Enter the Stock ID", "GOOG")
    end = datetime.now()
    start = datetime(end.year-10, end.month, end.day)

    data = yf.download(stock, end=end, start=start)

    data_close = data['Close'].values 

    data_close = data_close.reshape(-1, 1) 

    st.subheader("Stock Data")
    st.write(data.describe())


    # Plot 1: Closing Price vs Time chart
    st.subheader("Closing Price vs Time chart")
    fig1, ax1 = plt.subplots(figsize=(16, 6)) # Create a figure and axes
    ax1.plot(data['Close'])
    ax1.set_title('Close Price History')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot 2: Closing Price vs Time chart with 100MA & 200MA
    st.subheader("Closing Price vs Time chart with 100MA & 200MA")
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    fig2, ax2 = plt.subplots(figsize=(12, 6)) # Create a new figure and axes
    ax2.plot(data.Close, label='Close Price')
    ax2.plot(ma100, label='100MA')
    ax2.plot(ma200, label='200MA')
    ax2.legend()
    ax2.set_title('Close Price with Moving Averages')
    st.pyplot(fig2)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)

    X_train, X_test, X_val, y_train, y_test, y_val = create_data(scaled_data, n_future=1, n_past=60, train_test_split_percentage=0.8,
                                               validation_split_percentage = 0)
    model = load_model("latest_model.keras")

    # Making prediction

    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    st.subheader('Prediction vs Real')
    fig3 = plot_prediction(y_test, y_pred, stock)
    st.pyplot(fig3)



def create_data(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []
    for i in range(n_past, len(df) - n_future + 1):
        x_data.append(df[i - n_past:i, 0:n_feature])
        y_data.append(df[i + n_future - 1:i + n_future, 0])
    
    split_training_test_starting_point = int(round(train_test_split_percentage*len(x_data)))
    split_train_validation_starting_point = int(round(split_training_test_starting_point*(1-validation_split_percentage)))
    
    x_train = x_data[:split_train_validation_starting_point]
    y_train = y_data[:split_train_validation_starting_point]
    
    # if you want to choose the validation set by yourself, uncomment the below code.
    x_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]
    y_val =  x_data[split_train_validation_starting_point:split_training_test_starting_point]                                             
    
    x_test = x_data[split_training_test_starting_point:]
    y_test = y_data[split_training_test_starting_point:]
    
    return np.array(x_train), np.array(x_test), np.array(x_val), np.array(y_train), np.array(y_test), np.array(y_val)

def plot_prediction(test,prediction, stock):
    fig, ax = plt.subplots()
    ax.plot(test, color='red', label="Real")
    ax.plot(prediction, color="blue", label="Predicted")
    ax.set_title(f"{stock} Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{stock}")
    ax.legend()
    return fig

if __name__ == '__main__':
    main()
