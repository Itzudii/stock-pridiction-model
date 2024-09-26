import numpy as np
import math
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def name_get(stock,start_date,end_date):
    start=start_date
    end=end_date
    stock_symbol = stock
    # start='2010-01-01'
    # end='2019-12-31'
    # stock_symbol = 'AAPL'

    df =yf.download(stock_symbol,start,end)
    # tsla = yf.download('TSLA',start,end)

    with plt.rc_context({
    "figure.facecolor": "#15141b",
    "axes.facecolor": "#15141b",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
    "axes.titlecolor": "white",
    }):
       

    # Create a figure and axis with a specified figure size
       fig, ax = plt.subplots(figsize=(16, 8))
       # Set the title, labels, and grid
       close_path = 'static/GRAPH_IMG/close_graph.png'
       ax.set_title('Close Price History', color='white')
       ax.plot(df['Close'], color='blue',label = 'Close')
       ax.grid(color='white')
       ax.set_ylabel('Data', fontsize=18, color='white')
       ax.set_xlabel('Close Price USD ($)', fontsize=18, color='white')
       legend = ax.legend(facecolor='#15141b', edgecolor='white')
       plt.setp(legend.get_texts(), color='white')
       plt.savefig(close_path)

    #    fig, ax = plt.subplots(figsize=(16, 8))
    #    # Set the title, labels, and grid
    #    tsla_path = 'static/GRAPH_IMG/tsla.png'
    #    ax.set_title('Close Price History', color='white')
    #    ax.plot(tsla['Close'], color='blue',label = 'Close')
    #    ax.grid(color='white')
    #    ax.set_ylabel('Data', fontsize=18, color='white')
    #    ax.set_xlabel('Close Price USD ($)', fontsize=18, color='white')
    #    legend = ax.legend(facecolor='#15141b', edgecolor='white')
    #    plt.setp(legend.get_texts(), color='white')
    #    plt.savefig(tsla_path)
       


       fig, ax = plt.subplots(figsize=(16, 8))
       ma100_path = 'static/GRAPH_IMG/ma100_graph.png'
       ma100 = df.Close.rolling(100).mean()
       ax.set_title('Closing Price vs Time Chart with 100MA', color='white')
       ax.plot(df.Close, color='blue',label = 'Close')
       ax.plot(ma100, color='red',label = 'MA100')
       ax.grid(color='white')
       ax.set_ylabel('Data', fontsize=18, color='white')
       ax.set_xlabel('Close Price USD ($)', fontsize=18, color='white')
       legend = ax.legend(facecolor='#15141b', edgecolor='white')
       plt.setp(legend.get_texts(), color='white')
       plt.savefig(ma100_path)

    # Save the figure
       fig, ax = plt.subplots(figsize=(16, 8))
       ma200_path = 'static/GRAPH_IMG/ma200_graph.png'
       ma200 = df.Close.rolling(200).mean()
       ax.set_title('Closing Price vs Time Chart with 100MA & 200MA', color='white')
       ax.plot(df['Close'], color='blue',label = 'Close')
       ax.plot(ma100, color='red',label = 'MA100')
       ax.plot(ma200, color='green' , label = 'MA200')
       ax.grid(color='white')
       ax.set_ylabel('Data', fontsize=18, color='white')
       ax.set_xlabel('Close Price USD ($)', fontsize=18, color='white')
       legend = ax.legend(facecolor='#15141b', edgecolor='white')
       plt.setp(legend.get_texts(), color='white')
       plt.savefig(ma200_path)
    #    /////////////////////////////////////////////////

       data = df.filter(['Close'])
       dataset =  data.values
       trainig_data_len = math.ceil(len(dataset) * .8 )
       
       scaler = MinMaxScaler(feature_range=(0,1))
       scaled_data = scaler.fit_transform(dataset)
       
       train_data = scaled_data[0:trainig_data_len , :]
       
       x_train= []
       y_train= []
       for i in range(60,len(train_data)):
           x_train.append(train_data[i-60:i,0])
           y_train.append(train_data[i,0])
       
       x_train ,y_train = np.array(x_train), np.array(y_train)
       
       x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
       model = load_model('keras_model.keras')
       
       test_data = scaled_data[trainig_data_len - 60: , :]
       
       #create data sets x_test and y_test
       x_test = []
       y_test = dataset[trainig_data_len: , :]
       for i in range(60,len(test_data)):
           x_test.append(test_data[i-60:i, 0])
       
       
       x_test = np.array(x_test)
       
       x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
       
       predictions = model.predict(x_test)
       predictions = scaler.inverse_transform(predictions)
       
       train = data[:trainig_data_len]
       valid = data[trainig_data_len:]
       valid['predictions'] = predictions

   
       fig, ax = plt.subplots(figsize=(16, 8))
       prediction_orignal_path = 'static/GRAPH_IMG/prediction_orignal_graph.png'
       ax.set_title('Orignal vs Pridicated price of particular time', color='white')
       ax.plot(valid[['Close']], color='blue' , label = 'Prediction')
       ax.plot(valid[['predictions']], color='orange' , label = 'Prediction')
       ax.grid(color='white')
       ax.set_ylabel('Data', fontsize=18, color='white')
       ax.set_xlabel('Close Price USD ($)', fontsize=18, color='white')
       legend = ax.legend(['orginal price', 'predicted price'],facecolor='#15141b', edgecolor='white')
       plt.setp(legend.get_texts(), color='white')
       plt.savefig(prediction_orignal_path)
