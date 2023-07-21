import argparse
import certifi
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


API_KEY='ff170cfbc214454f0a10844eb6e9606e' #put the API key here
N_steps = 365

#INDEX='dowjones' #sp500 nasdaq
def denormalize(test_predict,df_close_prices):
  stocks=df_close_prices.columns.values.tolist()
  i=0
  #df_test=pd.DataFrame()
  df_predicted=pd.DataFrame()
  for ticker in stocks:
    stock_data=df_close_prices[ticker]
    ss= MinMaxScaler(feature_range=(0,1))
    data_close_scaled= ss.fit_transform(np.array(stock_data).reshape(-1,1))

    #test_gt=ss.inverse_transform(test_y[:,i].reshape(-1,1))
    #print(ticker)
    test_pred=ss.inverse_transform(test_predict[:,i].reshape(-1,1))
    #df_test[ticker]=test_gt.reshape(test_gt.shape[0]).tolist()
    df_predicted[ticker]=test_pred.reshape(test_pred.shape[0]).tolist()
    i=i+1
  return df_predicted

def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
 # find the end of this pattern
    end_ix = i + n_steps
 # check if we are beyond the dataset
    if end_ix > len(sequences)-1:
      break
 # gather input and output parts of the pattern
    seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def get_data(index):
  if index not in ['top10', 'etf']:
    url = ("https://financialmodelingprep.com/api/v3/"+index+"_constituent?apikey="+API_KEY)
    data= get_jsonparsed_data(url)
    stocks=list()
    for info in data:
      symbol=info['symbol']
      #if index=='sp500' and symbol=='EG':
        #continue
      stocks.append(symbol)

  df_close_prices = pd.DataFrame()

  if index=='top10':
    stocks=('SPY', 'QQQ', 'TSLA', 'NVDA', 'AAPL', 'AMZN', 'GOOGL', 'META', 'NFLX', 'ORCL')

  if index=='etf':
    stocks=('XLK', 'XLV','XLY','XLF','XLC', 'XLI','XLP','XLE','XLU','XLRE','XLB','SPY','QQQ','DIA')

  #stocks=('SPY', 'QQQ', 'TSLA', 'NVDA', 'AAPL', 'AMZN', 'GOOGL', 'META', 'NFLX', 'ORCL')
  close_array=[]
  for ticker in stocks:
      #print(ticker)
      url = ("https://financialmodelingprep.com/api/v3/historical-price-full/"+ticker+"?from=2015-03-12&to=2023-07-04&apikey="+API_KEY)
      data= get_jsonparsed_data(url)
      try:
        df = pd.DataFrame.from_dict(data['historical'])
      except:
        print('Data from ticker '+ticker+' is not available at this moment. Try again later')
      data_close= df['close']
      data_close = data_close.iloc[::-1]
      data_close=data_close.reset_index()['close']
      df_close_prices[ticker]=data_close


  print(df_close_prices)
  missing_val_stocks=[]
  for ticker in stocks:
    if len(df_close_prices[df_close_prices[ticker].isnull()].index.tolist())>0:
      missing_val_stocks.append(ticker)
      print(df_close_prices[df_close_prices[ticker].isnull()].index.tolist())
      #i=i+1
  df_close_prices=df_close_prices.drop(missing_val_stocks, axis=1)
  #print(df_close_prices)
  #print(missing_val_stocks)
  ss= MinMaxScaler(feature_range=(0,1))

  data_close_scaled= ss.fit_transform(np.array(df_close_prices))

  if index=='dowjones':
    df_close_prices_scaled = pd.DataFrame(data=data_close_scaled)
    df_close_prices_scaled=df_close_prices_scaled.replace(to_replace=0, method='ffill')
    data_close_scaled=np.array(df_close_prices_scaled)

  return data_close_scaled, df_close_prices

#close_data=np.array(close_array)
#close_data=np.transpose(close_data)
#print(close_data)



'''
training_size=int(len(data_close_scaled)*0.7)
test_size=len(data_close_scaled)-training_size
train_data,test_data=data_close_scaled[0:training_size,:],data_close_scaled[training_size:len(data_close_scaled),:]
'''
def prep_data(data_close_scaled):
  training_size=len(data_close_scaled)-2*N_steps
  test_size=2*N_steps
  train_data,test_data=data_close_scaled[0:training_size,:],data_close_scaled[training_size:len(data_close_scaled),:]

  # convert into input/output
  train_X, train_y = split_sequences(train_data, N_steps)
  test_X, test_y = split_sequences(test_data, N_steps)
  return test_X

if __name__ == '__main__':
  parser = argparse.ArgumentParser() # get a parser object
  parser.add_argument('--index', metavar='index', required=True,
                      help='Index to predict the stocks.') # add a required argument
  parser.add_argument('--model_dir', metavar='model_dir', required=True,
                      help='directory path for lstm models.') # add a required argument
  args = parser.parse_args()
  data_close_scaled, df_close_prices=get_data(args.index)
  input_data=prep_data(data_close_scaled)

  loaded_model = tf.keras.models.load_model(args.model_dir+"/stock_lstm_"+args.index+".keras")
  data_predict=loaded_model.predict(input_data)
  stocks_predict=denormalize(data_predict,df_close_prices)
  stocks_predict.to_csv('stocks_predict_'+args.index+'.csv')

  #print(data_close_scaled)
