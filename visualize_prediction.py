import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
import plotly.io as pio
import plotly.graph_objects as go

#index='sp500'
#ticker='ADP'

def tables(data,ticker):

    table=pd.DataFrame({'DAY 1': [data.loc[0,ticker]],'DAY 5': [data.loc[4,ticker]],'DAY 10': [data.loc[9,ticker]],'DAY 20': [data.loc[19,ticker]],'DAY 90': [data.loc[89,ticker]],
          'DAY 180': [data.loc[179,ticker]],'DAY 365': [data.loc[364,ticker]]})
    print(tabulate(table, headers='keys', tablefmt='psql',showindex=False))

def bollinger_bands(df):
    # takes dataframe on input
    sma = df.rolling(window=20).mean().dropna()
    rstd = df.rolling(window=20).std().dropna()

    upper_band = sma + 2 * rstd
    lower_band = sma - 2 * rstd

    upper_band = upper_band.rename(columns={'Close': 'upper'})
    lower_band = lower_band.rename(columns={'Close': 'lower'})
    bb = df.join(upper_band).join(lower_band)
    bb = bb.dropna()

    buyers = bb[bb['Close'] <= bb['lower']]
    sellers = bb[bb['Close'] >= bb['upper']]


    pio.templates.default = "plotly_dark"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lower_band.index,
                            y=lower_band['lower'],
                            name='Lower Band',
                            line_color='rgba(173,204,255,0.2)'
                            ))
    fig.add_trace(go.Scatter(x=upper_band.index,
                            y=upper_band['upper'],
                            name='Upper Band',
                            fill='tonexty',
                            fillcolor='rgba(173,204,255,0.2)',
                            line_color='rgba(173,204,255,0.2)'
                            ))
    fig.add_trace(go.Scatter(x=df.index,
                            y=df['Close'],
                            name='Close',
                            line_color='#636EFA'
                            ))
    fig.add_trace(go.Scatter(x=sma.index,
                            y=sma['Close'],
                            name='SMA',
                            line_color='#FECB52'
                            ))
    fig.add_trace(go.Scatter(x=buyers.index,
                            y=buyers['Close'],
                            name='Buyers',
                            mode='markers',
                            marker=dict(
                                color='#00CC96',
                                size=10,
                                )
                            ))
    fig.add_trace(go.Scatter(x=sellers.index,
                            y=sellers['Close'],
                            name='Sellers',
                            mode='markers',
                            marker=dict(
                                color='#EF553B',
                                size=10,
                                )
                            ))
    fig.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser() # get a parser object
  parser.add_argument('--index', metavar='index', required=True,
                      help='Index to predict the stocks.') # add a required argument
  parser.add_argument('--ticker', metavar='ticker', required=True,
                      help='Ticker from the mentioned index.') # add a required argument
  args = parser.parse_args()
  filename='stocks_predict_'+args.index+'.csv'
  try:
    data=pd.read_csv(filename)
  except:
    print('please first generate the csv file for '+args.index+' indexes by running stock_prediction.py')
    quit()
  tables(data,args.ticker)
  projected_price=pd.DataFrame()
  projected_price['Close']=data[args.ticker]
  bollinger_bands(projected_price)

