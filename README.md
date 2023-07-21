# Stock Prediction using LSTM

Code for predicting stocks' close prices from indexes like S&P500, NASDAQ, DOW, ETF (also includes Top 10 stocks)

## Installation

Install dependencies for the project by running command: pip3 install -r requirements.txt

## Get Predictions

1. First run the code to accumulate the index data, load model, get predictions and save it to a csv file: 

python3 stock_prediction.py --index INDEX_NAME --model_dir /path/to/model/dir

INDEX_NAME: index name. 5 options: sp500, dow, nasdaq, etf and top10

/path/to/model/dir: directory path for the saved models

2. Now run the code to visualize predictions in a table and a bollinger chart:

python3 visualize_prediction.py --index INDEX_NAME --ticker TICKER

INDEX_NAME: index name. 5 options: sp500, dow, nasdaq, etf and top10
TICKER: ticker name from the respective index

### Note

The following tickers from some indexes were excluded since they had many missing values. Avoid giving these tickers while visualizing results from an index


NASDAQ
['ABNB',
 'TEAM',
 'CEG',
 'CRWD',
 'DDOG',
 'GEHC',
 'GFS',
 'KHC',
 'LCID',
 'MRNA',
 'PYPL',
 'PDD',
 'ZM',
 'ZS']

DOW
['DOW']

S&P500
['CARR',
 'CDAY',
 'CEG',
 'CTVA',
 'DOW',
 'ETSY',
 'EG',
 'FTV',
 'FOXA',
 'GEHC',
 'HPE',
 'IR',
 'INVH',
 'KHC',
 'LW',
 'MRNA',
 'OGN',
 'OTIS',
 'PYPL',
 'SEDG',
 'VICI',
 'WRK']

ETF
['XLC', 'XLRE']
