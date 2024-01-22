import yfinance as yf
import pandas as pd
import json

def write_dataframe_to_csv(filename, dataframe):
    with open(filename, "w") as f:
        dataframe.to_csv(f)

ticker = "AY"
ticker_reader = yf.Ticker(ticker)

# get stock info
print(ticker_reader.info)
# json from dict
nice_json = json.dumps(ticker_reader.info, indent=4)
with open(f"{ticker}.json", "w") as f:
    f.write(nice_json)

print(nice_json)
# json from dict

# pandas df from dict
df = pd.read_json(nice_json)
print(df)
write_dataframe_to_csv(f"{ticker}_ticker.csv", df)

# get historical market data
hist = ticker_reader.history(period="3mo")
# The data will include stock prices along with dividends and stock splits
print(hist)
write_dataframe_to_csv(f"{ticker}_history.csv", hist)
with open(f"{ticker}_history.csv", "w") as f:
    hist.to_csv(f)

df = ticker_reader.get_financials()
write_dataframe_to_csv(f"{ticker}_financials.csv", df)
print(df)
df = ticker_reader.get_incomestmt()
write_dataframe_to_csv(f"{ticker}_incomestmt.csv", df)
print(df)
df = ticker_reader.get_balance_sheet()
write_dataframe_to_csv(f"{ticker}_balance_sheet.csv", df)
print(df)