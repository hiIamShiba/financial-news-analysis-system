import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_historical_trend(ticker, days=30):
    """
    Sử dụng Yahoo Finance để lấy dữ liệu vì nó ổn định và miễn phí.
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df_history = stock.history(start=start_date, end=end_date)

        if df_history.empty:
            print(f"No data found for {ticker} using Yahoo Finance.")
            return None
        df_history = df_history.reset_index()
        df_history = df_history[['Date', 'Close']]
        df_history.columns = ['Date', 'Price']
        df_history['Date'] = df_history['Date'].dt.strftime('%Y-%m-%d')
        
        return df_history

    except Exception as e:
        print(f"Yahoo Finance Error for {ticker}: {str(e)}")
        return None

if __name__ == "__main__":
    test_df = get_stock_historical_trend("AAPL")
    if test_df is not None:
        print(test_df.tail())