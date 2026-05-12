import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import Optional, List

class DataService:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_data(
        self, 
        symbol: str, 
        period: str = "2y", 
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data from yfinance.
        """
        cache_file = os.path.join(self.data_dir, f"{symbol}_{period}_{interval}.csv")
        
        if use_cache and os.path.exists(cache_file):
            # Check if cache is older than 1 day for daily data
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mod_time).days < 1:
                print(f"Loading fresh cached data for {symbol}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df
            else:
                print(f"Cache expired for {symbol}, downloading...")

        print(f"Downloading data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Basic cleaning
            df = df.dropna()
            
            # Save to cache
            df.to_csv(cache_file)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def fetch_multiple(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        """
        results = {}
        for s in symbols:
            try:
                results[s] = self.fetch_data(s, **kwargs)
            except Exception as e:
                print(f"Skipping {s} due to error: {e}")
        return results

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price.
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            raise ValueError(f"Could not fetch latest price for {symbol}")
        return float(data['Close'].iloc[-1])
