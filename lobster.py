"""
LOBSTER Data Processor - High-Frequency Limit Order Book Analysis
https://lobsterdata.com
"""

import numpy as np
import pandas as pd
import os
import requests
import gzip
import shutil
from tqdm import tqdm
from typing import Dict, Tuple, List
from collections import defaultdict
import matplotlib.pyplot as plt

class LOBSTERDownloader:
    """Handles downloading and organizing LOBSTER data"""
    
    BASE_URL = "https://lobsterdata.com/info/"
    
    def __init__(self, symbol: str, date: str, levels: int = 10):
        self.symbol = symbol
        self.date = date
        self.levels = levels
        self.file_types = [
            'orderbook', 'message', 
            'orderbook_updates', 'message_updates'
        ]
        
    def _get_filename(self, file_type: str) -> str:
        return f"{self.symbol}_{self.date}_{self.levels}_{file_type}.csv.gz"
    
    def download_dataset(self, save_dir: str = 'data/lobster') -> None:
        """Download complete LOBSTER dataset for given parameters"""
        os.makedirs(save_dir, exist_ok=True)
        
        for file_type in tqdm(self.file_types, desc="Downloading LOBSTER files"):
            url = f"{self.BASE_URL}{self._get_filename(file_type)}"
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                save_path = os.path.join(save_dir, self._get_filename(file_type))
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                self._extract_gz(save_path)
                
    def _extract_gz(self, file_path: str) -> None:
        """Decompress .gz files"""
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)

class LOBSTERParser:
    """High-performance parser for LOBSTER data files"""
    
    ORDERBOOK_COLUMNS = [
        'ask_price', 'ask_size', 'bid_price', 'bid_size'
    ]
    
    MESSAGE_COLUMNS = [
        'timestamp', 'event_type', 'order_id', 
        'size', 'price', 'direction'
    ]
    
    EVENT_CODES = {
        1: 'New limit order',
        2: 'Cancellation (partial)',
        3: 'Deletion',
        4: 'Execution (visible)',
        5: 'Execution (hidden)',
        6: 'Cross trade',
        7: 'Trading halt'
    }
    
    def __init__(self, data_dir: str = 'data/lobster'):
        self.data_dir = data_dir
        
    def parse_orderbook(self, file_path: str) -> pd.DataFrame:
        """Parse orderbook file into DataFrame with multi-level columns"""
        levels = int(file_path.split('_')[2])
        df = pd.read_csv(file_path, header=None)
        
        # Generate column names for multiple levels
        columns = []
        for level in range(1, levels+1):
            columns += [f'ask_price_{level}', f'ask_size_{level}', 
                       f'bid_price_{level}', f'bid_size_{level}']
            
        df.columns = columns
        return df
    
    def parse_messages(self, file_path: str) -> pd.DataFrame:
        """Parse message file with event type decoding"""
        df = pd.read_csv(file_path, header=None)
        df.columns = self.MESSAGE_COLUMNS
        df['event_description'] = df['event_type'].map(self.EVENT_CODES)
        return df
    
    def reconstruct_lob(self, orderbook_df: pd.DataFrame, 
                       messages_df: pd.DataFrame) -> Dict[float, Dict]:
        """Reconstruct limit order book at each timestamp"""
        lob = defaultdict(dict)
        current_book = self._initialize_book(orderbook_df.iloc[0])
        
        for _, row in tqdm(messages_df.iterrows(), total=len(messages_df), desc="Reconstructing LOB"):
            timestamp = row['timestamp']
            self._update_book(current_book, row)
            lob[timestamp] = self._snapshot_book(current_book)
            
        return lob
    
    def _initialize_book(self, initial_data: pd.Series) -> Dict:
        """Initialize order book from first orderbook entry"""
        book = {'asks': [], 'bids': []}
        
        for level in range(1, self.levels+1):
            book['asks'].append({
                'price': initial_data[f'ask_price_{level}'],
                'size': initial_data[f'ask_size_{level}']
            })
            book['bids'].append({
                'price': initial_data[f'bid_price_{level}'],
                'size': initial_data[f'bid_size_{level}']
            })
            
        return book
    
    def _update_book(self, book: Dict, event: pd.Series) -> None:
        """Update order book based on market event"""
        if event['event_type'] == 1:  # New order
            self._add_order(book, event)
        elif event['event_type'] in [2, 3]:  # Cancel/Delete
            self._remove_order(book, event)
        elif event['event_type'] in [4, 5]:  # Execution
            self._execute_order(book, event)

class LOBSTERAggregator:
    """Aggregate LOB data into different time intervals"""
    
    def __init__(self, lob_data: Dict):
        self.lob = lob_data
        self.timestamps = sorted(lob_data.keys())
        
    def resample(self, interval: str = '1S') -> pd.DataFrame:
        """Resample LOB to specified time interval"""
        df = pd.DataFrame.from_dict(self.lob, orient='index')
        df.index = pd.to_datetime(df.index, unit='s')
        return df.resample(interval).last().ffill()

class LOBVisualizer:
    """Visualization tools for LOBSTER data"""
    
    @staticmethod
    def plot_order_book(snapshot: Dict, levels: int = 5) -> None:
        """Visualize bid/ask ladder"""
        bids = sorted(snapshot['bids'], key=lambda x: x['price'], reverse=True)[:levels]
        asks = sorted(snapshot['asks'], key=lambda x: x['price'])[:levels]
        
        plt.figure(figsize=(10, 6))
        plt.barh([b['price'] for b in bids], [b['size'] for b in bids], color='green')
        plt.barh([a['price'] for a in asks], [a['size'] for a in asks], color='red')
        plt.xlabel('Order Size')
        plt.ylabel('Price')
        plt.title('Limit Order Book Snapshot')
        plt.tight_layout()

# Example Usage
if __name__ == "__main__":
    # Download LOBSTER data
    downloader = LOBSTERDownloader(symbol='AAPL', date='2023-01-03', levels=10)
    downloader.download_dataset()
    
    # Parse data files
    parser = LOBSTERParser()
    orderbook = parser.parse_orderbook('data/lobster/AAPL_2023-01-03_10_orderbook.csv')
    messages = parser.parse_messages('data/lobster/AAPL_2023-01-03_10_message.csv')
    
    # Reconstruct LOB
    lob_data = parser.reconstruct_lob(orderbook, messages)
    
    # Analyze and visualize
    aggregator = LOBSTERAggregator(lob_data)
    lob_1s = aggregator.resample('1S')
    
    # Plot first 5 order book snapshots
    for ts in list(lob_data.keys())[:5]:
        LOBVisualizer.plot_order_book(lob_data[ts])
    
    # Save processed data
    lob_1s.to_parquet('data/lobster/processed/AAPL_2023-01-03_1s.parquet')
