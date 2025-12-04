import pandas as pd
import gzip
from datetime import datetime
import logging


class NseDataLoader:
    """
    Handles loading and initial vectorized cleaning of NSE fixed-width .gz files.
    """

    # Define fixed-width specifications based on NSE documentation
    ORDER_SCHEMA = {
        'cols': [
            'record_type', 'segment', 'order_number', 'order_time', 'side', 'activity_type',
            'symbol', 'series', 'volume_disclosed', 'volume_original', 'limit_price',
            'trigger_price', 'is_market_order', 'is_stop_loss', 'is_ioc', 'algo_indicator', 'client_type'
        ],
        'widths': [
            2, 4, 16, 14, 1, 1, 10, 2, 8, 8, 8, 8, 1, 1, 1, 1, 1
        ],
        'dtypes': {
            'record_type': 'str', 'segment': 'str', 'order_number': 'int64', 'order_time': 'int64',
            'side': 'str', 'activity_type': 'int64', 'symbol': 'str', 'series': 'str',
            'volume_disclosed': 'int64', 'volume_original': 'int64', 'limit_price': 'int64',
            'trigger_price': 'int64', 'is_market_order': 'str', 'is_stop_loss': 'str',
            'is_ioc': 'str', 'algo_indicator': 'int64',
            # --- ERROR FIX: Read ambiguous single-char fields as 'str' to avoid conversion errors
            'client_type': 'str'
        }
    }

    TRADE_SCHEMA = {
        'cols': [
            'record_type', 'segment', 'trade_number', 'trade_time', 'symbol', 'series',
            'trade_price', 'trade_quantity', 'buy_order_number', 'buy_algo', 'buy_client',
            'sell_order_number', 'sell_algo', 'sell_client'
        ],
        'widths': [
            2, 4, 16, 14, 10, 2, 8, 8, 16, 1, 1, 16, 1, 1
        ],
        'dtypes': {
            'record_type': 'str', 'segment': 'str', 'trade_number': 'int64', 'trade_time': 'int64',
            'symbol': 'str', 'series': 'str', 'trade_price': 'int64', 'trade_quantity': 'int64',
            'buy_order_number': 'int64', 'buy_algo': 'int64',
            # --- ERROR FIX: Read ambiguous single-char fields as 'str' ---
            'buy_client': 'str',
            'sell_order_number': 'int64', 'sell_algo': 'int64',
            'sell_client': 'str'
        }
    }

    EPOCH = datetime(1980, 1, 1)
   

    def _read_file(self, filepath, schema):
        """Reads a .gz fixed-width file into a DataFrame."""
        try:
            # Use gzip to open the file in text mode
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                # Use pandas read_fwf, which is optimized for this task
                df = pd.read_fwf(
                    f,
                    widths=schema['widths'],
                    names=schema['cols'],
                    header=None,
                    dtype=schema['dtypes']  # Enforce types during read
                )
            return df
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {e}")
            return pd.DataFrame(columns=schema['cols'])

    def _jiffies_to_datetime(self, jiffies_series):
        """Vectorized conversion of NSE 'jiffies' to datetime objects."""
        # 65536 jiffies = 1 second
        return pd.to_timedelta(jiffies_series / 65536, unit='s') + self.EPOCH
    
    def _convert_paise_to_rupees(self, price_paise):
        """
        Converts a price in paise to Rupees.

        Args:
            price_paise: The price in paise.

        Returns:
            The price in Rupees.
        """
        return price_paise / 100

    def _clean_data(self, df, file_type):
        """Applies all vectorized cleaning and type conversion steps."""
        if df.empty:
            return df

        # 1. Filter for Regular Market ('RM') and 'EQ' series
        # df = df[(df['record_type'] == 'RM') & (df['series'] == 'EQ')].reset_index(drop=True)

        df = df[df['series'] == 'EQ'].reset_index(drop=True)
        
        # 2. Convert timestamps
        time_col = 'order_time' if file_type == 'order' else 'trade_time'
        df['timestamp'] = self._jiffies_to_datetime(df[time_col])

        # 3. Convert prices from paise to Rupees
        if file_type == 'order':
            df['limit_price'] = df['limit_price'] / 100.0
            df['trigger_price'] = df['trigger_price'] / 100.0
            
            # 4. Map categorical codes
            df['side'] = df['side'].map({'B': 'BUY', 'S': 'SELL'})
            df['is_buy'] = df['side'] == 'BUY'
            df['activity_type'] = df['activity_type'].map({1: 'ENTRY', 3: 'CANCEL', 4: 'MODIFY'})
            df['is_market_order'] = df['is_market_order'] == 'Y'
            df['is_ioc'] = df['is_ioc'] == 'Y'
            
            # Safely convert client_type to numeric after reading as str
            df['client_type'] = pd.to_numeric(df['client_type'], errors='coerce').fillna(0).astype(int)
            
            # Rename for easier merging later
            df = df.rename(columns={'volume_original': 'volume'})
            df = df.drop(columns=['order_time']) # Drop original jiffies col

        elif file_type == 'trade':
            df['trade_price'] = df['trade_price'] / 100.0
            
            # Safely convert client types to numeric
            df['buy_client'] = pd.to_numeric(df['buy_client'], errors='coerce').fillna(0).astype(int)
            df['sell_client'] = pd.to_numeric(df['sell_client'], errors='coerce').fillna(0).astype(int)

            # Rename for easier merging
            df = df.rename(columns={'trade_quantity': 'volume'})
            df = df.drop(columns=['trade_time']) # Drop original jiffies col
        
        return df

    def load_data(self, order_filepath, trade_filepath):
        """
        Public method to load and clean both order and trade files.
        
        Args:
            order_filepath (str): Path to the CASH_Orders_...DAT.gz file.
            trade_filepath (str): Path to the CASH_Trades_...DAT.gz file.
            
        Returns:
            (pd.DataFrame, pd.DataFrame): A tuple of (cleaned_orders_df, cleaned_trades_df)
        """
        logging.info("Loading and cleaning order data...")
        orders_df = self._read_file(order_filepath, self.ORDER_SCHEMA)
        orders_df = self._clean_data(orders_df, 'order')
        
        logging.info("Loading and cleaning trade data...")
        trades_df = self._read_file(trade_filepath, self.TRADE_SCHEMA)
        trades_df = self._clean_data(trades_df, 'trade')
        
        return orders_df, trades_df