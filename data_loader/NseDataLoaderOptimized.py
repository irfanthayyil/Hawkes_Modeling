import gzip
import logging
import itertools
import pandas as pd
import os
import re
import csv
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

# assuming NseDataLoader is defined as in your file
from NseDataLoader import NseDataLoader

class NseDataLoaderOptimized(NseDataLoader):
    """
    Extension of NseDataLoader that can stream large .DAT.gz files
    and extract a single symbol's orders/trades in a memory-efficient way.
    
    Features:
    - Streaming approach for large files with limited RAM
    - Symbol-specific extraction from multi-symbol files
    - File-saving capabilities with configurable output directories
    - Direct CSV writing to minimize memory usage
    - Batch processing with automatic file discovery
    """

    def _stream_fwf_symbol(self, filepath, schema, symbol, file_type):
        """
        Stream a fixed-width .DAT.gz file and return a DataFrame for ONE symbol.

        Args:
            filepath (str): path to CASH_Orders_YYYYMMDD.DAT.gz or CASH_Trades_YYYYMMDD.DAT.gz
            schema (dict): ORDER_SCHEMA or TRADE_SCHEMA
            symbol (str): e.g. 'INFY'
            file_type (str): 'order' or 'trade'

        Returns:
            pd.DataFrame with raw fields as per schema['cols']
            (jiffies still in {order_time, trade_time}, no cleaning yet)
        """
        cols    = schema['cols']
        widths  = schema['widths']
        n_cols  = len(cols)

        # precompute slice indices
        offsets = [0] + list(itertools.accumulate(widths))
        # index helpers
        idx_record   = cols.index('record_type') if 'record_type' in cols else None
        idx_segment  = cols.index('segment') if 'segment' in cols else None
        idx_symbol   = cols.index('symbol') if 'symbol' in cols else None
        idx_series   = cols.index('series')  if 'series'  in cols else None

        # initialize column-wise storage
        data = {c: [] for c in cols}

        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # cheap length guard
                if len(line) < offsets[-1]:
                    continue

                # slice the line according to widths
                fields = [line[offsets[i]:offsets[i+1]] for i in range(n_cols)]

                # basic filters (vectorized at line-level)
                rec_type = fields[idx_record].strip()   if idx_record is not None else ''
                seg      = fields[idx_segment].strip()  if idx_segment is not None else ''
                sym      = fields[idx_symbol].strip()   if idx_symbol is not None else ''
                ser      = fields[idx_series].strip()   if idx_series is not None else ''

                # Regular Market, CASH segment, requested symbol, EQ series only
                if rec_type != 'RM':
                    continue
                if seg != 'CASH':
                    continue
                if sym != symbol:
                    continue
                if ser != 'EQ':
                    continue

                # keep this row
                for c, v in zip(cols, fields):
                    data[c].append(v.strip())

        if not any(len(v) for v in data.values()):
            # no rows found
            return pd.DataFrame(columns=cols)

        # Build DataFrame
        df = pd.DataFrame(data, columns=cols)

        # Convert dtypes according to schema (post-hoc, small df now)
        for col, dtype in schema['dtypes'].items():
            if col not in df.columns:
                continue
            if dtype.startswith('int') or dtype.startswith('float'):
                # numeric conversion, robust to blanks
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
            else:
                df[col] = df[col].astype(dtype)

        return df

    def _extract_date_from_filename(self, filepath: str) -> str:
        """
        Extract date from NSE data filename.
        
        Args:
            filepath: Path like 'CASH_Orders_19082019.DAT.gz' or full path
            
        Returns:
            Date string in DDMMYYYY format (e.g., '19082019')
        """
        filename = Path(filepath).name
        # Pattern: CASH_Orders_DDMMYYYY.DAT.gz or CASH_Trades_DDMMYYYY.DAT.gz
        match = re.search(r'_(\d{8})\.DAT', filename)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract date from filename: {filename}")
    
    def _create_output_path(self, output_dir: str, symbol: str, date: str, file_type: str) -> str:
        """
        Generate standardized output file path.
        
        Args:
            output_dir: Base output directory
            symbol: NSE symbol
            date: Date string in DDMMYYYY format
            file_type: 'orders' or 'trades'
            
        Returns:
            Full path to output CSV file
        """
        filename = f"{symbol}_{file_type}_{date}.csv"
        return os.path.join(output_dir, filename)

    def load_symbol_for_day(self, order_filepath, trade_filepath, symbol, output_dir=None):
        """
        Load and clean order & trade data for a single symbol (e.g. INFY) for one day.

        This is optimized for huge full-universe .DAT.gz files:
        it streams line-by-line, keeps only relevant lines, then
        uses the existing _clean_data() to get the same output format
        as load_data().

        Args:
            order_filepath (str): path to CASH_Orders_YYYYMMDD.DAT.gz
            trade_filepath (str): path to CASH_Trades_YYYYMMDD.DAT.gz
            symbol (str): NSE symbol, e.g. 'INFY'
            output_dir (dict or None): Optional. If provided, should be dict with keys:
                'orders': directory for order CSV files
                'trades': directory for trade CSV files
                Example: {'orders': 'data/outputs/orders', 'trades': 'data/outputs/trades'}

        Returns:
            If output_dir is None:
                (orders_df, trades_df): cleaned DataFrames
            If output_dir is provided:
                (orders_csv_path, trades_csv_path): paths to saved CSV files
        """
        logging.info(f"Streaming orders for symbol={symbol} from {order_filepath} ...")
        raw_orders = self._stream_fwf_symbol(
            filepath=order_filepath,
            schema=self.ORDER_SCHEMA,
            symbol=symbol,
            file_type='order'
        )
        orders_df = self._clean_data(raw_orders, 'order')

        logging.info(f"Streaming trades for symbol={symbol} from {trade_filepath} ...")
        raw_trades = self._stream_fwf_symbol(
            filepath=trade_filepath,
            schema=self.TRADE_SCHEMA,
            symbol=symbol,
            file_type='trade'
        )
        trades_df = self._clean_data(raw_trades, 'trade')
        
        # Save to CSV if output directory provided
        if output_dir is not None:
            date = self._extract_date_from_filename(order_filepath)
            
            # Create output directories if they don't exist
            os.makedirs(output_dir['orders'], exist_ok=True)
            os.makedirs(output_dir['trades'], exist_ok=True)
            
            # Generate output paths
            orders_csv_path = self._create_output_path(output_dir['orders'], symbol, date, 'orders')
            trades_csv_path = self._create_output_path(output_dir['trades'], symbol, date, 'trades')
            
            # Save DataFrames
            logging.info(f"Saving orders to {orders_csv_path}")
            orders_df.to_csv(orders_csv_path, index=False)
            
            logging.info(f"Saving trades to {trades_csv_path}")
            trades_df.to_csv(trades_csv_path, index=False)
            
            logging.info(f"Saved {len(orders_df)} orders and {len(trades_df)} trades for {symbol} on {date}")
            return orders_csv_path, trades_csv_path

        return orders_df, trades_df

    def load_and_save_symbol_for_day_direct(self, order_filepath, trade_filepath, symbol, output_dir):
        """
        Memory-optimized version that writes directly to CSV without building full DataFrames.
        
        This method streams the input files, filters for the symbol, and writes cleaned data
        directly to CSV files. This minimizes memory usage for very large datasets.
        
        Args:
            order_filepath (str): path to CASH_Orders_YYYYMMDD.DAT.gz
            trade_filepath (str): path to CASH_Trades_YYYYMMDD.DAT.gz
            symbol (str): NSE symbol, e.g. 'INFY'
            output_dir (dict): Dict with keys 'orders' and 'trades' specifying output directories
            
        Returns:
            (orders_csv_path, trades_csv_path, stats): Tuple of output paths and statistics dict
        """
        date = self._extract_date_from_filename(order_filepath)
        
        # Create output directories
        os.makedirs(output_dir['orders'], exist_ok=True)
        os.makedirs(output_dir['trades'], exist_ok=True)
        
        # Generate output paths
        orders_csv_path = self._create_output_path(output_dir['orders'], symbol, date, 'orders')
        trades_csv_path = self._create_output_path(output_dir['trades'], symbol, date, 'trades')
        
        # Process orders
        logging.info(f"Direct CSV writing: orders for {symbol} from {order_filepath}")
        orders_count = self._stream_and_write_csv(
            filepath=order_filepath,
            output_csv=orders_csv_path,
            schema=self.ORDER_SCHEMA,
            symbol=symbol,
            file_type='order'
        )
        
        # Process trades
        logging.info(f"Direct CSV writing: trades for {symbol} from {trade_filepath}")
        trades_count = self._stream_and_write_csv(
            filepath=trade_filepath,
            output_csv=trades_csv_path,
            schema=self.TRADE_SCHEMA,
            symbol=symbol,
            file_type='trade'
        )
        
        stats = {
            'symbol': symbol,
            'date': date,
            'orders_count': orders_count,
            'trades_count': trades_count
        }
        
        logging.info(f"Direct write complete: {orders_count} orders, {trades_count} trades for {symbol} on {date}")
        return orders_csv_path, trades_csv_path, stats
    
    def _stream_and_write_csv(self, filepath, output_csv, schema, symbol, file_type):
        """
        Stream a DAT.gz file, filter by symbol, clean data, and write directly to CSV.
        
        This processes data in small chunks to minimize memory usage.
        
        Returns:
            int: Number of rows written
        """
        cols = schema['cols']
        widths = schema['widths']
        n_cols = len(cols)
        
        # Precompute slice indices
        offsets = [0] + list(itertools.accumulate(widths))
        
        # Index helpers
        idx_record = cols.index('record_type') if 'record_type' in cols else None
        idx_segment = cols.index('segment') if 'segment' in cols else None
        idx_symbol = cols.index('symbol') if 'symbol' in cols else None
        idx_series = cols.index('series') if 'series' in cols else None
        
        # Determine output columns after cleaning
        if file_type == 'order':
            output_cols = [
                'record_type', 'segment', 'order_number', 'timestamp', 'side', 'activity_type',
                'symbol', 'series', 'volume_disclosed', 'volume', 'limit_price',
                'trigger_price', 'is_market_order', 'is_stop_loss', 'is_ioc', 'algo_indicator',
                'client_type', 'is_buy'
            ]
        else:  # trade
            output_cols = [
                'record_type', 'segment', 'trade_number', 'timestamp', 'symbol', 'series',
                'trade_price', 'volume', 'buy_order_number', 'buy_algo', 'buy_client',
                'sell_order_number', 'sell_algo', 'sell_client'
            ]
        
        # Chunk size for batch processing
        chunk_size = 1000
        data_buffer = {c: [] for c in cols}
        row_count = 0
        header_written = False
        
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Cheap length guard
                if len(line) < offsets[-1]:
                    continue
                
                # Slice the line
                fields = [line[offsets[i]:offsets[i+1]] for i in range(n_cols)]
                
                # Filter
                rec_type = fields[idx_record].strip() if idx_record is not None else ''
                seg = fields[idx_segment].strip() if idx_segment is not None else ''
                sym = fields[idx_symbol].strip() if idx_symbol is not None else ''
                ser = fields[idx_series].strip() if idx_series is not None else ''
                
                if rec_type != 'RM' or seg != 'CASH' or sym != symbol or ser != 'EQ':
                    continue
                
                # Buffer this row
                for c, v in zip(cols, fields):
                    data_buffer[c].append(v.strip())
                
                # Process buffer when it reaches chunk_size
                if len(data_buffer[cols[0]]) >= chunk_size:
                    chunk_df = pd.DataFrame(data_buffer, columns=cols)
                    
                    # Convert dtypes
                    for col, dtype in schema['dtypes'].items():
                        if col not in chunk_df.columns:
                            continue
                        if dtype.startswith('int') or dtype.startswith('float'):
                            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0).astype(dtype)
                        else:
                            chunk_df[col] = chunk_df[col].astype(dtype)
                    
                    # Clean data
                    chunk_df = self._clean_data(chunk_df, file_type)
                    
                    # Write to CSV
                    chunk_df.to_csv(output_csv, mode='a', header=not header_written, index=False)
                    header_written = True
                    row_count += len(chunk_df)
                    
                    # Clear buffer
                    data_buffer = {c: [] for c in cols}
        
        # Process remaining data in buffer
        if any(len(v) for v in data_buffer.values()):
            chunk_df = pd.DataFrame(data_buffer, columns=cols)
            
            # Convert dtypes
            for col, dtype in schema['dtypes'].items():
                if col not in chunk_df.columns:
                    continue
                if dtype.startswith('int') or dtype.startswith('float'):
                    chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0).astype(dtype)
                else:
                    chunk_df[col] = chunk_df[col].astype(dtype)
            
            # Clean data
            chunk_df = self._clean_data(chunk_df, file_type)
            
            # Write to CSV
            chunk_df.to_csv(output_csv, mode='a', header=not header_written, index=False)
            row_count += len(chunk_df)
        
        return row_count

    def discover_file_pairs(self, orders_dir, trades_dir):
        """
        Automatically discover matching order and trade file pairs by date.
        
        Args:
            orders_dir (str): Directory containing CASH_Orders_*.DAT.gz files
            trades_dir (str): Directory containing CASH_Trades_*.DAT.gz files
            
        Returns:
            List[Tuple[str, str]]: List of (order_filepath, trade_filepath) pairs sorted by date
        """
        import glob
        
        # Find all order files and extract dates
        order_pattern = os.path.join(orders_dir, 'CASH_Orders_*.DAT.gz')
        order_files = glob.glob(order_pattern)
        
        # Find all trade files and extract dates
        trade_pattern = os.path.join(trades_dir, 'CASH_Trades_*.DAT.gz')
        trade_files = glob.glob(trade_pattern)
        
        # Build date-to-file mappings
        order_map = {}
        for order_file in order_files:
            try:
                date = self._extract_date_from_filename(order_file)
                order_map[date] = order_file
            except ValueError:
                logging.warning(f"Could not extract date from {order_file}, skipping")
        
        trade_map = {}
        for trade_file in trade_files:
            try:
                date = self._extract_date_from_filename(trade_file)
                trade_map[date] = trade_file
            except ValueError:
                logging.warning(f"Could not extract date from {trade_file}, skipping")
        
        # Find matching pairs
        common_dates = sorted(set(order_map.keys()) & set(trade_map.keys()))
        file_pairs = [(order_map[date], trade_map[date]) for date in common_dates]
        
        logging.info(f"Discovered {len(file_pairs)} matching file pairs for dates: {common_dates}")
        return file_pairs

    def process_multiple_days(self, symbol, orders_dir, trades_dir, output_dir, use_direct_write=True):
        """
        Process multiple days of order and trade files for a symbol with automatic file discovery.
        
        This method automatically discovers matching order/trade file pairs by date,
        processes them sequentially to minimize memory usage, and saves cleaned data to CSV.
        
        Args:
            symbol (str): NSE symbol to extract (e.g., 'INFY')
            orders_dir (str): Directory containing CASH_Orders_*.DAT.gz files
            trades_dir (str): Directory containing CASH_Trades_*.DAT.gz files
            output_dir (dict): Dict with keys 'orders' and 'trades' for output directories
                Example: {'orders': 'data/outputs/orders', 'trades': 'data/outputs/trades'}
            use_direct_write (bool): If True, use memory-optimized direct CSV writing
            
        Returns:
            dict: Summary statistics including processed files, total rows, and any errors
        """
        start_time = datetime.now()
        logging.info(f"Starting batch processing for symbol {symbol}")
        logging.info(f"Orders directory: {orders_dir}")
        logging.info(f"Trades directory: {trades_dir}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Direct write mode: {use_direct_write}")
        
        # Discover file pairs
        file_pairs = self.discover_file_pairs(orders_dir, trades_dir)
        
        if not file_pairs:
            logging.warning("No matching file pairs found!")
            return {
                'status': 'no_files',
                'processed_count': 0,
                'failed_count': 0,
                'total_orders': 0,
                'total_trades': 0,
                'errors': []
            }
        
        # Process each file pair
        results = []
        errors = []
        total_orders = 0
        total_trades = 0
        
        for i, (order_file, trade_file) in enumerate(file_pairs, 1):
            try:
                date = self._extract_date_from_filename(order_file)
                logging.info(f"[{i}/{len(file_pairs)}] Processing {symbol} for date {date}...")
                
                if use_direct_write:
                    orders_csv, trades_csv, stats = self.load_and_save_symbol_for_day_direct(
                        order_filepath=order_file,
                        trade_filepath=trade_file,
                        symbol=symbol,
                        output_dir=output_dir
                    )
                    total_orders += stats['orders_count']
                    total_trades += stats['trades_count']
                else:
                    orders_csv, trades_csv = self.load_symbol_for_day(
                        order_filepath=order_file,
                        trade_filepath=trade_file,
                        symbol=symbol,
                        output_dir=output_dir
                    )
                    # Count rows from saved CSV
                    orders_count = sum(1 for _ in open(orders_csv)) - 1  # subtract header
                    trades_count = sum(1 for _ in open(trades_csv)) - 1
                    total_orders += orders_count
                    total_trades += trades_count
                
                results.append({
                    'date': date,
                    'orders_csv': orders_csv,
                    'trades_csv': trades_csv,
                    'status': 'success'
                })
                
            except Exception as e:
                error_msg = f"Error processing {order_file}: {str(e)}"
                logging.error(error_msg)
                errors.append({
                    'order_file': order_file,
                    'trade_file': trade_file,
                    'error': str(e)
                })
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'status': 'completed',
            'symbol': symbol,
            'processed_count': len(results),
            'failed_count': len(errors),
            'total_files': len(file_pairs),
            'total_orders': total_orders,
            'total_trades': total_trades,
            'duration_seconds': duration,
            'results': results,
            'errors': errors
        }
        
        logging.info(f"Batch processing complete!")
        logging.info(f"Processed: {len(results)}/{len(file_pairs)} file pairs")
        logging.info(f"Total orders extracted: {total_orders}")
        logging.info(f"Total trades extracted: {total_trades}")
        logging.info(f"Duration: {duration:.2f} seconds")
        
        if errors:
            logging.warning(f"Encountered {len(errors)} errors during processing")
        
        return summary
