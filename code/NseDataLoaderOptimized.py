import gzip
import logging
import itertools
import pandas as pd

# assuming NseDataLoader is defined as in your file
from NseDataLoader import NseDataLoader

class NseDataLoaderOptimized(NseDataLoader):
    """
    Extension of NseDataLoader that can stream large .DAT.gz files
    and extract a single symbol's orders/trades in a memory-efficient way.
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

    def load_symbol_for_day(self, order_filepath, trade_filepath, symbol):
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

        Returns:
            (orders_df, trades_df): cleaned DataFrames as in NseDataLoader.load_data
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

        return orders_df, trades_df
