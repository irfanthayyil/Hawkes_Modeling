import os
import pandas as pd
import logging
import re
from pathlib import Path
from preprocessing.aggressor_calculator import aggressor_calculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path("C:/Users/Irfan/Hawkes_Modeling")
DATA_DIR = PROJECT_ROOT / "data" / "outputs"
TRADES_DIR = DATA_DIR / "trades"
ORDERS_DIR = DATA_DIR / "orders"
OUTPUT_FILE = DATA_DIR / "hawkes_modeling_data.csv"

def extract_date_from_filename(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def process_all_files():
    """
    Reads all trade and order files, matches them by date, 
    calculates aggressors, and saves a consolidated dataset.
    """
    logger.info("Starting data preparation for Hawkes analysis...")
    
    # Ensure output directories exist
    if not TRADES_DIR.exists():
        logger.error(f"Trades directory not found: {TRADES_DIR}")
        return
    if not ORDERS_DIR.exists():
        logger.error(f"Orders directory not found: {ORDERS_DIR}")
        return

    # List all files
    trade_files = [f for f in os.listdir(TRADES_DIR) if f.startswith("INFY_trades_") and f.endswith(".csv")]
    
    processed_dfs = []
    
    for trade_file in trade_files:
        # Extract date string like '13082019'
        date_str = extract_date_from_filename(trade_file, r"INFY_trades_(\d{8})\.csv")
        
        if not date_str:
            logger.warning(f"Could not extract date from filename: {trade_file}")
            continue
            
        # Construct expected order filename
        order_file = f"INFY_orders_{date_str}.csv"
        order_path = ORDERS_DIR / order_file
        
        if not order_path.exists():
            logger.warning(f"Order file matching {trade_file} not found: {order_file}. Skipping this date.")
            continue
            
        logger.info(f"Processing data for date: {date_str}")
        
        try:
            # Load data
            trade_path = TRADES_DIR / trade_file
            trades_df = pd.read_csv(trade_path)
            orders_df = pd.read_csv(order_path)
            
            # Calculate aggressor
            # This adds 'aggressor_side', 'buy_entry_ts', 'sell_entry_ts'
            trades_with_aggressor = aggressor_calculator(trades_df, orders_df)
            
            # Add date column for reference
            # Assuming format DDMMYYYY
            try:
                date_formatted = pd.to_datetime(date_str, format='%d%m%Y').strftime('%Y-%m-%d')
                trades_with_aggressor['date'] = date_formatted
            except Exception as e:
                logger.warning(f"Could not parse date string {date_str}: {e}")
                trades_with_aggressor['date'] = date_str
                
            processed_dfs.append(trades_with_aggressor)
            logger.info(f"Successfully processed {len(trades_with_aggressor)} trades for {date_str}")
            
        except Exception as e:
            logger.error(f"Error processing files for date {date_str}: {e}")
            
    # Concatenate and save
    if processed_dfs:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        logger.info(f"Consolidated data created with {len(final_df)} total rows.")
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Saved final dataset to {OUTPUT_FILE}")
    else:
        logger.warning("No data processed.")

if __name__ == "__main__":
    process_all_files()
