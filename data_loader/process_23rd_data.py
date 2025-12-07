import os
import sys
import logging
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from data_loader.NseDataLoaderOptimized import NseDataLoaderOptimized

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    loader = NseDataLoaderOptimized()
    
    # Input directories (Raw data)
    orders_dir = "data/orders_23"
    trades_dir = "data/trades_23"
    
    # Output directories (Final destination)
    output_dir = {
        'orders': 'data/outputs/orders',
        'trades': 'data/outputs/trades'
    }
    
    # Symbol with padding as observed in raw data
    raw_symbol = "bbbbbbINFY"
    target_symbol_name = "INFY"
    
    print(f"Processing data for {raw_symbol}...")
    
    summary = loader.process_multiple_days(
        symbol=raw_symbol,
        orders_dir=orders_dir,
        trades_dir=trades_dir,
        output_dir=output_dir,
        use_direct_write=True
    )
    
    print("\nProcessing Summary:")
    print(f"Processed: {summary['processed_count']} files")
    print(f"Total Orders: {summary['total_orders']}")
    print(f"Total Trades: {summary['total_trades']}")
    
    if summary['failed_count'] > 0:
        print(f"Errors: {summary['errors']}")
        
    # Rename files to remove 'bbbbbb' padding from filename
    print("\nRenaming output files...")
    
    # Renaming Orders
    for filename in os.listdir(output_dir['orders']):
        if raw_symbol in filename:
            old_path = os.path.join(output_dir['orders'], filename)
            new_filename = filename.replace(raw_symbol, target_symbol_name)
            new_path = os.path.join(output_dir['orders'], new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

    # Renaming Trades
    for filename in os.listdir(output_dir['trades']):
        if raw_symbol in filename:
            old_path = os.path.join(output_dir['trades'], filename)
            new_filename = filename.replace(raw_symbol, target_symbol_name)
            new_path = os.path.join(output_dir['trades'], new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    main()
