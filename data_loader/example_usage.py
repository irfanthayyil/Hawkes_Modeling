"""
Example script demonstrating how to use NseDataLoaderOptimized for batch processing.

This script shows how to:
1. Process a single day's files for a symbol
2. Process multiple days with automatic file discovery
3. Use both DataFrame-based and direct CSV writing methods
"""

import logging
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.NseDataLoaderOptimized import NseDataLoaderOptimized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def example_single_day():
    """Process a single day's data for a symbol."""
    print("\n" + "="*60)
    print("Example 1: Processing Single Day")
    print("="*60)
    
    loader = NseDataLoaderOptimized()
    
    # Define paths (adjust these to your actual file locations)
    order_file = "data/orders/CASH_Orders_19082019.DAT.gz"
    trade_file = "data/trades/CASH_Trades_19082019.DAT.gz"
    symbol = "INFY"
    
    # Method 1: Get DataFrames without saving
    print("\nMethod 1: Getting DataFrames only (no file saving)")
    orders_df, trades_df = loader.load_symbol_for_day(
        order_filepath=order_file,
        trade_filepath=trade_file,
        symbol=symbol
    )
    print(f"Orders DataFrame shape: {orders_df.shape}")
    print(f"Trades DataFrame shape: {trades_df.shape}")
    
    # Method 2: Get DataFrames AND save to CSV
    print("\nMethod 2: Getting DataFrames and saving to CSV")
    output_directories = {
        'orders': 'data/outputs/orders',
        'trades': 'data/outputs/trades'
    }
    orders_csv, trades_csv = loader.load_symbol_for_day(
        order_filepath=order_file,
        trade_filepath=trade_file,
        symbol=symbol,
        output_dir=output_directories
    )
    print(f"Saved orders to: {orders_csv}")
    print(f"Saved trades to: {trades_csv}")
    
    # Method 3: Direct CSV writing (memory-optimized)
    print("\nMethod 3: Direct CSV writing (memory-optimized)")
    orders_csv, trades_csv, stats = loader.load_and_save_symbol_for_day_direct(
        order_filepath=order_file,
        trade_filepath=trade_file,
        symbol=symbol,
        output_dir=output_directories
    )
    print(f"Orders: {stats['orders_count']} rows → {orders_csv}")
    print(f"Trades: {stats['trades_count']} rows → {trades_csv}")


def example_batch_processing():
    """Process multiple days with automatic file discovery."""
    print("\n" + "="*60)
    print("Example 2: Batch Processing Multiple Days")
    print("="*60)
    
    loader = NseDataLoaderOptimized()
    
    # Define directories
    orders_dir = "data/orders"  # Contains CASH_Orders_*.DAT.gz files
    trades_dir = "data/trades"  # Contains CASH_Trades_*.DAT.gz files
    symbol = "INFY"
    
    output_directories = {
        'orders': 'data/outputs/orders',
        'trades': 'data/outputs/trades'
    }
    
    # Process all available file pairs
    print(f"\nProcessing all files for {symbol}...")
    summary = loader.process_multiple_days(
        symbol=symbol,
        orders_dir=orders_dir,
        trades_dir=trades_dir,
        output_dir=output_directories,
        use_direct_write=True  # Use memory-optimized approach
    )
    
    # Print summary
    print("\n" + "-"*60)
    print("PROCESSING SUMMARY")
    print("-"*60)
    print(f"Symbol: {summary['symbol']}")
    print(f"Status: {summary['status']}")
    print(f"Files processed: {summary['processed_count']}/{summary['total_files']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Total orders extracted: {summary['total_orders']:,}")
    print(f"Total trades extracted: {summary['total_trades']:,}")
    print(f"Duration: {summary['duration_seconds']:.2f} seconds")
    
    if summary['errors']:
        print(f"\nErrors encountered:")
        for error in summary['errors']:
            print(f"  - {error['order_file']}: {error['error']}")
    
    print("\nProcessed files:")
    for result in summary['results']:
        print(f"  Date {result['date']}:")
        print(f"    Orders: {result['orders_csv']}")
        print(f"    Trades: {result['trades_csv']}")


def example_file_discovery():
    """Demonstrate automatic file discovery."""
    print("\n" + "="*60)
    print("Example 3: Automatic File Discovery")
    print("="*60)
    
    loader = NseDataLoaderOptimized()
    
    orders_dir = "data/orders"
    trades_dir = "data/trades"
    
    # Discover available file pairs
    file_pairs = loader.discover_file_pairs(orders_dir, trades_dir)
    
    print(f"\nFound {len(file_pairs)} matching file pairs:")
    for order_file, trade_file in file_pairs:
        date = loader._extract_date_from_filename(order_file)
        print(f"  Date {date}:")
        print(f"    Order: {os.path.basename(order_file)}")
        print(f"    Trade: {os.path.basename(trade_file)}")


if __name__ == "__main__":
    # Run examples
    # Uncomment the examples you want to run:
    
    # example_single_day()
    example_batch_processing()
    # example_file_discovery()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
