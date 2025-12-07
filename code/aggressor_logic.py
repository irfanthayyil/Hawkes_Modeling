import pandas as pd
import numpy as np
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def determine_aggressor_side(trades_df, orders_df):
    """
    Determines the aggressor side for each trade based on order entry timestamps.
    
    MARKET MICROSTRUCTURE EXPLANATION:
    -----------------------------------
    NSE uses a price-time priority algorithm for order matching:
    - Orders rest in the order book (passive liquidity providers)
    - New incoming orders that cross the spread match with resting orders (aggressors)
    - The aggressor is the side that arrived LATER and demanded immediate execution
    - Buyer-initiated: Active buy order matches with passive sell order (buyer crossed the spread)
    - Seller-initiated: Active sell order matches with passive buy order (seller crossed the spread)
    
    LOGIC:
    ------
    1. For each trade, we identify the buy_order_number and sell_order_number
    2. We look up the ENTRY timestamp for each order in the orders file
    3. The order with the LATER timestamp is the aggressor:
       - If buy_entry_ts > sell_entry_ts → Buyer arrived later → Buyer-initiated (+1)
       - If sell_entry_ts > buy_entry_ts → Seller arrived later → Seller-initiated (-1)
    
    EDGE CASES:
    -----------
    - If only buy order found in orders file: Buy arrived first (passive), Seller is aggressor (-1)
    - If only sell order found in orders file: Sell arrived first (passive), Buyer is aggressor (+1)
    - If both timestamps equal or both missing: Unknown/Simultaneous (0)
    
    Args:
        trades_df (pd.DataFrame): DataFrame containing trade data with:
            - 'buy_order_number': Order number of the buy side
            - 'sell_order_number': Order number of the sell side
        orders_df (pd.DataFrame): DataFrame containing order data with:
            - 'order_number': Unique identifier for each order
            - 'timestamp': Order arrival time (in jiffies or datetime)
            - 'activity_type': Type of order activity ('ENTRY', 'MODIFY', 'CANCEL')
            - 'side': Order side ('BUY' or 'SELL')
        
    Returns:
        pd.DataFrame: The trades_df with added columns:
            - 'buy_entry_ts': Entry timestamp of the buy order
            - 'sell_entry_ts': Entry timestamp of the sell order
            - 'aggressor_side': +1 if Buyer Initiated, -1 if Seller Initiated, 0 if unknown/same time
    
    Raises:
        ValueError: If required columns are missing from input DataFrames
    """
    
    # Validate input DataFrames
    required_trade_cols = ['buy_order_number', 'sell_order_number']
    required_order_cols = ['order_number', 'timestamp']
    
    missing_trade_cols = [col for col in required_trade_cols if col not in trades_df.columns]
    missing_order_cols = [col for col in required_order_cols if col not in orders_df.columns]
    
    if missing_trade_cols:
        raise ValueError(f"trades_df missing required columns: {missing_trade_cols}")
    if missing_order_cols:
        raise ValueError(f"orders_df missing required columns: {missing_order_cols}")
    
    # Log initial statistics
    logger.info(f"Processing {len(trades_df)} trades with {len(orders_df)} order records")
    
    # Filter for ENTRY orders to get original arrival time
    # ENTRY represents when the order first entered the order book
    # MODIFY and CANCEL are subsequent activities on the same order_number
    if 'activity_type' in orders_df.columns:
        orders_entry = orders_df[orders_df['activity_type'] == 'ENTRY'].copy()
        logger.info(f"Filtered to {len(orders_entry)} ENTRY orders from {len(orders_df)} total order activities")
    else:
        # Fallback if activity_type is missing
        logger.warning("'activity_type' column not found in orders_df. Using all order records.")
        orders_entry = orders_df.copy()
    
    # Create lookup dictionary for order timestamps
    # Using drop_duplicates to ensure one timestamp per order_number
    # For ENTRY activity, order_number should be unique, but this is a safety measure
    if orders_entry['order_number'].duplicated().any():
        n_dupes = orders_entry['order_number'].duplicated().sum()
        logger.warning(f"Found {n_dupes} duplicate order_numbers in ENTRY orders. Keeping first occurrence.")
    
    orders_ts = orders_entry.drop_duplicates('order_number').set_index('order_number')['timestamp']
    logger.info(f"Created timestamp lookup for {len(orders_ts)} unique order numbers")
    
    # Create a copy to avoid modifying the original DataFrame
    trades_df = trades_df.copy()
    
    # Map entry timestamps to trades for both buy and sell sides
    trades_df['buy_entry_ts'] = trades_df['buy_order_number'].map(orders_ts)
    trades_df['sell_entry_ts'] = trades_df['sell_order_number'].map(orders_ts)
    
    # Check for missing order numbers (trades where order is not in order file)
    buy_missing = trades_df['buy_entry_ts'].isna().sum()
    sell_missing = trades_df['sell_entry_ts'].isna().sum()
    both_missing = (trades_df['buy_entry_ts'].isna() & trades_df['sell_entry_ts'].isna()).sum()
    
    if buy_missing > 0:
        logger.warning(f"{buy_missing} trades have buy_order_number not found in orders file")
    if sell_missing > 0:
        logger.warning(f"{sell_missing} trades have sell_order_number not found in orders file")
    if both_missing > 0:
        logger.warning(f"{both_missing} trades have BOTH order numbers missing from orders file")
    
    def get_aggressor(row):
        """
        Determine aggressor side for a single trade row.
        
        Returns:
            +1: Buyer initiated (buy order arrived later, crossed the spread)
            -1: Seller initiated (sell order arrived later, crossed the spread)
             0: Unknown or simultaneous
        """
        buy_ts = row['buy_entry_ts']
        sell_ts = row['sell_entry_ts']
        
        buy_present = pd.notna(buy_ts)
        sell_present = pd.notna(sell_ts)
        
        if buy_present and sell_present:
            # Both orders found in order file - compare timestamps
            if buy_ts > sell_ts:
                # Buy order arrived AFTER sell order
                # Sell order was resting (passive), buy order crossed the spread (active)
                return 1  # Buyer initiated
            elif sell_ts > buy_ts:
                # Sell order arrived AFTER buy order
                # Buy order was resting (passive), sell order crossed the spread (active)
                return -1  # Seller initiated
            else:
                # Same timestamp - extremely rare but theoretically possible
                # Cannot determine aggressor
                return 0
        elif buy_present and not sell_present:
            # Only buy order found in order file
            # This means buy arrived first (passive), sell arrived later but not recorded
            # Sell order is the aggressor
            return -1  # Seller initiated
        elif not buy_present and sell_present:
            # Only sell order found in order file
            # This means sell arrived first (passive), buy arrived later but not recorded
            # Buy order is the aggressor
            return 1  # Buyer initiated
        else:
            # Both missing from order file - cannot determine
            return 0
    
    # Apply aggressor determination to all trades
    trades_df['aggressor_side'] = trades_df.apply(get_aggressor, axis=1)
    
    # Log final statistics
    buyer_initiated = (trades_df['aggressor_side'] == 1).sum()
    seller_initiated = (trades_df['aggressor_side'] == -1).sum()
    unknown = (trades_df['aggressor_side'] == 0).sum()
    
    logger.info(f"Aggressor determination complete:")
    logger.info(f"  Buyer-initiated: {buyer_initiated} ({buyer_initiated/len(trades_df)*100:.2f}%)")
    logger.info(f"  Seller-initiated: {seller_initiated} ({seller_initiated/len(trades_df)*100:.2f}%)")
    logger.info(f"  Unknown/Simultaneous: {unknown} ({unknown/len(trades_df)*100:.2f}%)")
    
    return trades_df
