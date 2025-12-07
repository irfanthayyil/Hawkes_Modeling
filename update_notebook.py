import json
import os

notebook_path = r'c:\Users\Irfan\Hawkes_Modeling\notebooks\01_data_prep.ipynb'

new_code = [
    "def determine_aggressor_side(trades_df, orders_df):\n",
    "    \"\"\"\n",
    "    Determines the aggressor side for each trade based on order entry timestamps.\n",
    "    \n",
    "    Args:\n",
    "        trades_df (pd.DataFrame): DataFrame containing trade data with 'buy_order_number' and 'sell_order_number'.\n",
    "        orders_df (pd.DataFrame): DataFrame containing order data with 'order_number', 'timestamp', and 'activity_type'.\n",
    "        \n",
    "    Returns:\n",
    "        pd.DataFrame: The trades_df with added columns 'buy_entry_ts', 'sell_entry_ts', and 'aggressor_side'.\n",
    "                      aggressor_side: +1 if Buyer Initiated, -1 if Seller Initiated, 0 if unknown/same time.\n",
    "    \"\"\"\n",
    "    # Filter for ENTRY orders to get original arrival time\n",
    "    if 'activity_type' in orders_df.columns:\n",
    "        orders_entry = orders_df[orders_df['activity_type'] == 'ENTRY']\n",
    "    else:\n",
    "        orders_entry = orders_df\n",
    "    \n",
    "    # Create lookup for order timestamps\n",
    "    # Using drop_duplicates to keep the first occurrence of each order_number\n",
    "    orders_ts = orders_entry.drop_duplicates('order_number').set_index('order_number')['timestamp']\n",
    "    \n",
    "    # Map timestamps to trades\n",
    "    trades_df = trades_df.copy()\n",
    "    trades_df['buy_entry_ts'] = trades_df['buy_order_number'].map(orders_ts)\n",
    "    trades_df['sell_entry_ts'] = trades_df['sell_order_number'].map(orders_ts)\n",
    "    \n",
    "    # Extract timestamps for vectorized comparison\n",
    "    buy_ts = trades_df['buy_entry_ts']\n",
    "    sell_ts = trades_df['sell_entry_ts']\n",
    "    \n",
    "    # Initialize aggressor column with 0\n",
    "    aggressor = np.zeros(len(trades_df), dtype=int)\n",
    "    \n",
    "    # Create masks for existence of timestamps\n",
    "    buy_exists = pd.notna(buy_ts)\n",
    "    sell_exists = pd.notna(sell_ts)\n",
    "    \n",
    "    # Logic:\n",
    "    # 1. If both timestamps exist: The LATER timestamp is the aggressor.\n",
    "    # 2. If one timestamp is missing: The MISSING one is assumed to be OLDER (from previous day/session),\n",
    "    #    so the PRESENT one is NEWER and therefore the aggressor.\n",
    "    \n",
    "    # Case 1: Both exist\n",
    "    both_exist = buy_exists & sell_exists\n",
    "    # Buy > Sell implies Buy arrived later -> Buy Aggressor (+1)\n",
    "    aggressor[both_exist & (buy_ts > sell_ts)] = 1\n",
    "    # Sell > Buy implies Sell arrived later -> Sell Aggressor (-1)\n",
    "    aggressor[both_exist & (sell_ts > buy_ts)] = -1\n",
    "    \n",
    "    # Case 2: Only Buy is present (Sell is missing/older) -> Buy is Newer -> Buy Aggressor (+1)\n",
    "    aggressor[buy_exists & ~sell_exists] = 1\n",
    "    \n",
    "    # Case 3: Only Sell is present (Buy is missing/older) -> Sell is Newer -> Sell Aggressor (-1)\n",
    "    aggressor[~buy_exists & sell_exists] = -1\n",
    "    \n",
    "    trades_df['aggressor_side'] = aggressor\n",
    "    return trades_df\n"
]

print(f"Reading {notebook_path}...")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def determine_aggressor_side(trades_df, orders_df):" in source and "trades_df.apply(get_aggressor, axis=1)" in source:
            print("Found target cell. Updating content...")
            cell['source'] = new_code
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found! Please check the notebook content.")
