from NseDataLoaderOptimized import NseDataLoaderOptimized
import logging

logging.basicConfig(level=logging.INFO)
loader = NseDataLoaderOptimized()

import os
print(os.getcwd())
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

summary = loader.process_multiple_days(
    symbol="bbbbbbINFY",
    orders_dir="data/orders_23/",
    trades_dir="data/trades_23/",
    output_dir={'orders': 'data/outputs/orders_23/', 'trades': 'data/outputs/trades_23/'},
    use_direct_write=True  # Memory-optimized
)

# summary = loader.process_all_trades(
#     trades_dir='data/trades',
#     symbol='bbbbbbINFY',
#     output_dir='data/outputs/trades'
# )

# print(f"Processed {summary['processed_count']} files")
# print(f"Total trades: {summary['total_trades']}")
# print(f"Duration: {summary['duration_seconds']:.2f} seconds")