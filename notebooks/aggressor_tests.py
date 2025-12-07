import pandas as pd
import numpy as np

def validate_aggressor_logic(df):
    """
    Runs consistency tests on the aggressor logic in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'sell_entry_ts', 'buy_entry_ts', 'aggressor_side', and 'timestamp'.
        Timestamps should be datetime objects.
        
    Returns:
    --------
    dict
        Dictionary containing counts of failures for each rule.
    """
    
    # Ensure datetimes
    for col in ['sell_entry_ts', 'buy_entry_ts', 'timestamp']:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime. {e}")

    results = {
        'rule_1_failures': 0, # sell_entry_ts is NaN
        'rule_2_failures': 0, # buy_entry_ts is NaN
        'rule_3_failures': 0, # Both not NaN
    }
    
    print("Running Aggressor Logic Tests...")
    
    # Rule 1: if 'sell_entry_ts' is NaN
    # check if 'aggressor_side' is -1, and 'timestamp' is after 'buy_entry_ts'
    mask_sell_nan = df['sell_entry_ts'].isna()
    if mask_sell_nan.any():
        sub = df[mask_sell_nan].copy()
        
        # Conditions
        cond_agg = sub['aggressor_side'] == -1
        
        valid_buy_ts = ~sub['buy_entry_ts'].isna()
        cond_time = pd.Series(True, index=sub.index)
        
        # Check time condition only where buy_entry_ts is valid
        if valid_buy_ts.any():
            cond_time[valid_buy_ts] = sub.loc[valid_buy_ts, 'timestamp'] > sub.loc[valid_buy_ts, 'buy_entry_ts']
        
        # Any failure in either condition
        failures = ~(cond_agg & cond_time)
        results['rule_1_failures'] = failures.sum()
        
        if failures.sum() > 0:
            print(f"Rule 1 Failed for {failures.sum()} rows (Sell Entry NaN).")
            rule_1_failures = sub[failures]
            # print("Sample failures:", sub[failures].head())

    # Rule 2: if buy_entry_ts in NaN
    # check if 'aggressor_side' is +1, and 'timestamp' is after 'sell_entry_ts'
    mask_buy_nan = df['buy_entry_ts'].isna()
    if mask_buy_nan.any():
        sub = df[mask_buy_nan].copy()
        
        cond_agg = sub['aggressor_side'] == 1
        
        valid_sell_ts = ~sub['sell_entry_ts'].isna()
        cond_time = pd.Series(True, index=sub.index)
        
        if valid_sell_ts.any():
            cond_time[valid_sell_ts] = sub.loc[valid_sell_ts, 'timestamp'] > sub.loc[valid_sell_ts, 'sell_entry_ts']
        
        failures = ~(cond_agg & cond_time)
        results['rule_2_failures'] = failures.sum()
        
        if failures.sum() > 0:
            print(f"Rule 2 Failed for {failures.sum()} rows (Buy Entry NaN).")
            rule_2_failures = sub[failures]
                            
    # Rule 3: if both 'buy_entry_ts' and 'sell_entry_ts' are not NaN
    # check if 'aggressor_side' is 1, the buy_entry_ts > sell_entry_ts, and otherwise
    mask_both = (~df['buy_entry_ts'].isna()) & (~df['sell_entry_ts'].isna())
    if mask_both.any():
        sub = df[mask_both].copy()
        
        # If Aggressor 1: Buy > Sell
        mask_agg1 = sub['aggressor_side'] == 1
        fail_agg1 = sub[mask_agg1 & (sub['buy_entry_ts'] <= sub['sell_entry_ts'])]
        
        # Otherwise (assuming Aggressor -1 or just NOT 1): Buy < Sell 
        # (Strictly speaking: Sell > Buy)
        mask_not_agg1 = sub['aggressor_side'] != 1
        fail_not_agg1 = sub[mask_not_agg1 & (sub['buy_entry_ts'] >= sub['sell_entry_ts'])] # >= includes equality which might be fail
        
        check_failures = len(fail_agg1) + len(fail_not_agg1)
        
        results['rule_3_failures'] = check_failures
        
        if check_failures > 0:
             print(f"Rule 3 Failed for {check_failures} rows (Both entries present).")
             
        print("Test Complete.")
    return results, rule_1_failures, rule_2_failures, fail_agg1, fail_not_agg1
