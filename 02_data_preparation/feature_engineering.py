import pandas as pd
import numpy as np
import os
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PROJECT_ROOT

DEBUG = True
DEBUG_SIZE = 10000

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem), flush=True)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem), flush=True)
    
    return df

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    for c in new_columns:
        df[c] = df[c].astype(float)
    return df, new_columns

def load_parquet(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"Loading {filename}...", flush=True)
        df = pd.read_parquet(path)
        return reduce_mem_usage(df)
    else:
        print(f"Warning: {filename} not found.")
        return None

def process_bureau_and_balance(valid_ids=None):
    print("Processing Bureau and Balance...", flush=True)
    bureau = load_parquet('bureau.parquet')
    if bureau is None: return None

    if valid_ids is not None:
        print(f"Filtering Bureau by {len(valid_ids)} IDs...", flush=True)
        bureau = bureau[bureau['SK_ID_CURR'].isin(valid_ids)]

    bb = load_parquet('bureau_balance.parquet')
    # If bb is missing, we can proceed with just bureau, but code expects bb.
    # For now, return None if either is missing to be safe, or handle gracefully.
    if bb is None: 
        print("Warning: bureau_balance.parquet not found. Skipping bureau processing.")
        return None

    if valid_ids is not None:
        # Filter bb by bureau IDs present in filtered bureau
        valid_bureau_ids = bureau['SK_ID_BUREAU'].unique()
        print(f"Filtering Bureau Balance by {len(valid_bureau_ids)} Bureau IDs...", flush=True)
        bb = bb[bb['SK_ID_BUREAU'].isin(valid_bureau_ids)]

    print("OHE Bureau Balance...", flush=True)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    print("OHE Bureau...", flush=True)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)
    
    # Bureau balance: Perform aggregations and merge with bureau.org
    print("Aggregating Bureau Balance...", flush=True)
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    print("Joining Bureau Balance to Bureau...", flush=True)
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    print("Aggregating Bureau...", flush=True)
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    # Check if bb columns exist in bb_agg (renamed)
    # The previous code iterated bb_cat but used bb_agg columns.
    # bb_agg columns are [original_name]_[agg]. 
    # e.g. STATUS_0_MEAN. 
    # But bb_cat contains STATUS_0.
    # So we need to aggregate STATUS_0_MEAN from bb_agg? 
    # No, bb_agg IS aggregated by SK_ID_BUREAU.
    # Now we aggregate bureau (which has joined bb_agg) by SK_ID_CURR.
    # So we need to aggregate the columns that came from bb_agg.
    # The columns from bb_agg are e.g. STATUS_0_MEAN.
    # So we should include them in num_aggregations or cat_aggregations?
    # They are numerical means, so they go into num_aggregations or similar.
    
    # The original code:
    # for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    # This assumes bb_agg produced columns like "STATUS_0_MEAN".
    # And now we aggregate "STATUS_0_MEAN" by SK_ID_CURR using 'mean'.
    
    # Let's verify column names in bb_agg.
    # bb_agg.columns = [e[0] + "_" + e[1].upper() ...]
    # e.g. STATUS_0_MEAN.
    # So yes, bureau has STATUS_0_MEAN.
    # So cat_aggregations[cat + "_MEAN"] = ['mean'] is correct.

    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    
    return bureau_agg

def process_previous_applications(valid_ids=None):
    prev = load_parquet('previous_application.parquet')
    if prev is None: return None
    
    if valid_ids is not None:
        print(f"Filtering Previous Applications by {len(valid_ids)} IDs...", flush=True)
        prev = prev[prev['SK_ID_CURR'].isin(valid_ids)]

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Previous Applications: Approved Applications - only numerical aggregations
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical aggregations
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    del prev, approved, refused, approved_agg, refused_agg
    gc.collect()
    return prev_agg

def process_pos_cash(valid_ids=None):
    pos = load_parquet('POS_CASH_balance.parquet')
    if pos is None: return None
    
    if valid_ids is not None:
        print(f"Filtering POS CASH by {len(valid_ids)} IDs...", flush=True)
        pos = pos[pos['SK_ID_CURR'].isin(valid_ids)]

    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    del pos
    gc.collect()
    return pos_agg

def process_installments(valid_ids=None):
    ins = load_parquet('installments_payments.parquet')
    if ins is None: return None
    
    if valid_ids is not None:
        print(f"Filtering Installments by {len(valid_ids)} IDs...", flush=True)
        ins = ins[ins['SK_ID_CURR'].isin(valid_ids)]

    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    del ins
    gc.collect()
    return ins_agg

def process_credit_card(valid_ids=None):
    cc = load_parquet('credit_card_balance.parquet')
    if cc is None: return None
    
    if valid_ids is not None:
        print(f"Filtering Credit Card by {len(valid_ids)} IDs...", flush=True)
        cc = cc[cc['SK_ID_CURR'].isin(valid_ids)]

    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    
    del cc
    gc.collect()
    return cc_agg

def main():
    print("Starting Feature Engineering...", flush=True)
    if DEBUG:
        print(f"DEBUG MODE ON: Sampling {DEBUG_SIZE} rows from application data.", flush=True)
    
    # Load Application train/test
    df = load_parquet('application_.parquet')
    if df is None: return

    if DEBUG:
        df = df.sample(n=DEBUG_SIZE, random_state=42)
        print(f"Sampled application shape: {df.shape}", flush=True)

    valid_ids = df['SK_ID_CURR'].unique()
    print(f"Valid SK_ID_CURR count: {len(valid_ids)}", flush=True)

    # Simple preprocessing for application data
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical encoding for application data
    df, _ = one_hot_encoder(df, nan_as_category=True)
    
    # Join Bureau
    bureau_agg = process_bureau_and_balance(valid_ids)
    if bureau_agg is not None:
        print("Joining Bureau data...", flush=True)
        df = df.join(bureau_agg, how='left', on='SK_ID_CURR')
        del bureau_agg
        gc.collect()

    # Join Previous Applications
    prev_agg = process_previous_applications(valid_ids)
    if prev_agg is not None:
        print("Joining Previous Application data...", flush=True)
        df = df.join(prev_agg, how='left', on='SK_ID_CURR')
        del prev_agg
        gc.collect()

    # Join POS CASH
    pos_agg = process_pos_cash(valid_ids)
    if pos_agg is not None:
        print("Joining POS CASH data...", flush=True)
        df = df.join(pos_agg, how='left', on='SK_ID_CURR')
        del pos_agg
        gc.collect()
        
    # Join Installments
    ins_agg = process_installments(valid_ids)
    if ins_agg is not None:
        print("Joining Installments data...", flush=True)
        df = df.join(ins_agg, how='left', on='SK_ID_CURR')
        del ins_agg
        gc.collect()
        
    # Join Credit Card
    cc_agg = process_credit_card(valid_ids)
    if cc_agg is not None:
        print("Joining Credit Card data...", flush=True)
        # Ensure no object columns in cc_agg
        for col in cc_agg.columns:
            if cc_agg[col].dtype == 'object':
                print(f"Warning: Column {col} is object. Casting to float/NaN.", flush=True)
                cc_agg[col] = pd.to_numeric(cc_agg[col], errors='coerce')
                
        df = df.join(cc_agg, how='left', on='SK_ID_CURR')
        del cc_agg
        gc.collect()

    print(f"Final Dataset Shape: {df.shape}", flush=True)
    
    # Save processed data
    output_path = os.path.join(DATA_DIR, 'processed_data.parquet')
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
