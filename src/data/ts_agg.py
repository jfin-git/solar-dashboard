import pandas as pd
import glob
import os

def aggregate_time_series(raw_dir='data/raw', processed_dir='data/processed'):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    # Get all unique site names from the filenames (Farmington, DWW, etc.)
    all_files = glob.glob(f"{raw_dir}/*.csv")
    site_names = set([os.path.basename(f).split('_20')[0] for f in all_files])
    
    for site in site_names:
        print(f"Aggregating data for {site}...")
        
        # 1. Grab all years for this specific site
        site_files = glob.glob(f"{raw_dir}/{site}_*.csv")
        site_files.sort() # Ensure years are in order
        
        df_list = [pd.read_csv(f) for f in site_files]
        full_df = pd.concat(df_list, ignore_index=True)
        
        # 2. Convert separate columns to a single Datetime object
        # NREL columns: Year, Month, Day, Hour, Minute
        full_df['timestamp'] = pd.to_datetime(full_df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        
        # 3. Set timestamp as index (Standard for Time-Series Analysis)
        full_df.set_index('timestamp', inplace=True)
        
        # 4. Drop the original split columns to keep the dataset lean
        full_df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
        
        # 5. Sort index (important for 5-minute continuity)
        full_df.sort_index(inplace=True)
        
        # Save the master file for this site
        save_path = f"{processed_dir}/{site}_master_ts.csv"
        full_df.to_csv(save_path)
        print(f"  Successfully created {save_path} with {len(full_df)} observations.")

if __name__ == "__main__":
    aggregate_time_series()