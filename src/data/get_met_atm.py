import os
import requests
import pandas as pd
import io
import time

# --- CONFIG ---
API_KEY = "eVrTdZuo6Gr8aYyPcgU1jDDg9qcmNH7FCFUFiRJZ"
EMAIL = "jsfinney1@gmail.com"
META_PATH = 'data/processed/solar_farm_meta.csv'
OUTPUT_DIR = 'data/raw'
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"
ATTRIBUTES = 'air_temperature,clearsky_ghi,ghi,cloud_type,dew_point,relative_humidity,solar_zenith_angle,wind_speed'
YEARS = ['2022', '2023', '2024']

# All possible features from NSRDB
# YEARS = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']
# ATTRIBUTES = 'air_temperature,clearsky_ghi,ghi,cloud_type,dew_point,relative_humidity,solar_zenith_angle,surface_albedo,wind_speed'


def download_from_meta():
    # 1. Load your metadata
    if not os.path.exists(META_PATH):
        print(f"Error: Could not find metadata file at {META_PATH}")
        return
    
    meta_df = pd.read_csv(META_PATH)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Iterate through each farm in the CSV
    for _, row in meta_df.iterrows():
        # 1. Clean the name to prevent file path errors
        site_name = str(row['Project Short Name']).strip().replace(" ", "_")
    
        # 2. Force convert to float and strip any potential whitespace/formatting
        lat = float(str(row['Latitude']).strip().replace(",", ""))
        lon = float(str(row['Longitude']).strip().replace(",", ""))
    
        # 3. WKT must be exactly: POINT(lon lat) with NO comma
        wkt = f"POINT({lon} {lat})"
    
        print(f"\n--- Processing: {site_name} ({lat}, {lon}) ---")
        # Format WKT for the API
        wkt = f"POINT({lon} {lat})"
        
        for year in YEARS:
            params = {
                'api_key': API_KEY,
                'email': EMAIL,
                'names': year,
                'wkt': wkt,
                'attributes': ATTRIBUTES,
                'interval': '5',
                'utc': 'false', 
                'full_name': 'SolarForecaster',
                'reason': 'Research'
            }

            print(f"  Requesting {year}...")
            response = requests.get(BASE_URL, params=params)

            if response.status_code == 200:
                # Read CSV, skipping the metadata header rows
                df = pd.read_csv(io.StringIO(response.text), skiprows=2)
                
                # Tag the data with the Project Name for easier merging later
                df['project_name'] = row['Project Short Name']
                
                save_path = f"{OUTPUT_DIR}/{site_name}_{year}.csv"
                df.to_csv(save_path, index=False)
                print(f"  Saved to {save_path}")
            else:
                print(f"  Failed {year}: {response.status_code}")
            
            # Respect NREL rate limits
            time.sleep(1)

if __name__ == "__main__":
    download_from_meta()