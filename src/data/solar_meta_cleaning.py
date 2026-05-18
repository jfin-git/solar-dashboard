import pandas as pd
import re

def clean_uspvdb_meta(filepath='data/raw/solar_farm_meta.csv'):
    # Read the file
    df = pd.read_csv(filepath)
    
    # The Regex '\s+' catches all types of whitespace (normal, non-breaking, tabs, etc.)
    # 1. Clean Column Names first
    df.columns = [re.sub(r'\s+', ' ', col).strip() for col in df.columns]
    
    # 2. Clean the Content
    for col in df.columns:
        if df[col].dtype == "object":
            # This regex replaces all strange whitespace with a standard space
            df[col] = df[col].str.replace(r'\s+', '', regex=True)
            
    # 3. Clean the Numeric Capacity (Removes "MW" and spaces)
    df['Rated Capacity (AC)'] = df['Rated Capacity (AC)'].str.replace('MW', '').astype(float)
    
    # 4. Add Industry Defaults for New England (Since they were missing)
    # 20-25 degrees is the sweet spot for Maine/CT/MA
    df['p_tilt'] = 20.0
    df['p_azimuth'] = 180.0
    
    # 5. Fix Site Names for filenames (No trailing spaces!)
    df['Project Short Name'] = df['Project Short Name'].str.strip()
    
    # Save the cleaned version
    df.to_csv('data/processed/solar_farm_meta_cleaned.csv', index=False)
    print("Cleaned CSV saved to data/processed/solar_farm_meta_cleaned.csv")
    return df

if __name__ == "__main__":
    clean_uspvdb_meta()
