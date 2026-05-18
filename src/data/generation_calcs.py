import pandas as pd
import pvlib
from pvlib.pvsystem import PVSystem, SingleAxisTrackerMount, FixedMount, Array
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import os

def run_final_physics_engine():
    # Load your cleaned meta
    meta_path = 'data/processed/solar_farm_meta_cleaned.csv'
    meta = pd.read_csv(meta_path)
    
    # Standard SAPM temperature parameters for a ground-mounted open rack
    # This is what most professional developers use for utility-scale sites
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    for _, row in meta.iterrows():
        name = row['Project Short Name']
        input_path = f'data/processed/{name}_master_ts.csv'
        
        if not os.path.exists(input_path):
            continue
            
        df = pd.read_csv(input_path, index_col='timestamp', parse_dates=True)
        loc = Location(latitude=row['Latitude'], longitude=row['Longitude'], tz='US/Eastern')
        
        # 1. Mounting
        if 'single-axis' in str(row['Axis Type']).lower():
            mount = SingleAxisTrackerMount(backtrack=True, max_angle=60)
        else:
            mount = FixedMount(surface_tilt=20, surface_azimuth=180)

        # 2. Capacity & Temp Params
        cap_watts = row['Rated Capacity (AC)'] * 1e6
        
        array = Array(
            mount=mount, 
            module_parameters={'pdc0': cap_watts, 'gamma_pdc': -0.004},
            temperature_model_parameters=temp_params # <--- This satisfies the requirement
        )

        # 3. System & ModelChain
        system = PVSystem(arrays=[array], inverter_parameters={'pdc0': cap_watts})

        mc = ModelChain(
            system, 
            loc, 
            aoi_model='no_loss',
            spectral_model='no_loss',
            temperature_model='sapm',
            losses_model='no_loss'
        )
        
        # 4. Weather prep
        weather = df.rename(columns={'GHI': 'ghi', 'Temperature': 'temp_air', 'Wind Speed': 'wind_speed'})
        solpos = loc.get_solarposition(weather.index)
        erbs_data = pvlib.irradiance.erbs(weather['ghi'], solpos['zenith'], weather.index)
        weather['dni'] = erbs_data['dni']
        weather['dhi'] = erbs_data['dhi']

        # 5. Run
        print(f"Running simulation for {name}...")
        mc.run_model(weather)
        
        df['target_mw'] = mc.results.ac.clip(lower=0) / 1e6
        df.to_csv(f'data/processed/{name}_final_training.csv')

if __name__ == "__main__":
    run_final_physics_engine()