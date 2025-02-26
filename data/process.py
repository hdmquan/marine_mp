import pandas as pd
import numpy as np
from pathlib import Path
from pyhdf.SD import SD, SDC
from datetime import datetime
from src.utils import PATH

def get_modis_cell_coords(lat, lon):
    """Convert lat/lon to MODIS cell coordinates"""
    cell_lat = np.floor(lat * 100) / 100
    cell_lon = np.floor(lon * 100) / 100
    return cell_lat, cell_lon

def extract_modis_bands(hdf_file, lat, lon):
    """Extract 7 bands data from MODIS file for given location"""
    dataset = SD(str(hdf_file), SDC.READ)
    
    # Get all surface reflectance bands
    bands = []
    for i in range(1, 8):
        sds = dataset.select(f'sur_refl_b{i:02d}')
        data = sds.get()
        # Convert lat/lon to pixel coordinates (simplified - needs adjustment)
        row = int((90 - lat) * dataset.dimensions()[0] / 180)
        col = int((lon + 180) * dataset.dimensions()[1] / 360)
        bands.append(data[row, col])
    
    return bands

def main():
    # Read MP data
    mp_data = pd.read_csv(PATH.RAW / "Marine Microplastics WGS84.csv")
    mp_data['Date'] = pd.to_datetime(mp_data['Date'])
    
    # Create final dataset
    results = []
    
    for _, row in mp_data.iterrows():
        date = row['Date']
        lat = row['Latitude']
        lon = row['Longitude']
        
        # Get MODIS cell coordinates
        cell_lat, cell_lon = get_modis_cell_coords(lat, lon)
        
        # Find corresponding MODIS file
        date_str = date.strftime('%Y%j')
        h = int((lon + 180) / 10)
        v = int((90 - lat) / 10)
        modis_file = PATH.RAW / f'MYD09GA.A{date_str}.h{h:02d}v{v:02d}.006.hdf'
        
        if modis_file.exists():
            # Extract bands data
            bands = extract_modis_bands(modis_file, lat, lon)
            
            results.append({
                'date': date.date(),
                'cell_lat': cell_lat,
                'cell_lon': cell_lon,
                'mp_reading': row['Measurement'],
                'band_1': bands[0],
                'band_2': bands[1],
                'band_3': bands[2],
                'band_4': bands[3],
                'band_5': bands[4],
                'band_6': bands[5],
                'band_7': bands[6]
            })
    
    # Create final DataFrame and save
    final_df = pd.DataFrame(results)
    final_df.to_csv(PATH.PROCESSED / 'modis_mp_dataset.csv', index=False)

if __name__ == "__main__":
    main()