import pandas as pd
import numpy as np
from pathlib import Path
from pyhdf.SD import SD, SDC
from loguru import logger
import datetime
import os
import re
import pyproj
import rasterio
import geopandas as gpd
from shapely.geometry import Point

from src.utils import PATH

def find_modis_file(date, lat, lon):
    """Find the appropriate MODIS file for a given date and location"""
    # Format date for MODIS filename pattern (YYYYDDD)
    date_str = date.strftime('%Y%j')
    modis_files = list(PATH.RAW.glob(f'MYD09GA.A{date_str}*.hdf'))
    
    if not modis_files:
        logger.warning(f"No MODIS file found for date {date}")
        return None
    
    # If multiple files exist for the same date, we could implement tile selection logic
    # based on lat/lon, but for simplicity we'll use the first file
    return modis_files[0]

def get_modis_geotransform(hdf_file):
    """Extract geotransform information from MODIS HDF file"""
    try:
        # Open the HDF file with rasterio
        with rasterio.open(str(hdf_file)) as src:
            if src.transform and src.transform != rasterio.transform.IDENTITY:
                return src.transform.to_gdal()
            
        # If direct method fails, try to get from metadata using HDF4 reader
        dataset = SD(str(hdf_file), SDC.READ)
        metadata = dataset.attributes()
        
        # Look for common metadata keys that might contain projection info
        projection_keys = [
            'StructMetadata.0', 
            'PROJECTIONINFO', 
            'PROJECTION_INFO',
            'HDFEOS_GRIDS_GRID1_Projection'
        ]
        
        for key in projection_keys:
            if key in metadata:
                # Parse the metadata string to extract projection parameters
                meta_str = metadata[key]
                
                # Extract upper left and lower right coordinates
                ul_regex = r'UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)'
                lr_regex = r'LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)'
                
                ul_match = re.search(ul_regex, meta_str)
                lr_match = re.search(lr_regex, meta_str)
                
                if ul_match and lr_match:
                    ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
                    lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
                    
                    # Get dimensions from first dataset
                    datasets_dict = dataset.datasets()
                    if datasets_dict:
                        first_dataset = list(datasets_dict.keys())[0]
                        sds = dataset.select(first_dataset)
                        height, width = sds.info()[2]
                        
                        # Calculate pixel size
                        pixel_width = (lr_x - ul_x) / width
                        pixel_height = (ul_y - lr_y) / height
                        
                        # Create geotransform (GDAL format: [ul_x, pixel_width, 0, ul_y, 0, -pixel_height])
                        return (ul_x, pixel_width, 0, ul_y, 0, -pixel_height)
        
        logger.warning(f"Using default MODIS sinusoidal parameters for {hdf_file}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting geotransform: {e}")
        return None

def latlon_to_pixel(lat, lon, hdf_file):
    """Convert lat/lon to pixel coordinates in the MODIS grid"""
    try:
        # Try to get geotransform from file
        geotransform = get_modis_geotransform(hdf_file)
        
        if geotransform:
            # Create GeoDataFrame with the point
            point = Point(lon, lat)
            gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
            
            # MODIS uses sinusoidal projection
            modis_crs = pyproj.CRS('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext')
            
            # Reproject point to MODIS projection
            gdf_modis = gdf.to_crs(modis_crs)
            x, y = gdf_modis.geometry.iloc[0].x, gdf_modis.geometry.iloc[0].y
            
            # Convert to pixel coordinates using geotransform
            pixel_x = int((x - geotransform[0]) / geotransform[1])
            pixel_y = int((y - geotransform[3]) / geotransform[5])
            
            return pixel_y, pixel_x
        
        # Fallback to approximate method if geotransform not available
        # Open the HDF file to get dimensions
        dataset = SD(str(hdf_file), SDC.READ)
        dims = dataset.dimensions()
        
        # Try different dimension names that might be present
        y_dim_names = ['YDim_Grid', 'YDim', 'y', 'rows']
        x_dim_names = ['XDim_Grid', 'XDim', 'x', 'columns']
        
        y_dim = None
        for name in y_dim_names:
            if name in dims:
                y_dim = dims[name].length
                break
                
        x_dim = None
        for name in x_dim_names:
            if name in dims:
                x_dim = dims[name].length
                break
        
        if y_dim is None or x_dim is None:
            # If we can't find dimensions, try to get them from the first dataset
            datasets_dict = dataset.datasets()
            if datasets_dict:
                first_dataset = list(datasets_dict.keys())[0]
                sds = dataset.select(first_dataset)
                info = sds.info()
                y_dim, x_dim = info[2]  # dimensions tuple
        
        if y_dim is None or x_dim is None:
            logger.error(f"Could not determine dimensions for {hdf_file}")
            return None
            
        # MODIS sinusoidal grid conversion (approximate)
        row = int((90 - lat) * y_dim / 180)
        col = int((lon + 180) * x_dim / 360)
        
        # Ensure row/col are within bounds
        row = max(0, min(row, y_dim - 1))
        col = max(0, min(col, x_dim - 1))
        
        return row, col
        
    except Exception as e:
        logger.error(f"Error converting lat/lon to pixel: {e}")
        return None

def extract_modis_data(hdf_file, lat, lon):
    """Extract all available bands from MODIS file for given location"""
    try:
        # Open the HDF file
        dataset = SD(str(hdf_file), SDC.READ)
        
        # Get dataset information
        datasets_dict = dataset.datasets()
        
        # MODIS sinusoidal grid is 1200x1200 for 1km data and 2400x2400 for 500m data
        # Convert lat/lon to row/col in the grid (simplified approach)
        
        # For 1km data (1200x1200 grid)
        # The grid covers approximately -180 to 180 longitude and -90 to 90 latitude
        row_1km = int((90 - lat) * 1200 / 180)
        col_1km = int((lon + 180) * 1200 / 360)
        
        # For 500m data (2400x2400 grid)
        row_500m = int((90 - lat) * 2400 / 180)
        col_500m = int((lon + 180) * 2400 / 360)
        
        # Ensure row/col are within bounds
        row_1km = max(0, min(row_1km, 1199))
        col_1km = max(0, min(col_1km, 1199))
        row_500m = max(0, min(row_500m, 2399))
        col_500m = max(0, min(col_500m, 2399))
        
        # Extract all band values
        data = {}
        
        for ds_name in datasets_dict.keys():
            try:
                sds = dataset.select(ds_name)
                
                # Get attributes
                attrs = sds.attributes()
                scale_factor = 1.0
                add_offset = 0.0
                fill_value = None
                
                # Get scale factor if available
                if 'scale_factor' in attrs:
                    scale_factor = attrs['scale_factor']
                
                # Get offset if available
                if 'add_offset' in attrs:
                    add_offset = attrs['add_offset']
                
                # Get fill value if available
                fill_value_keys = ['_FillValue', 'fill_value', '_FILLVALUE', 'missing_value']
                for key in fill_value_keys:
                    if key in attrs:
                        fill_value = attrs[key]
                        break
                
                # Get dimensions
                dims = sds.info()[2]
                
                # Skip datasets that aren't 2D arrays
                if len(dims) != 2:
                    continue
                
                # Choose appropriate row/col based on dataset dimensions
                if dims[0] == 1200 and dims[1] == 1200:  # 1km data
                    row, col = row_1km, col_1km
                elif dims[0] == 2400 and dims[1] == 2400:  # 500m data
                    row, col = row_500m, col_500m
                else:
                    # Skip datasets with unexpected dimensions
                    continue
                
                # Extract the value at the specific pixel
                raw_value = sds[row, col]
                
                # Apply scale and offset if needed
                if isinstance(raw_value, np.ndarray) and raw_value.size == 1:
                    raw_value = raw_value.item()
                
                # Handle fill values
                if fill_value is not None and raw_value == fill_value:
                    data[ds_name] = np.nan
                else:
                    data[ds_name] = raw_value * scale_factor + add_offset
                
            except Exception as e:
                logger.warning(f"Could not extract {ds_name}: {e}")
                data[ds_name] = np.nan
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to process {hdf_file}: {e}")
        return None
    
def clean_data(df):
    # Replace any remaining NaN values with a suitable placeholder
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns with more than 50% NaN values
    df = df.dropna(axis=1, thresh=0.5 * len(df))
    
    # Drop rows with more than 50% NaN values
    df = df.dropna(axis=0, thresh=0.5 * len(df.columns))
    
    # Print the shape of the cleaned dataframe
    logger.info(f"Cleaned dataframe shape: {df.shape}")
    
    return df

def main():
    # Ensure required packages are installed
    try:
        import pyproj
    except ImportError:
        logger.error("pyproj package is required. Please install with: pip install pyproj")
        return
        
    # Read microplastic data
    mp_data = pd.read_csv(PATH.RAW / "Marine Microplastics WGS84.csv")
    mp_data['Date'] = pd.to_datetime(mp_data['Date'])
    
    results = []
    
    # Process each microplastic observation
    for idx, row in mp_data.iterrows():
        date = row['Date']
        lat = row['Latitude']
        lon = row['Longitude']
        
        logger.info(f"Processing observation {idx+1}/{len(mp_data)}: {date} at ({lat}, {lon})")
        
        # Find corresponding MODIS file
        modis_file = find_modis_file(date, lat, lon)
        
        if modis_file is None:
            logger.warning(f"Skipping observation - no MODIS data available")
            continue
            
        # Extract MODIS data at the exact location
        modis_data = extract_modis_data(modis_file, lat, lon)
        
        if modis_data:
            # Create a record combining MP and MODIS data
            result = {
                'date': date,
                'latitude': lat,
                'longitude': lon,
                'mp_concentration': row['Measurement'],
                'mp_unit': row['Unit'],
                'mp_sampling_method': row['Sampling Method'],
                'modis_file': modis_file.name,
            }
            
            # Add all MODIS bands to the result
            for band_name, band_value in modis_data.items():
                # Create clean column names for MODIS bands
                clean_name = f"modis_{band_name.replace(' ', '_').lower()}"
                result[clean_name] = band_value
                
            results.append(result)
        else:
            logger.warning(f"Could not extract MODIS data for observation")
    
    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)
        
        # Basic data cleaning
        df = clean_data(df)
        
        # Save the processed data
        output_file = PATH.PROCESSED / 'microplastics_modis_combined.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")
        
    else:
        logger.error("No data was processed!")

if __name__ == "__main__":
    main()