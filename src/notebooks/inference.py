# %% Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import lightgbm as lgb
from pyhdf.SD import SD, SDC
import rasterio
from rasterio.transform import from_origin
import pyproj
from loguru import logger
from src.utils import PATH
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from cmr import GranuleQuery
from modis_tools.granule_handler import GranuleHandler
from modis_tools.auth import ModisSession
from modis_tools.models import Granule

# %% Constants
# Eastern North America coordinates (example: off the coast of Nova Scotia)
LAT = 44.0
LON = -63.0
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2021, 12, 31)

# Features used in training (must match the model)
FEATURES = [
    "modis_sur_refl_b01_1",
    "modis_sur_refl_b02_1",
    "modis_sur_refl_b03_1",
    "modis_sur_refl_b04_1",
    "modis_sur_refl_b05_1",
    "modis_sur_refl_b07_1",
]

# Add download-related constants
BUFFER = 0.1  # degrees
MAX_DOWNLOAD_SIZE = 1024 * 1024 * 1024  # 1GB in bytes

# %% Load model
model = lgb.Booster(model_file=str(PATH.WEIGHTS / "lgbm_model.txt"))

# %% Helper functions
def generate_dates():
    """Generate dates for the first of each month in the date range"""
    dates = []
    current = START_DATE
    while current <= END_DATE:
        dates.append(current)
        # Move to first of next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates

def download_modis_data(target_date):
    """Download MODIS data for a specific date and location, searching nearby dates if needed"""
    # Load environment variables and create session with credentials
    load_dotenv()
    session = ModisSession(
        username=os.getenv('EARTHDATA_USERNAME'),
        password=os.getenv('EARTHDATA_PASSWORD')
    )
    
    handler = GranuleHandler()
    
    # Create bounding box
    bbox = [
        LON - BUFFER,  # min longitude
        LAT - BUFFER,  # min latitude
        LON + BUFFER,  # max longitude
        LAT + BUFFER   # max latitude
    ]
    
    # Try up to 7 days (3 before and 3 after the target date)
    for day_offset in range(-3, 4):
        search_date = target_date + timedelta(days=day_offset)
        logger.info(f"Searching for data on {search_date} (offset: {day_offset} days)")
        
        # Query available granules
        api = GranuleQuery()
        granules = api.short_name("MYD09GA")\
                     .version("061")\
                     .temporal(search_date.strftime("%Y-%m-%d"), search_date.strftime("%Y-%m-%d"))\
                     .bounding_box(*bbox)\
                     .get()
        
        if not granules:
            logger.debug(f"No granules found for date {search_date}")
            continue
        
        logger.info(f"Found {len(granules)} granule(s) for {search_date}")
        
        granule_objects = [Granule.parse_obj(g) for g in granules]
        
        # Check for existing files
        granules_to_download = []
        for granule in granule_objects:
            try:
                filename = Path(granule.links[0].href).name
                file_path = PATH.RAW / filename
                if file_path.exists():
                    logger.info(f"File already exists: {filename}")
                    return file_path
                granules_to_download.append(granule)
            except (IndexError, AttributeError):
                granules_to_download.append(granule)
        
        if not granules_to_download:
            continue
        
        # Download missing files
        try:
            new_files = handler.download_from_granules(
                granules_to_download,
                modis_session=session,
                path=PATH.RAW,
                threads=1,
                force=False
            )
            
            if new_files:
                return Path(new_files[0])
            
        except FileNotFoundError as e:
            logger.error(f"Failed to download granules for {search_date}: {str(e)}")
            continue
    
    logger.warning(f"No data found within ±3 days of {target_date}")
    return None

def find_modis_file(date):
    """Find or download the appropriate MODIS file for a given date"""
    # First try to find existing file
    date_str = date.strftime('%Y%j')  # Year and day of year
    
    patterns = [
        f'MYD09GA.A{date_str}*.hdf',
        f'MOD09GA.A{date_str}*.hdf',
        f'*{date_str}*.hdf'
    ]
    
    for pattern in patterns:
        modis_files = list(PATH.RAW.glob(pattern))
        if modis_files:
            logger.debug(f"Found existing file matching pattern: {pattern}")
            return modis_files[0]
    
    # If no file found, try downloading
    logger.info(f"No existing file found for {date}, attempting download...")
    return download_modis_data(date)

def process_modis_tile(hdf_file):
    """Extract and process all relevant bands from a MODIS tile"""
    try:
        dataset = SD(str(hdf_file), SDC.READ)
        
        # Initialize array to store band data
        band_data = {}
        
        # Get all available datasets
        datasets_dict = dataset.datasets()
        logger.debug(f"Available datasets in HDF file: {datasets_dict}")
        
        # Process each dataset
        for ds_name, ds_info in datasets_dict.items():
            try:
                sds = dataset.select(ds_name)
                
                # Get attributes
                attrs = sds.attributes()
                scale_factor = 1.0
                add_offset = 0.0
                
                # Get scale factor if available
                if 'scale_factor' in attrs:
                    scale_factor = attrs['scale_factor']
                
                # Get offset if available
                if 'add_offset' in attrs:
                    add_offset = attrs['add_offset']
                
                # Get fill value if available
                fill_value = None
                fill_value_keys = ['_FillValue', 'fill_value', '_FILLVALUE', 'missing_value']
                for key in fill_value_keys:
                    if key in attrs:
                        fill_value = attrs[key]
                        break
                
                # Get the data
                data = sds.get()
                
                # Handle fill values
                if fill_value is not None:
                    data = np.where(data == fill_value, np.nan, data)
                
                # Apply scaling and offset
                data = data * scale_factor + add_offset
                
                # Create clean name for the band
                clean_name = f"modis_{ds_name.replace(' ', '_').lower()}"
                band_data[clean_name] = data
                
                logger.debug(f"Successfully processed dataset: {ds_name}")
                
            except Exception as e:
                logger.debug(f"Failed to process dataset {ds_name}: {e}")
                continue
        
        # Check if we have the required features
        missing_features = [f for f in FEATURES if f not in band_data]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            logger.error(f"Available features: {list(band_data.keys())}")
            return None
        
        # Only keep the features we need
        return {f: band_data[f] for f in FEATURES}
        
    except Exception as e:
        logger.error(f"Failed to process {hdf_file}: {e}")
        return None

def predict_tile(band_data):
    """Make predictions for entire tile"""
    # Get dimensions from the first band
    height, width = next(iter(band_data.values())).shape
    logger.debug(f"Input data dimensions: {height}x{width}")
    
    # Calculate total number of pixels
    n_pixels = height * width
    
    # Prepare data for prediction
    X = np.zeros((n_pixels, len(FEATURES)))
    
    for i, feature in enumerate(FEATURES):
        # Ensure the feature exists in band_data
        if feature not in band_data:
            raise ValueError(f"Required feature {feature} not found in band data")
        
        # Reshape the band data to a 1D array
        X[:, i] = band_data[feature].reshape(-1)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    
    # Make predictions
    logger.debug(f"Making predictions for {n_pixels} pixels")
    predictions = model.predict(X)
    
    # Reshape predictions to (n_pixels, n_classes)
    n_classes = predictions.size // n_pixels
    predictions = predictions.reshape(-1, n_classes)
    
    # Convert probabilities to class labels (0, 1, 2)
    class_labels = np.argmax(predictions, axis=1)
    
    logger.debug(f"Converted probabilities to {n_classes} classes")
    
    # Verify the reshape will work
    if class_labels.size != height * width:
        raise ValueError(f"Prediction size {class_labels.size} does not match input dimensions {height}x{width}")
    
    # Reshape back to tile dimensions
    return class_labels.reshape(height, width)

def save_prediction(prediction, date, output_file):
    """Save prediction as a GeoTIFF with proper georeferencing"""
    # Define MODIS sinusoidal projection
    modis_crs = pyproj.CRS('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext')
    
    # Calculate approximate pixel size (500m for MODIS)
    pixel_size = 500  # meters
    
    # Create transform (simplified - assumes regular grid)
    transform = from_origin(LON, LAT, pixel_size, pixel_size)
    
    # Save as GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=prediction.shape[0],
        width=prediction.shape[1],
        count=1,
        dtype=prediction.dtype,
        crs=modis_crs,
        transform=transform,
    ) as dst:
        dst.write(prediction, 1)
        
        # Add metadata
        dst.update_tags(
            date=date.strftime('%Y-%m-%d'),
            latitude=str(LAT),
            longitude=str(LON)
        )

# %% Main processing loop
def main():
    # Create output directory
    output_dir = PATH.PROCESSED / "predictions"
    output_dir.mkdir(exist_ok=True)
    
    # Generate dates
    dates = generate_dates()
    
    for date in dates:
        logger.info(f"Processing date: {date}")
        
        # Find MODIS file
        modis_file = find_modis_file(date)
        if modis_file is None:
            logger.warning(f"No MODIS file found for {date}")
            continue
            
        # Process tile
        band_data = process_modis_tile(modis_file)
        if band_data is None:
            continue
            
        # Make predictions
        prediction = predict_tile(band_data)
        
        # Save results
        output_file = output_dir / f"prediction_{date.strftime('%Y%m%d')}.tif"
        save_prediction(prediction, date, output_file)
        
        # Create quick visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(prediction, cmap='viridis')
        plt.colorbar(label='MP Concentration Category')
        plt.title(f'Predictions for {date.strftime("%Y-%m-%d")}')
        plt.savefig(output_dir / f"prediction_{date.strftime('%Y%m%d')}.png")
        plt.close()
        
        logger.info(f"Saved prediction to {output_file}")

if __name__ == "__main__":
    main()
