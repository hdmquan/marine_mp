import os
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
from modis_tools.granule_handler import GranuleHandler
from modis_tools.auth import ModisSession
from pathlib import Path

from src.utils import PATH

def download_modis_data(dates, locations):
    """
    Download MODIS data using modis_tools for a point location
    locations: List of (lat, lon) tuples
    """
    # Load environment variables and create session with credentials
    load_dotenv()
    session = ModisSession(
        username=os.getenv('EARTHDATA_USERNAME'),
        password=os.getenv('EARTHDATA_PASSWORD')
    )
    
    handler = GranuleHandler()
    
    # For a point location, we'll add a small buffer (0.1 degrees ~ 11km)
    # This ensures we get the correct tile while keeping download size minimal
    if len(locations) > 1:
        logger.warning("Multiple locations provided. Using first point only.")
    
    lat, lon = locations[0]  # Take first point
    buffer = 0.1  # degrees
    bbox = [
        lon - buffer,  # min longitude
        lat - buffer,  # min latitude
        lon + buffer,  # max longitude
        lat + buffer   # max latitude
    ]
    
    logger.info(f"Downloading MODIS data for point ({lat}, {lon}) with {buffer}° buffer")
    
    downloaded_size = 0  # Track total download size
    max_size = 1024 * 1024 * 1024  # 1GB in bytes
    
    for date in dates:
        # Check if we're approaching size limit
        if downloaded_size > max_size:
            logger.warning(f"Reached size limit of 1GB. Stopping downloads.")
            break
            
        logger.info(f"Processing date: {date}")
        
        from cmr import GranuleQuery
        api = GranuleQuery()
        granules = api.short_name("MYD09GA")\
                     .version("061")\
                     .temporal(date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))\
                     .bounding_box(*bbox)\
                     .get()
        
        if not granules:
            logger.warning(f"No granules found for date {date}")
            continue
        
        logger.info(f"Found {len(granules)} granule(s) for {date}")
        
        from modis_tools.models import Granule
        granule_objects = [Granule.parse_obj(g) for g in granules]
        
        # Check for existing files and their sizes
        granules_to_download = []
        for granule in granule_objects:
            try:
                filename = Path(granule.links[0].href).name
                file_path = Path(PATH.RAW) / filename
                if file_path.exists():
                    downloaded_size += file_path.stat().st_size
                    logger.info(f"File already exists: {filename}")
                    continue
                granules_to_download.append(granule)
            except (IndexError, AttributeError):
                granules_to_download.append(granule)
        
        if not granules_to_download:
            logger.info(f"All files for {date} already downloaded")
            continue
        
        # Download only missing files
        try:
            new_files = handler.download_from_granules(
                granules_to_download,
                modis_session=session,
                path=PATH.RAW,
                threads=1,  # Single thread for better control
                force=False
            )
            
            # Update size tracking
            for file_path in new_files:
                file_size = Path(file_path).stat().st_size
                downloaded_size += file_size
                logger.info(f"Downloaded {file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
                
            logger.info(f"Total size so far: {downloaded_size / 1024 / 1024:.1f} MB")
            
        except FileNotFoundError as e:
            logger.error(f"Failed to download granules for {date}: {str(e)}")
            continue

def main():
    # Read MP data
    mp_data = pd.read_csv(PATH.RAW / "Marine Microplastics WGS84.csv")
    mp_data['Date'] = pd.to_datetime(mp_data['Date'])
    
    # Get unique combinations of dates and locations
    unique_observations = mp_data[['Date', 'Latitude', 'Longitude']].drop_duplicates()
    
    # Process in smaller batches
    batch_size = 10  # Adjust this number based on your needs
    for i in range(0, len(unique_observations), batch_size):
        batch = unique_observations.iloc[i:i+batch_size]
        
        # Prepare data
        dates = batch['Date'].tolist()
        locations = batch[['Latitude', 'Longitude']].values.tolist()
        
        logger.info(f"Processing batch {i//batch_size + 1} with {len(dates)} observations")
        download_modis_data(dates, locations)

if __name__ == "__main__":
    main()
