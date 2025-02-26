import os
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
from modis_tools.granule_handler import GranuleHandler
from modis_tools.auth import ModisSession

from src.utils import PATH

def download_modis_data(dates, locations):
    """Download MODIS data using modis_tools"""
    # Load environment variables and create session with credentials
    load_dotenv()
    session = ModisSession(
        username=os.getenv('EARTHDATA_USERNAME'),
        password=os.getenv('EARTHDATA_PASSWORD')
    )
    
    # Initialize granule handler
    handler = GranuleHandler()
    
    # Search and download granules for each date and location
    for date in dates:
        logger.info(f"Processing date: {date}")
        
        bbox = [
            min(loc[1] for loc in locations),  # min longitude
            min(loc[0] for loc in locations),  # min latitude
            max(loc[1] for loc in locations),  # max longitude
            max(loc[0] for loc in locations)   # max latitude
        ]
        
        # Use CMR API to search for granules
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
            
        # Convert CMR results to Granule objects
        from modis_tools.models import Granule
        granule_objects = [Granule.parse_obj(g) for g in granules]
        
        # Download the granules
        handler.download_from_granules(
            granule_objects,
            modis_session=session,
            path=PATH.RAW,
            threads=4  # Optional: use multiple threads for downloading
        )

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
