import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from datetime import datetime, timezone

from src.utils import PATH

load_dotenv()

def get_token():
    """Get authentication token from AppEEARS API"""
    username = os.getenv('EARTHDATA_USERNAME')
    password = os.getenv('EARTHDATA_PASSWORD')
    
    response = requests.post(
        'https://appeears.earthdatacloud.nasa.gov/api/login',
        auth=(username, password)
    )
    
    if response.status_code == 200:
        token_data = response.json()
        # Check if token is expired
        expiration = datetime.fromisoformat(token_data['expiration'].replace('Z', '+00:00'))
        if expiration > datetime.now(timezone.utc):
            logger.info("Successfully obtained API token")
            return token_data['token']
    
    logger.error(f"Failed to get token: {response.status_code} {response.text}")
    return None

def submit_appeears_request(dates, locations, token):
    """Submit a request to AppEEARS API for MODIS data"""
    headers = {'Authorization': f'Bearer {token}'}

    response = requests.get('https://appeears.earthdatacloud.nasa.gov/api/product')
    product_response = response.json()
    # create a dictionary indexed by the product name and version
    products = {p['ProductAndVersion']: p for p in product_response}
    logger.debug(products['MYD09GA.061'])
    
    # Construct the task request
    task = {
        "task_type": "point",
        "task_name": "MODIS_Download",
        "params": {
            "dates": [
                {
                    "startDate": date.strftime('%m-%d-%Y'),
                    "endDate": date.strftime('%m-%d-%Y')
                }
                for date in dates
            ],
            "layers": [
                {"product": "MYD09GA.061", "layer": "sur_refl_b01_1"},  # Red (620-670 nm)
                {"product": "MYD09GA.061", "layer": "sur_refl_b02_1"},  # NIR (841-876 nm)
                {"product": "MYD09GA.061", "layer": "sur_refl_b03_1"},  # Blue (459-479 nm)
                {"product": "MYD09GA.061", "layer": "sur_refl_b06_1"},  # SWIR 1 (1628-1652 nm)
                {"product": "MYD09GA.061", "layer": "sur_refl_b07_1"}   # SWIR 2 (2105-2155 nm)
            ],
            "coordinates": [
                {
                    "id": f"point_{i}",
                    "latitude": lat,
                    "longitude": lon
                }
                for i, (lat, lon) in enumerate(locations)
            ]
        }
    }
    
    # Submit the task
    response = requests.post(
        'https://appeears.earthdatacloud.nasa.gov/api/task',
        headers=headers,
        json=task
    )
    
    if response.status_code == 202:
        task_id = response.json()['task_id']
        logger.info(f"Task submitted successfully. Task ID: {task_id}")
        return task_id
    else:
        logger.error(f"Failed to submit task: {response.status_code} {response.text}")
        return None

def check_task_status(task_id, token):
    """Check the status of an AppEEARS task"""
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(
        f'https://appeears.earthdatacloud.nasa.gov/api/task/{task_id}',
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()['status']
    return None

def download_task_results(task_id, token):
    """Download the results of a completed AppEEARS task"""
    headers = {'Authorization': f'Bearer {token}'}
    
    # Add retries for getting bundle info
    max_retries = 3
    retry_delay = 30  # seconds
    
    for attempt in range(max_retries):
        # Get available files
        response = requests.get(
            f'https://appeears.earthdatacloud.nasa.gov/api/task/{task_id}/bundle',
            headers=headers
        )
        
        if response.status_code == 200:
            break
        elif attempt < max_retries - 1:
            logger.warning(f"Attempt {attempt + 1} failed to get bundle info. Waiting {retry_delay} seconds...")
            time.sleep(retry_delay)
            continue
        else:
            logger.error(f"Failed to get bundle info after {max_retries} attempts: {response.status_code} {response.text}")
            return
    
    # Rest of the download function remains the same
    for file_info in response.json()['files']:
        filename = PATH.RAW / file_info['file_name']
        
        if not filename.exists():
            logger.info(f"Downloading {filename}...")
            download_url = f'https://appeears.earthdatacloud.nasa.gov/api/bundle/{task_id}/{file_info["file_id"]}'
            
            with requests.get(download_url, headers=headers, stream=True) as r:
                total_size = int(r.headers.get('content-length', 0))
                with open(filename, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

def main():
    # Get authentication token
    token = get_token()
    if not token:
        return
    
    # Read MP data
    mp_data = pd.read_csv(PATH.RAW / "Marine Microplastics WGS84.csv")
    mp_data['Date'] = pd.to_datetime(mp_data['Date'])
    
    # Get unique combinations of dates and locations
    unique_observations = mp_data[['Date', 'Latitude', 'Longitude']].drop_duplicates()
    
    # Process in smaller batches
    batch_size = 10  # Adjust this number based on your needs
    for i in range(0, len(unique_observations), batch_size):
        batch = unique_observations.iloc[i:i+batch_size]
        
        # Prepare data for AppEEARS request
        dates = batch['Date'].tolist()
        locations = batch[['Latitude', 'Longitude']].values.tolist()
        
        logger.info(f"Processing batch {i//batch_size + 1} with {len(dates)} observations")
        
        # Submit request
        task_id = submit_appeears_request(dates, locations, token)
        if task_id:
            # Wait for task completion
            start_time = time.time()
            while True:
                status = check_task_status(task_id, token)
                elapsed_minutes = (time.time() - start_time) / 60
                logger.info(f"Task status after {elapsed_minutes:.0f} minutes: {status}")
                
                if status == 'done':
                    logger.info("Task completed successfully")
                    logger.info("Waiting 30 seconds for results to be ready...")
                    time.sleep(30)  # Add delay before downloading
                    download_task_results(task_id, token)
                    break
                elif status == 'error':
                    logger.error("Task failed")
                    break
                elif status == 'running':
                    logger.info("Task is still processing...")
                # elif status == 'pending':
                #     logger.info("Task is in queue...")
                
                time.sleep(60)

if __name__ == "__main__":
    main()
