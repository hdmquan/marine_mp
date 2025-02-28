import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from pathlib import Path
import imageio
from datetime import datetime
from loguru import logger
import cv2
from src.utils import PATH

# Constants (matching inference.py)
LAT = 44.0
LON = -63.0
BUFFER = 0.1
OUTPUT_SIZE = (800, 800)  # Reduced size for the GIF

def load_prediction(file_path):
    """Load a prediction TIFF file and its metadata"""
    with rasterio.open(file_path) as src:
        # Read the prediction data
        prediction = src.read(1)
        
        # Replace any NaN values with 0
        prediction = np.nan_to_num(prediction, nan=0)
        
        # Ensure the array is in uint8 format for OpenCV
        prediction = prediction.astype(np.uint8)
        
        # Get the date from metadata
        date_str = src.tags().get('date')
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        return prediction, date

def create_map_frame(prediction, date, extent):
    """Create a single frame showing the prediction data with coastlines"""
    # Create figure and axis with Cartopy projection
    fig = plt.figure(figsize=(10, 10), dpi=200)  # Set consistent DPI
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add coastlines and other features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Plot the prediction data
    im = ax.imshow(
        prediction,
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0,
        vmax=2
    )
    
    # Add gridlines
    ax.gridlines(draw_labels=True)
    
    # Add title
    plt.title(f'Predictions for {date.strftime("%Y-%m-%d")}')
    
    # Convert plot to image array
    plt.tight_layout()
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    
    plt.close()
    
    return frame, im

def main():
    # Set up paths
    predictions_dir = PATH.PROCESSED / "predictions"
    output_file = predictions_dir / "prediction_animation.gif"
    
    # Get all prediction files
    prediction_files = sorted(predictions_dir.glob("prediction_*.tif"))
    
    if not prediction_files:
        logger.error("No prediction files found!")
        return
    
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    # Calculate map extent
    extent = [
        LON - BUFFER,  # min longitude
        LON + BUFFER,  # max longitude
        LAT - BUFFER,  # min latitude
        LAT + BUFFER   # max latitude
    ]
    
    # Create a figure for the colorbar with matching width
    colorbar_fig = plt.figure(figsize=(10, 1), dpi=200)  # Match figsize and DPI with main plot
    ax = colorbar_fig.add_axes([0.05, 0.5, 0.9, 0.15])
    colorbar = plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(0, 2),
            cmap='viridis'
        ),
        cax=ax,
        orientation='horizontal',
        label='MP Concentration Category'
    )
    colorbar_fig.canvas.draw()
    colorbar_image = np.array(colorbar_fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(colorbar_fig)

    # Create frames
    frames = []
    for file_path in prediction_files:
        logger.info(f"Processing {file_path.name}")
        
        try:
            prediction, date = load_prediction(file_path)
            frame, _ = create_map_frame(prediction, date, extent)
            
            # Resize colorbar image to match frame width
            colorbar_resized = cv2.resize(colorbar_image, (frame.shape[1], colorbar_image.shape[0]))
            
            # Combine frame with colorbar
            combined_frame = np.vstack([frame, colorbar_resized])
            frames.append(combined_frame)
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    if not frames:
        logger.error("No frames were created!")
        return
    
    # Save as GIF
    logger.info("Creating GIF...")
    imageio.mimsave(
        output_file,
        frames,
        fps=1,  # 1 frame per second
        loop=0   # loop forever
    )
    logger.info(f"Animation saved to {output_file}")

if __name__ == "__main__":
    main() 