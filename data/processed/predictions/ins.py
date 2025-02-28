import rasterio
from pathlib import Path
from src.utils import PATH

# Get the predictions directory
predictions_dir = PATH.PROCESSED / "predictions"

# Get all prediction files
prediction_files = sorted(predictions_dir.glob("prediction_*.tif"))

if not prediction_files:
    print("No prediction files found!")
else:
    # Print info for each prediction file
    for file_path in prediction_files:
        print(f"\nAnalyzing: {file_path.name}")
        
        with rasterio.open(file_path) as src:
            # Print basic information
            print(f"Dimensions (width x height): {src.width} x {src.height}")
            print(f"Number of bands: {src.count}")
            print(f"Data type: {src.dtypes[0]}")
            
            # Print spatial information
            print(f"Coordinate system: {src.crs}")
            print(f"Bounds: {src.bounds}")
            
            # Print any available metadata tags
            if src.tags():
                print("\nMetadata tags:")
                for key, value in src.tags().items():
                    print(f"  {key}: {value}")
