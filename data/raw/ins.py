from pathlib import Path
import sys
from pyhdf.SD import SD, SDC

HERE = Path(__file__).parent

data_path = HERE / 'MYD09GA.A2006290.h11v07.061.2020274205514.hdf'

try:
    # Open the HDF-EOS file
    hdf = SD(str(data_path), SDC.READ)
    
    # Print datasets in the file
    print("Datasets in the file:")
    datasets = hdf.datasets()
    for idx, sds in enumerate(datasets.keys()):
        print(f"- {sds}")
        # Get dataset object
        sds_obj = hdf.select(sds)
        # Get dataset info
        print(f"  Shape: {sds_obj.info()[2]}")
        print(f"  Type: {sds_obj.info()[3]}")
        
        # Print attributes if any
        attrs = sds_obj.attributes()
        if attrs:
            print("  Attributes:")
            for attr_name, attr_value in attrs.items():
                print(f"    {attr_name}: {attr_value}")
        
    hdf.end()

except Exception as e:
    print(f"Error opening HDF-EOS file: {e}")
    sys.exit(1)
