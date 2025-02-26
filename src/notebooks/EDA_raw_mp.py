#%% Imports
import holoviews as hv
import geoviews as gv
import panel as pn
import numpy as np
from holoviews.element.tiles import EsriImagery
import pandas as pd

from src.utils import PATH

# Enable notebook extension
hv.extension('bokeh')

#%% Load and inspect data
mp_data = pd.read_csv(PATH.RAW / "Marine Microplastics WGS84.csv")

# Display basic information about the dataset
print("Dataset Info:")
print(mp_data.info())

print("\nSummary Statistics:")
print(mp_data.describe())

#%% Data preprocessing
# Convert date column to datetime
mp_data['Date'] = pd.to_datetime(mp_data['Date'])

# Create year column for temporal analysis
mp_data['Year'] = mp_data['Date'].dt.year

#%% Create base map
tiles = gv.tile_sources.EsriImagery().opts(
    width=800, 
    height=600,
    bgcolor='lightgray',
)

#%% Create interactive visualization
# Create widgets for filtering and coloring
color_dim = pn.widgets.Select(
    name='Color by', 
    options=['Measurement', 'Density Range', 'Density Class', 'Year', 'Oceans'],
    value='Measurement'  # Set a default value
)

ocean_filter = pn.widgets.MultiChoice(
    name='Filter by Ocean',
    options=list(mp_data['Oceans'].dropna().unique()),  # Handle potential NaN values
    value=[]
)

year_range = pn.widgets.RangeSlider(
    name='Year Range',
    start=int(mp_data['Year'].min()),  # Convert to int to avoid floating point issues
    end=int(mp_data['Year'].max()),
    value=(int(mp_data['Year'].min()), int(mp_data['Year'].max())),
    step=1
)

@pn.depends(color_dim.param.value, ocean_filter.param.value, year_range.param.value)
def get_plot(color, selected_oceans, year_range):
    # Filter data based on selections
    filtered_data = mp_data.copy()
    
    if selected_oceans:
        filtered_data = filtered_data[filtered_data['Oceans'].isin(selected_oceans)]
    
    filtered_data = filtered_data[
        (filtered_data['Year'] >= year_range[0]) & 
        (filtered_data['Year'] <= year_range[1])
    ]
    
    # Create points
    points = gv.Points(
        filtered_data, 
        kdims=['Longitude', 'Latitude'], 
        vdims=['Measurement', 'Oceans', 'Sampling Method', 'Year', 'Density Class']
    ).opts(
        color=color,  # Use the color parameter directly
        size=8,
        tools=['hover'],
        colorbar=True,
        cmap='viridis',
        title='Marine Microplastics Distribution',
        xlabel='Longitude',
        ylabel='Latitude'
    )
    
    return tiles * points

#%% Create dashboard
dashboard = pn.Column(
    pn.Row(
        pn.Column(
            "## Marine Microplastics Explorer",
            color_dim,
            ocean_filter,
            year_range,
        ),
        width=300
    ),
    get_plot  # Use the function directly instead of pn.bind
)

#%% Additional visualizations
# Distribution of measurements by ocean
measurement_by_ocean = hv.BoxWhisker(
    mp_data, 
    'Oceans', 
    'Measurement'
).opts(
    width=800,
    height=400,
    title='Distribution of Microplastic Measurements by Ocean',
    tools=['hover'],
    box_fill_color='lightblue',
    xrotation=45
)

# Time series of measurements
time_series = hv.Points(
    mp_data, 
    ['Date', 'Measurement']
).opts(
    width=800,
    height=400,
    title='Microplastic Measurements Over Time',
    tools=['hover'],
    color='Oceans',
    size=8
)

#%% Display all visualizations
pn.Column(
    dashboard,
    measurement_by_ocean,
    time_series
)
