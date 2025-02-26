#%% Setup
import holoviews as hv
import pandas as pd
import numpy as np
from holoviews import opts
from src.utils import PATH

hv.extension('bokeh')

#%% Load data
df = pd.read_csv(PATH.PROCESSED / 'microplastics_modis_combined.csv')

#%% Inspect data



#%% Basic data cleaning and preparation
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Select relevant MODIS bands (surface reflectance bands)
modis_bands = [col for col in df.columns if 'modis_sur_refl_b' in col]

#%% Create scatter plots for each MODIS band vs microplastics concentration
scatter_plots = {}
for band in modis_bands:
    scatter = hv.Points(
        df, 
        [band, 'mp_concentration'],
        ['date']  # Include date as hover information
    ).opts(
        title=f'Microplastics vs {band}',
        width=400,
        height=400,
        tools=['hover'],
        xlabel=band,
        ylabel='Microplastics Concentration (pieces/m3)'
    )
    scatter_plots[band] = scatter

# Combine all scatter plots
scatter_layout = hv.Layout(list(scatter_plots.values())).cols(2)

#%% Create time series plot
time_series = hv.Points(
    df, 
    ['date', 'mp_concentration']
).opts(
    title='Microplastics Concentration Over Time',
    width=800,
    height=400,
    tools=['hover'],
    xlabel='Date',
    ylabel='Microplastics Concentration (pieces/m3)'
)

#%% Create correlation heatmap
correlation_data = df[['mp_concentration'] + modis_bands].corr()
heatmap = hv.HeatMap(
    correlation_data
).opts(
    title='Correlation Heatmap',
    width=600,
    height=600,
    tools=['hover'],
    xrotation=45,
    colorbar=True,
    cmap='RdBu_r'
)

#%% Display plots
# Combine all plots into a single layout for display
combined_layout = (scatter_layout + time_series + heatmap).cols(1)

# Display the combined layout
combined_layout

#%%
