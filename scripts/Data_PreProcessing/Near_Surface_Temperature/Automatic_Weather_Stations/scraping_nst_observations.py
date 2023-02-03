import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tabula

base_path = '/home/jez/'
pdf_path = f'{base_path}Bias_Correction/data/NST_Station_Info.pdf'
outpath = f'{base_path}Bias_Correction/data/ProcessedData/NST_Observations.nc'

### scraping csvs from https://amrdcdata.ssec.wisc.edu/dataset/antaws-dataset

source = requests.get('https://amrdcdata.ssec.wisc.edu/dataset/antaws-dataset')
soup = BeautifulSoup(source.content)
div = soup.find_all('div', {'class':'dropdown btn-group'})

station_names = []
hrefs = []

for a in soup.find_all('a', href=True):
    if all(x in a['href'] for x in ['csv','day']):
        if any(href == a['href'] for href in hrefs):
            continue
        station_name = a['href'].split('/')[-1].split('_day.csv')[0]
        station_names.append(station_name)
        hrefs.append(a['href'])   
        
dfs = []
for station_name,href in zip(station_names,hrefs):
    df = pd.read_csv(f'{href}',encoding_errors='ignore')
    df['Station']=station_name
    dfs.append(df)
df_all = pd.concat(dfs)
df_all = df_all.reset_index(drop=True)

### scraping locations from supplementary table from paper (https://essd.copernicus.org/preprints/essd-2022-241/essd-2022-241-supplement.pdf)

locations_dfs = tabula.read_pdf(pdf_path, pages='6-10')
columns = locations_dfs[0].columns
for df in locations_dfs[1:]:
    df.loc[-1] = df.columns
    df.sort_index(inplace=True) 
    df.columns = columns
    
locations_df = pd.concat(locations_dfs)
locations_df.reset_index(drop=True)

### renaming some stations for merge to work

locations_df['Station_Lower'] = [x.lower() for x in locations_df['Station']]
locations_df['Station_Lower'] = locations_df['Station_Lower'].str.replace(' ','-')
locations_df['Station_Lower'] = locations_df['Station_Lower'].str.replace('(','')
locations_df['Station_Lower'] = locations_df['Station_Lower'].str.replace(')','')
locations_df['Station_Lower'] = locations_df['Station_Lower'].str.replace('!','')

locations_df['Station_Lower'] = locations_df['Station_Lower'].str.replace('marilyn-byrd-glacier','marilyn')
locations_df['Station_Lower'] = locations_df['Station_Lower'].replace('mt.erebus','mt.-erebus')
locations_df['Station_Lower'] = locations_df['Station_Lower'].replace('mt.fleming','mt.-fleming')
locations_df['Station_Lower'] = locations_df['Station_Lower'].replace('panda-south','panda_south')

### merging station data and location data

df_all_combined = locations_df.merge(df_all, how='outer', left_on='Station_Lower', right_on='Station')

### creating xarray dataset and adjusting coordinates

ds_all_combined = df_all_combined.set_index(['Station_Lower','Year','Month','Day']).to_xarray()

coords = ['Lat(°C)','Lon(°C)','Elevation(m)','Institution']
coords_df = df_all_combined.groupby('Station_Lower').first()[coords]
for coord in coords_df:
    ds_all_combined = ds_all_combined.drop(coord)
    coord_values = coords_df[coord].values
    coord_values[coord_values=='-']=np.nan
    ds_all_combined = ds_all_combined.assign_coords({coord:("Station_Lower", coord_values)})

### saving dataset
ds_all_combined['Temperature()'].to_netcdf(outpath)



# ### adjustments needed for saving to work
# ds_all_combined = ds_all_combined.rename({'Wind Speed(m/s)':'Wind Speed(mpers)'})
# for key in list(ds_all_combined.keys()):
#     Nans = ds_all_combined[key].data == '-'
#     ds_all_combined[key].data[Nans]=np.nan

# ### saving

# ds_all_combined.to_netcdf(outpath)


# ### reformatting pandas dataframe into xarray dataset 

# new_index = df_all_combined.groupby(['Station_Lower','Month']).cumcount()
# ds_all_combined = df_all_combined.set_index(['Station_Lower','Month',new_index]).to_xarray()

# ds_all_combined_filtered = ds_all_combined.copy()

# coords = ['Lat(°C)','Lon(°C)','Elevation(m)','Institution']
# extra_vars_to_drop = ['Station_x','Station_y','Relative Humidity(%)','No.','Pressure(hPa)','wt_P','wt_T','Wind Speed(m/s)','wt_WS','Wind Direction','wt_WD','wt_RH']
# ds_all_combined_filtered=ds_all_combined_filtered.drop(coords)
# ds_all_combined_filtered=ds_all_combined_filtered.drop(extra_vars_to_drop)

# for coord in coords:
#     ds_all_combined_filtered = ds_all_combined_filtered.assign_coords({f"{coord}": ds_all_combined.isel(level_2=0).drop(['level_2'])[f'{coord}']})
    
# ds_all_combined_filtered=ds_all_combined_filtered.rename({'Day':'Day_of_Month'})
# ds_all_combined_filtered=ds_all_combined_filtered.rename({'level_2':'Day'})

# Elevation_Nans = ds_all_combined_filtered['Elevation(m)'].data == '-'
# ds_all_combined_filtered['Elevation(m)'].data[Elevation_Nans]=np.nan

# ### saving reformatted dataset

# ds_all_combined_filtered.to_netcdf(outpath)