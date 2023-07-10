'''
A script for creating an animation cycling through fits of the PDF from the numpyro model and the kde plots of the raw data. 
Useful url: https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
'''

import numpy as np
import arviz as az
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats

base_path = '/home/jez/Bias_Correction/'
idata_path = f'{base_path}data/Lima2021/Daily_Temperature_NFit.nc'
antarctica_shapefile_path = f'{base_path}data/Antarctica_Shapefile/antarctica_shapefile.shp'

idata = az.from_netcdf(idata_path)
expectations = idata.posterior.mean(['chain','draw'])
df = idata.observed_data.obs.to_dataframe().reset_index()
group = df.groupby('sites')['obs']
keys = group.groups.keys()
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

def animate(i):
    ax.cla()
    group.get_group(list(keys)[i]).plot.kde(ax=ax)
    mu = expectations['loc'][i]
    sigma = expectations['scale'][i]
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x,stats.norm.pdf(x, mu, sigma),color='g',linestyle='dashed')
    ax.set_xlim([-30,10])
    ax.set_ylim([0,0.2])
    ax.set_xlabel('Temperature (Degrees)')
    ax.set_ylabel('Probability Density')

fig,ax = plt.subplots(1,1,figsize=(12,5))
ani = animation.FuncAnimation(fig, animate, frames=56, interval=1,repeat=True)

ani.save('/home/jez/Bias_Correction/results/Lima_NST/PDF_Fit_Animation.mp4', writer=writer)