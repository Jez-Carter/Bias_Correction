import os
import matplotlib.pyplot as plt
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap
import numpy as np
from numpy import meshgrid
import iris
from iris.analysis.cartography import unrotate_pole 
from tinygp import kernels, GaussianProcess,transforms
import jax
jax.config.update("jax_enable_x64", True)

def standardise(data):
    return((data-data.mean())/data.std())

def unstandardise(newdata,refdata):
    return(newdata*refdata.std()+refdata.mean())

def grid_coords_to_2d_latlon_coords(ds,ref_file):
    ds_updated = ds.copy()
    rotated_grid_latitude = ds.grid_latitude.data
    rotated_grid_longitude = ds.grid_longitude.data
    rotated_grid_lons,rotated_grid_lats = meshgrid(rotated_grid_longitude, rotated_grid_latitude)
    cube = iris.load(ref_file)[0]
    cs = cube.coord_system()
    lons,lats = unrotate_pole(rotated_grid_lons,rotated_grid_lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    ds_updated = ds.assign_coords(
        latitude=(["grid_longitude","grid_latitude"], lats),
        longitude=(["grid_longitude","grid_latitude"], lons),
    )
    return (ds_updated)

def build_gp(idata,param):
    expectations = idata.posterior.mean(['chain','draw'])
    kernel = expectations[f'{param}_kern_var'].data * kernels.Exp(expectations[f'{param}_lengthscale'].data)
    X = np.vstack([idata.observed_data.grid_longitude_standardised.data,idata.observed_data.grid_latitude_standardised.data]).T
    return GaussianProcess(
        kernel,
        X,
        diag=expectations[f'{param}_like_var'].data+1e-5,
        mean=expectations[f'{param}_mean'].data,
    )


