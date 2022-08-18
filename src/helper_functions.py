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
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

def standardise(data,refdata=None):
    if refdata is None:
        standardised_data = (data-data.mean())/data.std()
    else:
        standardised_data = (data-data.mean())/data.std()
    return(standardised_data)

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

def empirical_cdf(data):
    ecdf = ECDF(data)
    return(ecdf(data))

def quantile_mapping(prsn,ecdf,diff_p,p,alpha,beta):
    positive_diff_p = diff_p>0
    negative_diff_p = diff_p<0
    non_zero = prsn > 0
    lower_ecdf_than_min_pred = ecdf < (1-p)
    greater_ecdf_than_min_pred = ecdf > (1-p)
    
    gamma = stats.gamma(a=alpha,loc=0, scale=1/beta)
    corrected_prsn_non_zeros = gamma.ppf((ecdf-(1-p))/p) /10**5
    
    corrected_prsn = np.where(positive_diff_p & non_zero & greater_ecdf_than_min_pred,corrected_prsn_non_zeros,prsn)
    corrected_prsn = np.where(positive_diff_p & non_zero & lower_ecdf_than_min_pred,0,corrected_prsn)

    zero_indicies = np.argwhere(np.array(prsn.data) == 0)
    index_in_random_selection = prsn != prsn

    if np.array(negative_diff_p.data)==True:
        number_to_convert = int(len(prsn)*np.abs(diff_p))
        random_selection_zero_indicies = np.random.choice(zero_indicies.ravel(), size=number_to_convert, replace=False, p=None)
        index_in_random_selection[random_selection_zero_indicies] = True

    new_ecdf = np.linspace(1-p,ecdf.min(),len(prsn))
    np.random.shuffle(new_ecdf)
    adjusted_ecdf = ecdf.copy()
    adjusted_ecdf.data = np.where(index_in_random_selection,new_ecdf,adjusted_ecdf)

    corrected_prsn_neg_diff = gamma.ppf((adjusted_ecdf-(1-p))/p) /10**5

    corrected_prsn = np.where(negative_diff_p & non_zero,corrected_prsn_neg_diff,corrected_prsn)
    corrected_prsn = np.where(negative_diff_p & index_in_random_selection,corrected_prsn_neg_diff,corrected_prsn)
    
    return(corrected_prsn)

