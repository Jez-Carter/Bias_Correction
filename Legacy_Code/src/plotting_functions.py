import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpyro.diagnostics import hpdi
from src.model_fitting_functions import BernoulliGamma

def bernoulli_gamma_pdf_plot(dictionary,months,sites,bounds):
    
    # ps = jnp.expand_dims(samples["p"], -1)[:, month, sites,0]
    # alphas = jnp.expand_dims(samples["alpha"], -1)[:, month, sites,0]
    # betas = jnp.expand_dims(samples["beta"], -1)[:, month, sites,0]
    ps = dictionary["p"][:,months][:,:,sites]
    alphas = dictionary["alpha"][:,months][:,:,sites]
    betas = dictionary["beta"][:,months][:,:,sites]
    
    dist = BernoulliGamma((ps,alphas,betas))

    xs = jnp.linspace(bounds[0],bounds[1],50)
    log_likelihoods = dist.log_prob(xs.reshape(1,len(xs),1,1,1))#vmap(dist.log_prob)(xs.reshape(50,1,1,1))
    likelihoods = jnp.exp(log_likelihoods)
    zero_log_likelihood = dist.log_prob(np.array([0]).reshape(1,1,1,1))
    zero_likelihood = jnp.exp(zero_log_likelihood)

    for m in np.arange(0,len(months),1):    
        for s in np.arange(0,len(sites),1): 
            ys = likelihoods[:,:,m,s]
            color = next(plt.gca()._get_lines.prop_cycler)["color"]
            plt.plot(xs, ys.mean(axis=1),color=color)
            ys_lower,ys_upper = hpdi(np.moveaxis(ys, 1, 0),0.95)
            plt.fill_between(xs, y1=ys_lower, y2=ys_upper, alpha=0.4,color=color)

            y0 = zero_likelihood[:,m,s]
            error = abs(hpdi(y0,0.95)-y0.mean()).reshape(2, 1)
            plt.errorbar(
                0,
                y0.mean(),
                yerr=error,
                capsize=3,
                color=color,
                label=f"month={m},site={sites[s]}",
            )
        
def histogram_plot(data,months,sites,bounds,bin_width):
    bins = np.arange(bounds[0], bounds[1], bin_width)
    bin_centers = bins + bin_width / 2
    for month in months:
        for site in sites:
            sitedata = data[:,month,site]
            p = len(sitedata[sitedata != 0]) / len(sitedata)
            weight_value = p / bin_width / len(sitedata[sitedata != 0])
            weights = np.full(len(sitedata[sitedata != 0]), weight_value)

            color = next(plt.gca()._get_lines.prop_cycler)["color"]
            plt.hist(
                    sitedata[sitedata != 0],
                    bins=bins,
                    histtype="step",
                    stacked=True,
                    fill=False,
                    weights=weights,
                    color=color,
                )

            plt.plot(0,1-p,marker='o',color=color)
            
def lima_alpha_beta_relationship_scatter(dictionary):
    a0 = dictionary["a0"].mean()
    a1 = dictionary["a1"].mean()
    betavar = dictionary["betavar"].mean()
    alphas = dictionary["alpha"].mean(axis=0).flatten()
    log_betas = np.log(dictionary["beta"]).mean(axis=0).flatten()
    
    plt.scatter(alphas,log_betas,marker="+")
    
    x = np.array([alphas.min(),alphas.max()])
    y = a0 + a1 * x
    plt.plot(x, y, label="Relationship based on a0 and a1")
    plt.fill_between(x,
    y1=y - betavar,
    y2=y + betavar,
    alpha=0.4)
    plt.xlabel("Alpha")
    plt.ylabel("Log-Beta")
    
def pcolormesh_basemapplot_cube(cube,basemap,vmin,vmax,cmap=None):

    longitudes = cube.coord('longitude').points
    latitudes = cube.coord('latitude').points
    
    basemap.readshapefile('/data/climatedata/antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    
    return(basemap.pcolormesh(longitudes,latitudes,cube.data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest',alpha=1))

def pcolormesh_basemapplot_data(data,latitudes,longitudes,basemap,vmin,vmax,cmap=None,label=None):
    
    basemap.readshapefile('/data/climatedata/antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    
    return(basemap.pcolormesh(longitudes,latitudes,data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest',alpha=1,label=label))
    
def pcolormesh_df(df,variable,basemap):
    shape = len(df.index.levels[0]),len(df.index.levels[1])
    latitude = df['latitude'].to_numpy().reshape(shape)
    longitude = df['longitude'].to_numpy().reshape(shape)
    data = df[variable].to_numpy().reshape(shape)
    vmin,vmax = data.min(),data.max()
    pcolormesh_basemapplot_data(data,latitude,longitude,basemap,vmin,vmax,cmap=None,label=f'{variable}')
    # plt.colorbar()
    
def plot_locations_df(df,basemap,color,label):
    basemap.readshapefile('/data/climatedata/antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    x, y = basemap(df['longitude'],df['latitude'])
    plt.plot(x,y, 'o', ms=1, markerfacecolor="None",markeredgecolor=color, markeredgewidth=0.1,label=label)

def plot_observations_df(df,basemap,variable,label):
    basemap.readshapefile('/data/climatedata/antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    x, y = basemap(df['longitude'],df['latitude'])
    c = df[f'{variable}']
    # plt.plot(x,y, 'o', ms=1, markerfacecolor="None",markeredgecolor=color, markeredgewidth=0.1,label=label)
    plt.scatter(x, y, c=c,label=label)#, s=500)
    
def pcolormesh_basemapplot(ds,var,metric,basemap,vmin,vmax,cmap=None,alpha=1):

    longitudes = ds.longitude.data
    latitudes = ds.latitude.data
    data = ds.sel(metric=metric)[f'{var}'].data
    # vmin = data.min()
    # vmax=data.max()
    
    shapefile_path = '/data/notebooks/jupyterlab-biascorrlab/data/Antarctica_Shapefile/antarctica_shapefile'
    basemap.readshapefile(shapefile_path, 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    
    return(basemap.pcolormesh(longitudes,latitudes,data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest',alpha=alpha,linewidth=0.05))

def scatter_basemapplot(ds,var,basemap,vmin,vmax,cmap=None,alpha=1):

    longitudes = ds.longitude.data
    latitudes = ds.latitude.data
    x, y = basemap(longitudes, latitudes)
    
    data = ds[f'{var}'].data
    # vmin = data.min()
    # vmax=data.max()
    
    shapefile_path = '/data/notebooks/jupyterlab-biascorrlab/data/Antarctica_Shapefile/antarctica_shapefile'
    basemap.readshapefile(shapefile_path, 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    
    return(basemap.scatter(x, y, 10, marker='o', c=data, vmin=vmin,vmax=vmax, cmap=cmap,alpha=alpha, zorder=3))


    
# from shapely.geometry import Point
# import geopandas as gpd

# crs = {'init':'EPSG:4326'}
# geometry = [Point(xy) for xy in zip(observations_df['longitude'], observations_df['latitude'])]
# geo_df = gpd.GeoDataFrame(observations_df, crs = crs, geometry = geometry)
# antarctica_map = gpd.read_file('/data/climatedata/antarctica_shapefile.shp')

# fig, ax = plt.subplots(figsize = (10,10))
# antarctica_map.to_crs(epsg=3031).plot(ax=ax, color='lightgrey')
# geo_df.to_crs(epsg=3031).plot(ax=ax)
# bounds = geo_df.to_crs(epsg=3031).geometry.total_bounds
# xmin, ymin, xmax, ymax = bounds
# xmin,xmax = -3000000,-900000
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)

# def empirical_pdf_plot(data, months, sites, bins):
#     ps = np.count_nonzero(data, axis=0) / len(data[:, 0, 0])  # shape (months,sites)
#     bin_width = bins[1] - bins[0]
#     weight_value = 1 / (
#         bin_width * len(jdata[:, 0, 0])
#     )  # converts histogram to pdf, shape ()

#     for k in months:
#         for j in sites:
#             single_site_month_data = jdata[:, k, j]
#             single_site_month_data = single_site_month_data[single_site_month_data != 0]
#             weights = np.full(
#                 len(single_site_month_data), weight_value
#             )  # shape (samples,)
#             # color = next(plt.gca()._get_lines.prop_cycler)["color"]
#             plt.hist(
#                 single_site_month_data,
#                 bins=bins,
#                 histtype="step",
#                 stacked=True,
#                 fill=False,
#                 weights=weights,
#                 # color=color,
#             )
            
# def mcmc_pdf_plot(samples, bin_centers, months, sites, alpha_adjustment, scale_adjustment):
#     ps = jnp.expand_dims(samples[f"p"], -1)  # shape (samples,months,sites,1)
#     alphas = (
#         jnp.expand_dims(samples[f"alpha"], -1) + alpha_adjustment
#     )  # shape (samples,months,sites,1)
#     scales = jnp.reciprocal(
#         jnp.expand_dims(samples[f"beta"], -1) + scale_adjustment
#     )  # shape (samples,months,sites,1)

#     likelihood_values = ps * jgamma.pdf(
#         bin_centers, a=alphas, loc=0, scale=scales
#     )  # shape (samples,months,sites,estimates)

#     zero_likelihood_values = 1 - ps  # shape (samples,months,sites,1)

#     mean_l = jnp.mean(
#         jnp.array(likelihood_values), axis=0
#     )  # shape (months,sites,estimates)
#     hpdi_l = hpdi(
#         jnp.array(likelihood_values), 0.95
#     )  # shape (2,months,sites,estimates)
#     lower_l = hpdi_l[0]  # shape (months,sites,estimates)
#     upper_l = hpdi_l[1]  # shape (months,sites,estimates)

#     zero_mean_l = jnp.mean(
#         jnp.array(zero_likelihood_values), axis=0
#     )  # shape (months,sites,1)
#     zero_hpdi_l = hpdi(
#         jnp.array(zero_likelihood_values), 0.95
#     )  # shape (2,months,sites,1)
#     zero_lower_l = zero_hpdi_l[0]  # shape (months,sites,1)
#     zero_upper_l = zero_hpdi_l[1]  # shape (months,sites,1)
#     zero_lower_error = zero_mean_l - zero_lower_l  # shape (months,sites,1)
#     zero_upper_error = zero_upper_l - zero_mean_l  # shape (months,sites,1)
#     error = np.array([zero_lower_error, zero_upper_error])  # shape (2,months,sites,1)

#     for k in months:
#         for j in sites:
#             color = next(plt.gca()._get_lines.prop_cycler)["color"]

#             plt.plot(bin_centers, mean_l[k, j], color=color)
#             plt.fill_between(
#                 bin_centers, y1=lower_l[k, j], y2=upper_l[k, j], alpha=0.4, color=color
#             )

#             plt.plot(0, 1 - ps[:, k, j].mean(), marker="o", color=color)
#             plt.errorbar(
#                 0,
#                 zero_mean_l[k, j],
#                 yerr=error[:, k, j],
#                 capsize=3,
#                 color=color,
#                 label=f"month={k},site={j}",
#             )
            