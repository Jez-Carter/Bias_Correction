import iris
from iris.util import equalise_attributes
from iris.util import unify_time_units
import iris.coord_categorisation
from iris.analysis.cartography import unrotate_pole
from iris.coords import AuxCoord
from numpy import meshgrid

# import os

# def pcolormesh_basemapplot(cube,basemap,vmin,vmax,cmap=None):

#     longitudes = cube.coord('longitude').points
#     latitudes = cube.coord('latitude').points

#     current_dir = os.getcwd()
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('/data/climatedata')
#     basemap.readshapefile('antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
#     os.chdir(current_dir)
    
#     return(basemap.pcolormesh(longitudes,latitudes,cube.data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest',alpha=1))

def concatenate_cubes(cubelist):
    equalise_attributes(cubelist)
    unify_time_units(cubelist)
    return cubelist.concatenate_cube()

def add_2d_latlon_aux_coords(cube):
    rotated_grid_latitude = cube.coord('grid_latitude').points
    rotated_grid_longitude = cube.coord('grid_longitude').points
    lons,lats = meshgrid(rotated_grid_longitude, rotated_grid_latitude)
    cs = cube.coord_system()
    lons,lats = unrotate_pole(lons,lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    #lons,lats = rotate_pole(lons,lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    
    grid_lat_dim = cube.coord_dims('grid_latitude')[0]
    grid_lon_dim = cube.coord_dims('grid_longitude')[0]
    
    cube.add_aux_coord(AuxCoord(points=lats, standard_name='latitude', units='degrees'),(grid_lat_dim,grid_lon_dim))
    cube.add_aux_coord(AuxCoord(points=lons, standard_name='longitude', units='degrees'),(grid_lat_dim,grid_lon_dim))
    
def aggregate_to_daily(cube):
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')
    return cube.aggregated_by(['year','day_of_year'],iris.analysis.MEAN)

# def stl_decomposition(timeseries):
#     data = timeseries[:]
#     ds = pd.Series(data, index=pd.date_range('1-1-1979', periods=len(data), freq='M'), name = 'Melt')
#     stl = STL(ds, seasonal=13)
#     #stl = STL(ds, seasonal=13,robust=True)
#     res = stl.fit()
    
#     return ([res.trend,res.seasonal,res.resid]) 
