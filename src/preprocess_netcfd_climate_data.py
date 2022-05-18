#Loading 6hourly snowfall data and aggregating to daily as well as adding lat,lon coordinates

years = np.arange(1981,2019,1)
cubes = []
grid_cell = 50,50

for year in years:
    print(year)
    url = f"http://192.171.173.134/thredds/fileServer/dsnefiles/Jez/MetUM_Data/CORDEX_044_6Hourly/Antarctic_CORDEX_MetUM_0p44deg_6_hourly_mean_prsn_{year}.nc#mode=bytes"
    metum_cube = iris.load(url)[0]
    metum_cube = aggregate_to_daily(metum_cube)  # Note after aggregating this cube has shape 365,98,126 
    add_2d_latlon_aux_coords(metum_cube)
    cubes.append(metum_cube)

metum_cube = concatenate_cubes(iris.cube.CubeList(cubes))
data_directory = '/data/climatedata/'
iris.save(metum_cube,f'{data_directory}metum_cube_lres.nc')