#Loading daily snowfall data and fitting a Bernoulli-Gamma distribution using MCMC
import timeit
import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit
import iris
import iris.coord_categorisation
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, BarkerMH
from src.model_fitting_functions import lima_model 
from src.model_fitting_functions import BernoulliGamma

month_numbers = [1,6] #Jan,Feb,March
data_directory = '/data/climatedata/'
metum_cube = iris.load(f'{data_directory}metum_cube_lres.nc')[0]
iris.coord_categorisation.add_month_number(metum_cube, metum_cube.coord('time'), name='month_number')

metum_cube = metum_cube[:,50:60,15:25] # (time,grid_lat,grid_lon) #[:,40:70,5:40] selects Antarctic Peninsula

combine_lat_lon_size = metum_cube.shape[1]*metum_cube.shape[2]

# Loading data for each month into list
data = []
for month_number in month_numbers:
    month_cube = metum_cube[metum_cube.coord('month_number').points==month_number]
    month_data = month_cube.data.data #shape [#times,#lats,#lons]
    month_data = month_data.reshape(month_cube.shape[0],combine_lat_lon_size) # reshaping to [#times,#lats*#lons]
    month_data = month_data*10**5 # Note I found that for some reason the fitting procedure struggles with very small values of snowfall, so I multiply by 10^5
    data.append(month_data)
    
# Converting to JAX array (needed for Numpyro)
jdata = jnp.array([i[:1000] for i in data]) # JAX doesn't like object arrays so data for each month needs to be the same shape
jdata = np.moveaxis(jdata, 1, 0) # adjusting the axes so that it's [days,months,sites]

np.save('/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall.npy',jdata)

# Numpyro syntax for running MCMC
starttime = timeit.default_timer()

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(lima_model)
#kernel = HMC(lima_model)
#kernel = BarkerMH(lima_model)
num_samples = 2000
num_warmup = 1000
mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    #    chain_method="vectorized",
    num_chains=1,
)
mcmc.run(rng_key_, jdata)
mcmc.print_summary()
samples = mcmc.get_samples()

print("Time Taken:", timeit.default_timer() - starttime)

samples_array = np.array(list(samples.items()),dtype=object)
np.save('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_MCMC_samples.npy',samples_array)
    
