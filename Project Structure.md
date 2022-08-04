Initial aim is to replicate the methodology used in the Lima 2021 paper. The scripts for this can be found in /scripts/Lima2021/. The scripts consist of:

- loading_aggregating_raw_data.py : loads 6-Hourly climate model data for snowfall over Antarctica and aggregates to daily.
- simulating_observations.py : takes daily snowfall data from the climate model and samples to particular sites, which we treat as 'observations'.
- bg_lima.py : takes daily 'observational' snowfall data and uses numpyro to fit a Bernoulli-Gamma distribution to the data using MCMC. 
- gp_lima.py : takes estimates of the Bernoulli-Gamma parameters alpha and p at all 'observation' locations and fits a Gaussian Process to the spatial distribution of alpha and p seperately. 
- qm_lima.py : takes estimates of the Bernoulli-Gamma parameters alpha and p at all locations across our region of interest (Antarctic Peninsula) as well as estimates in a0,a1 and betavar, which are used to generate estimates of beta. These estimates are derived from the 'observations' and are used for applying a Quantile Mapping (AM) correction to the time series of daily rainfall from the climate model at every site across the domain.

The secondary aim is to combine the Bernoulli-Gamma and Gaussian Process parts into a single bayesian model. The scripts for this include:

- bg_gp.py : takes daily 'observational' snowfall data and uses numpyro to fit a hierarchial MVN - Bernoulli-Gamma model to the data using MCMC. 

