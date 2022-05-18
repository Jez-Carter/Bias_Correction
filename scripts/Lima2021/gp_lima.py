#Loading alpha values for daily snowfall data and fitting a Gaussian Process distribution using MCMC
import numpy as np
import pandas as pd
import iris
import gpflow 
from gpflow.utilities import print_summary
from gpflow.utilities import read_values
from gpflow.ci_utils import ci_niter
from gpflow.kernels import Matern32
from gpflow.models import SGPR
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from scipy.cluster.vq import kmeans2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.model_fitting_functions import train_gp
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys

folder_path = sys.argv[1]
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
f64 = gpflow.utilities.to_default_float # convert to float64 for tfp to play nicely with gpflow in 64
tf.random.set_seed(123)

#reading in data and standardising
df = pd.read_csv(f'{folder_path}AP_BGLima_Snowfall_044_Mean_Estimates.csv', index_col=0)
df = df[df['month']==1]
df_stand = df.copy()
for i in ['p','alpha','grid_latitude','grid_longitude']:
    scale =  StandardScaler().fit(df[[i]])
    df_stand[f'{i}_stand'] = scale.transform(df[[i]])
df_stand.to_csv(f'{folder_path}AP_BGLima_Snowfall_044_Mean_Alpha_stand.csv',index=True)

#initialising model
alpha_kern = gpflow.kernels.Exponential(lengthscales=[1],variance=[0.5])
alpha_model = gpflow.models.GPR(data=(df_stand[['grid_latitude_stand','grid_longitude_stand']], df_stand[['alpha_stand']]), kernel=alpha_kern, mean_function=None, noise_variance=0.5)
p_kern = gpflow.kernels.Exponential(lengthscales=[1],variance=[0.5])
p_model = gpflow.models.GPR(data=(df_stand[['grid_latitude_stand','grid_longitude_stand']], df_stand[['p']]), kernel=p_kern, mean_function=None, noise_variance=0.5)
print('Model Initialisation before MLE')
print('Alpha')
print_summary(alpha_model)
print('P')
print_summary(p_model)

#maximum likelihood estimation of parameters
opt = tf.optimizers.Adam(learning_rate=0.01)
print('MLE Estimation of Alpha Parameter')
alpha_m, logfs = train_gp(alpha_model, 2000, opt, verbose=True)
print('MLE Estimation of P Parameter')
p_m, logfs = train_gp(p_model, 2000, opt, verbose=True)

print('Model Summary after MLE')
print('Alpha')
print_summary(alpha_model)
print('P')
print_summary(p_model)

#saving model summary of parameters estimated through MLE
alpha_model_summary = np.array(list(read_values(alpha_model).items()),dtype=object)
np.save(f'{folder_path}AP_GPLima_MLE_Alpha_Model_Summary.npy',alpha_model_summary)
p_model_summary = np.array(list(read_values(p_model).items()),dtype=object)
np.save(f'{folder_path}AP_GPLima_MLE_P_Model_Summary.npy',p_model_summary)

print('MCMC Estimation of Parameters')

# creating priors for applying mcmc with initial estimates from MLE for initialisation
alpha_model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
alpha_model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
alpha_model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

p_model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
p_model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
p_model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
print('Model Initialisation before MCMC')
print('Alpha')
print_summary(alpha_model)
print('P')
print_summary(p_model)

######################################################
# Need to ask Erick about above - unclear what Gamma(1,1) prior is? Code from: https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/mcmc.html?highlight=mcmc
######################################################

#running mcmc with tensorflow
num_burnin_steps = ci_niter(300)
num_samples = ci_niter(500)

alpha_hmc_helper = gpflow.optimizers.SamplingHelper(
    alpha_model.log_posterior_density, alpha_model.trainable_parameters
)

p_hmc_helper = gpflow.optimizers.SamplingHelper(
    p_model.log_posterior_density, p_model.trainable_parameters
)

alpha_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=alpha_hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
)

p_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=p_hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
)

alpha_adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    alpha_hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
)

p_adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    p_hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
)

@tf.function
def alpha_run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=alpha_hmc_helper.current_state,
        kernel=alpha_adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

@tf.function
def p_run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=p_hmc_helper.current_state,
        kernel=p_adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

alpha_samples, alpha_traces = alpha_run_chain_fn()
alpha_parameter_samples = alpha_hmc_helper.convert_to_constrained_values(alpha_samples)

p_samples, p_traces = p_run_chain_fn()
p_parameter_samples = p_hmc_helper.convert_to_constrained_values(p_samples)

alpha_param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(alpha_model).items()}
p_param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(p_model).items()}

print('Model Summary after MCMC')
print('Alpha')
print_summary(alpha_model)
print('P')
print_summary(p_model)

#saving mcmc parameter samples
parameters = ['kernel_lengthscales','kernel_variance','likelihood_variance']

alpha_parameter_samples
alpha_samples_list = []
for param_sample,param_name in zip(alpha_parameter_samples,parameters):
    alpha_samples_list.append([param_name,param_sample.numpy()])
alpha_samples_array = np.array(alpha_samples_list,dtype=object)
np.save(f'{folder_path}AP_GPLima_MCMC_Alpha_Model_Summary.npy',alpha_samples_array)

p_parameter_samples
p_samples_list = []
for param_sample,param_name in zip(p_parameter_samples,parameters):
    p_samples_list.append([param_name,param_sample.numpy()])
p_samples_array = np.array(p_samples_list,dtype=object)
np.save(f'{folder_path}AP_GPLima_MCMC_P_Model_Summary.npy',p_samples_array)



