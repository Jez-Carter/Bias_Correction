# %%
#Importing packages
import numpyro.distributions as dist
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from jax import random

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

# %% [markdown]
#Normal Distribution: \
#Typical parameterisation (loc,scale) (a.k.a, mean, stddev):
#\
#In Numpyro they use mean (loc) and stddev (scale). 

# %%
fig, ax = plt.subplots()
xs = np.linspace(-3,3,100)
ys = norm.pdf(xs, 0, 1)
ax.plot(xs, ys, lw=2, label='Scipy')
plt.hist(dist.Normal(0.0,1.0).sample(rng_key,(10000,)),density=True,bins=100,label='Numpyro')
plt.legend()
plt.show()

