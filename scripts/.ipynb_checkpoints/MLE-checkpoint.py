import sys
import os
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("figure", figsize=(10, 10))
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"

data_path = sys.argv[1]
data = np.load(data_path) 

# Defining the function to minimise:

def negloglikelihoodsum(params, data):
    p, alpha, scale = (
        params[0],
        params[1],
        params[2],
    )  # Note scale is 1/Beta and is the parameter used in the scipy.stats.gamma function instead of beta

    likelihood_values = np.concatenate(
        (
            np.full(len(data[data == 0]), 1 - p),
            p * stats.gamma.pdf(data[data != 0], a=alpha, loc=0, scale=scale),
        )
    )

    return -np.sum(np.log(likelihood_values))

# Applying Nelder-Mead Minimiser

# Parameter Guesses [p,alpha,scale]
guess = [0.5, 0.1, 1.0]

# Parameter Bounds (I've found this is needed to improve stability of the minimiser)
bounds = [
    (0.001, 0.99),
    (0.00001, 50),
    (0.00001, 50),
]

# Minimising Function
results = minimize(
    negloglikelihoodsum,
    guess,
    args=data,
    method="Nelder-Mead",
    options={"disp": True},
    bounds=bounds,
)
print(results["x"])

# Histogram of Data

bins = np.arange(0, 5, 0.1)
bin_width = bins[1] - bins[0]
bin_centers = bins + bin_width / 2

p = len(data[data != 0]) / len(data)
print(f"Probability of Snowfall (Data) = {round(1-p,2)}")
weights = np.full(len(data[data != 0]), p) / len(data[data != 0])
plt.hist(
    data[data != 0],
    bins=bins,
    histtype="step",
    stacked=True,
    fill=False,
    weights=weights,
    label="Histogram of Data",
)

# Plotting the Probability Density Function that Maximises the Likelihood
p, alpha, scale = results["x"]
xs = np.arange(0, 5, 0.01)
likelihoods = p * stats.gamma.pdf(xs, a=alpha, loc=0, scale=scale)
estimate = likelihoods * bin_width
print(f"Probability of Snowfall (Estimate) = {round(1-p,2)}")
plt.plot(xs, estimate, label="MLE")

plt.legend()
plt.show()