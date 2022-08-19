[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

***Current Progress:***

The TinyGP, Numpyro and Xarray packages have been used to reproduce the model from [Lima et al. 2021](http://dx.doi.org/10.1016/j.jhydrol.2021.126095). The python scripts to fit the Lima model are in /scripts/Model_Development/Lima_Methodology and include a Bernoulli-Gamma model fit (bg_lima.py) a Gaussian Process fit (gp_lima_tinygp.py) and a quantile mapping procedure (qm_lima.py). A fully Bayesian hierarchical version of this model has been coded and run on a GPU on colab taking around 10mins (/scripts/Model_Development/bg_tinygp.py) - the colab notebook for running this is the following: [Bias Correction](https://colab.research.google.com/drive/1d4HDeDqS8yW86ohiRQLM2W7aSQM8Osip?usp=sharing). 

Modules have been created with useful functions that can be imported, these are within the /src folder and include: model_fitting_functions.py (contains functions defining models); helper_functions.py (contains functions that perform specific tasks such as reshaping data etc.); and plotting_functions.py (contains functions for doing map plots with coastlines etc.).

Raw data is stored in /data/RawData and there are some preprocessing steps before fitting the models, which are included in: /scripts/Model_Development/Data_PreProcessing/. Processed data and model outputs are stored in: /data/ProcessedData.

Notebooks exploring some of the model output are included in: /scripts/Model_Development/Examining_Results. 

Notebooks exploring aspects such as how a Gamma distribution looks with different parameters is are included in: /scripts/Exploratory_Notebooks.

Tests for the code are included in: /tests. These remain very limited at the moment and only contain a simple test for the Bernoulli-Gamma distribution we define. 

***Next Steps***

- At the moment estimates of the parameters are only dependent on our sudo observations. There is no trust or information gained from the climate model output itself. For sparse observations it makes sense to use the climate model output as well to make inference on for example the correlation length scale of our underlying process we're modelling. One option here is to assume a shared GP between the parameters $\Phi_Z$ and  $\Phi_Y$ (W) and that there is some additional GP (D) for $\Phi_Z$ that describes the bias to observations and we expect to have a longer length scale that W. For example, if the bias is constant over the whole domain, this is equivalent to saying the length scale for D is $\infty$.

$W = GP(0,\Sigma_W)$ 

$D = GP(0,\Sigma_D)$ 

$\Phi_Y \sim W + \mu_W + e_Y$

$\Phi_Z \sim W + \mu_W + D + \mu_D + e_Z$

(Note I could include the means within our GP definitions for the code)

Note to code this up requires including $W_{\alpha},D_{\alpha},W_p,D_p$ so 2 extra GPs to what we currently use. We also need the model to ingest both a Y and Z dataset. Our estimate of the parameters associated with the W GP can then be used to make predictions of $\Phi_Y$ at new locations. ***I need to clarify with Erick where the cross correlations come in***. It is expected that to fit this model will require a lot more computational cost. The climate model has 30x35=1050 grid cells over our region, whereas we have been using 100 sudo observations so far. 

Note also, we might need to use different sudo observations to get realistic results of how this method would work. 

- We have found that applying a fully Bayesian model smooths the predictions and results in much greater uncertainty estimates. This is expected as in the modular approach we fit the GP to only one realisation of the parameters $\Phi_Y$ (the expectation), whereas in the fully Bayesian approach our GPs parameters need to fit to all the realisations of our parameters $\Phi_Y$. This is not necessarily a bad thing and the uncertainty is in theory more realistic of the truth. However, our 'expert knowledge' might suggest that the length scale is now too large and the fit too smooth - in which case we can narrow our prior beliefs (at the moment we use a half normal of mean 1 to model the lengthscale of log_alpha, this is equivalent to considering a half normal of mean of 2.718 to model the lengthscale of alpha and this is perhaps a bit too non-informative (Erick suggests that often a constraint is placed with a maximum of 1/3 the domain width, although this depends on the patterns observed). ***Discuss with Erick what happens when you want for example a mean of 1 for the half normal describing the lengthscale of alpha, which corresponds to 0 for log alpha).***

***Transforming the lengthscale using a simple say exp(log_alpha_lengthscale) is incorrect and to compare the fully bayesian model we may have to rewrite the modular model to make inference on log alpha. This is going to impact interpretations as well maybe readup about this.***

In addition, Erick mentioned that sometimes the error term is excluded from the GP - ***I'm not sure how this works when we have multiple realisations of alpha to fit to etc***.  

- At the moment our sudo observations are just randomly selected grid cells from our climate model. Therefore, we do not need to worry about differences in support. In reality observations are point like whereas the climate model output is area averaged. To account for this we might want to include some aggregating term the relates $W_{\alpha,Y}$ to $W_{\alpha,Z}$ for example.

- The examining results notebooks currently don't compare to what would happen if we fit a BG to all grid-cells of our climate model independently - it would be interesting to include this. 

- We want to put an uncertainty band on our bias corrected time series at each location, which is not done in Lima - would be quite nice to plot.
 