[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

***Current Progress:***

The TinyGP, Numpyro and Xarray packages have been used to reproduce the model from [Lima et al. 2021](http://dx.doi.org/10.1016/j.jhydrol.2021.126095). The python scripts to fit the Lima model are in /scripts/Model_Development/Lima_Methodology and include a Bernoulli-Gamma model fit (bg_lima.py) a Gaussian Process fit (gp_lima_tinygp.py) and a quantile mapping procedure (qm_lima.py). A fully Bayesian hierarchical version of this model has been coded and run on a GPU on colab taking around 10mins (/scripts/Model_Development/bg_tinygp.py) - the colab notebook for running this is the following: [Bias Correction](https://colab.research.google.com/drive/1d4HDeDqS8yW86ohiRQLM2W7aSQM8Osip?usp=sharing). 

Modules have been created with useful functions that can be imported, these are within the /src folder and include: model_fitting_functions.py (contains functions defining models); helper_functions.py (contains functions that perform specific tasks such as reshaping data etc.); and plotting_functions.py (contains functions for doing map plots with coastlines etc.).

Raw data is stored in /data/RawData and there are some preprocessing steps before fitting the models, which are included in: /scripts/Model_Development/Data_PreProcessing/. Processed data and model outputs are stored in: /data/ProcessedData.

Notebooks exploring some of the model output are included in: /scripts/Model_Development/Examining_Results. 

Notebooks exploring aspects such as how a Gamma distribution looks with different parameters is are included in: /scripts/Exploratory_Notebooks.

Tests for the code are included in: /tests. These remain very limited at the moment and only contain a simple test for the Bernoulli-Gamma distribution we define. 

***Comments***

***Transforming the lengthscale using a simple say exp(log_alpha_lengthscale) is incorrect and to compare the fully bayesian model we may have to rewrite the modular model to make inference on log alpha. This is going to impact interpretations as well maybe readup about this.***

In addition, Erick mentioned that sometimes the error term is excluded from the GP - ***I'm not sure how this works when we have multiple realisations of alpha to fit to etc***.  
