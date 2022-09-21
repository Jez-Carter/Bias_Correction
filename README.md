[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

***Current Progress:***

The TinyGP, Numpyro and Xarray packages have been used to reproduce the model from [Lima et al. 2021](http://dx.doi.org/10.1016/j.jhydrol.2021.126095). The python scripts to fit the Lima model are in /scripts/Model_Development/Lima_Methodology and include a Bernoulli-Gamma model fit (bg_lima.py) a Gaussian Process fit (gp_lima_tinygp.py) and a quantile mapping procedure (qm_lima.py). A fully Bayesian hierarchical version of this model has been coded and run on a GPU on colab taking around 10mins (/scripts/Model_Development/bg_tinygp.py) - the colab notebook for running this is the following: [Bias Correction](https://colab.research.google.com/drive/1d4HDeDqS8yW86ohiRQLM2W7aSQM8Osip?usp=sharing). 

Modules have been created with useful functions that can be imported, these are within the /src folder and include: model_fitting_functions.py (contains functions defining models); helper_functions.py (contains functions that perform specific tasks such as reshaping data etc.); and plotting_functions.py (contains functions for doing map plots with coastlines etc.).

Raw data is stored in /data/RawData and there are some preprocessing steps before fitting the models, which are included in: /scripts/Model_Development/Data_PreProcessing/. Processed data and model outputs are stored in: /data/ProcessedData.

Notebooks exploring some of the model output are included in: /scripts/Model_Development/Examining_Results. 

Notebooks exploring aspects such as how a Gamma distribution looks with different parameters is are included in: /scripts/Exploratory_Notebooks.

Tests for the code are included in: /tests. These remain very limited at the moment and only contain a simple test for the Bernoulli-Gamma distribution we define. 

***Next Steps***

- At the moment estimates of the parameters are only dependent on our sudo observations. There is no trust or information gained from the climate model output itself. For sparse observations it makes sense to use the climate model output as well to make inference on for example the correlation length scale of our underlying process we're modelling. One option here is to assume a shared GP between the parameters $\Phi_Z$ and  $\Phi_Y$ (W) and that there is some additional GP (D) for $\Phi_Z$ that describes the bias to observations and that is expected to have a longer length scale than W. For example, if the bias is constant over the whole domain, this is equivalent to saying the length scale for D is $\infty$.

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

***Code Adjustments/Errors that need fixing***

- scripts/Model_Development/Examining_Results/Examining GP Output Lima.ipynb needs some further work and updates - bulleted list in notebook shows what updates to do
- scripts/Model_Development/Examining_Results/Examining Bayesian Hierarchical Model Output.ipynb needs to be updated base on the 'Examining GP Output Lima.ipynb' notebook and it would be nice to have some direct comparison plots between the fully Bayesian and modular approaches - this might require first using link functions in the modular approach so that the GP is being fit to the logit_p, log_alpha etc and then lengthscales and uncertainties can be directly compared.
- I need to watch the 'how to think in jax' video and in particular the jit compiling section to understand where in the code can be sped up
- It would be nice if the scripts/Model_Development/Lima_Methodology/Running Scripts.ipynb ran from top to bottom and generated all results for the lima methodology - maybe I need to think about how I could get this to run on colab quickly and then copy across the results?
- scripts/Model_Development/bg_tinygp.py needs adjusting such that it can take in an argument defining the input data path and the output path.
- It might help to create a colab folder containing notebooks for running code in colab, these can initially be opened by going to the github repo, finding the notebook and then using the open in colab tab. (Alternatively I could create a colab initialise script that takes all the preamble needed in colab notebooks and can be called whenever using colab - e.g. changing directory and the base directory of the project perhaps? - pip installing various things etc.)
- scripts/Model_Development/Lima_Methodology/bg_lima_all.py needs merging with bg_lima.py such that we just have one bg_lima.py script - this should be simple and just involves adjusting a couple of hard coded aspects like using 100 rather than 1050 sites etc. Note there is a Fitting BG to All Grid Cells.ipynb colab notebook I created, this could be a good test case for seeing if we can create a pre-amble script to use when using colab rather than having a seperate colab notebook saved somewhere.
- Potentially adjust the way I do reshaping in scripts/Model_Development/Lima_Methodology/qm_lima.py and qm_proper_uncertainty_propagation.py to the way I do it in scripts/Model_Development/Examining_Results/Examining GP Output Lima.ipynb, which is a bit cleaner.
- scripts/Model_Development/Lima_Methodology/qm_proper_uncertainty_propagation.py has been created and for which the code is still being developed in scripts/Code Development.ipynb, this notebook and script need some more developing and in addition the scripts/Model_Development/Examining_Results/QM Output Lima.ipynb needs updating and developed to nicely plot/capture uncertainty bands. 
- Exploring getting %matplotlib widget working and using a new lab with the voila functionality.
- General note that I need to check that uncertainty is being calculated in the proper way, where we take lots of realisations of the GPs parameters and predictions and then build up our posterior predictive from that.