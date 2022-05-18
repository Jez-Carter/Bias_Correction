import numpy as np
import pytest
from scipy.stats import gamma
from math import isclose
from src.model_fitting_functions import BernoulliGamma

p,alpha,beta = 0.7,0.5,2
scale = 1/beta
dist = BernoulliGamma((p,alpha,beta))

def test_loglikelihood_typical():
    values = np.array([0]) 
    assert isclose(dist.log_prob(values), np.log(1-p), rel_tol=1e-3, abs_tol=0.0)
    values = np.array([1]) 
    assert isclose(dist.log_prob(values), np.log(p)+np.log(gamma.pdf(values,a=alpha,loc=0,scale=scale)), rel_tol=1e-3, abs_tol=0.0)
    values = np.array([-1]) 
    assert np.isfinite(dist.log_prob(values))
    values = np.array([1000]) 
    assert np.isfinite(dist.log_prob(values))
        
