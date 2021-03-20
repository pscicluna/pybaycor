# pybaycor 
<sub><sup>*It's all you knead*</sup></sub>
---

Pybaycor ("Pie Baker") is package for estimating Bayesian correlation coefficients with python. It reimplements the ["Bayesian First Aid"](http://www.sumsar.net/blog/2014/03/bayesian-first-aid-pearson-correlation-test/) robust and non-robust Bayesian correlation coefficients in python using PyMC3. It should natively work for datasets with more than 2 features (although that hasn't been tested yet, and the plotting routines for this case are still work in progress). It also provides hierarchical inference of correlations in the presence of measurement uncertainty, based on Matzke et al. (2017), who in turn based their approach on Behseta et al. (2009). This package extends their approaches using multivariate T distributions to provide robust alternatives to the methods they lay out.

## Installation:

Pybaycor can be installed with pip:

    pip install pybaycor
    

## Dependencies:

Pybaycor depends on a small number of packages:

* Numpy
* Matplotlib
* Seaborn
* PyMC3
* Arviz

## Usage:

Pybaycor implements a number of classes for different kinds of inference. The most basic of these is the `BayesianCorrelation` class. This class can be used quite straightfowardly to infer correlations with a multi-dimensional dataset with no measurement uncertainty:

    import pybaycor as pbc
    baycor = pbc.BayesianCorrelation(data=data) #where data is an (n_points, n_dimensions) array or array_like
    baycor.fit(steps=1000, tune=1000) #do MCMC to infer the correlations
    baycor.summarise() #print out a summary of the posteriors from the the MCMC
    baycor.plot_trace(show=True) #Plot the trace and marginal distributions
    baycor.plot_data(show=True) #Plot the original data with the 2-sigma ellipse superimposed on it
    
The summary table will contain rows `chol_corr`, which indicate the summary statistics for the correlation coefficients. This includes the posterior mean and 2-sigma credible interval, as well as Rhat for the chains. The `chol_corr[i,i]` rows should all give means of 1 and standard deviations of 0, while the `chol_corr[i,j]` rows are the rows of interest. Remember that the covariance matrix is symmetrical, so `chol_corr[0,1] == chol_cor[1,0]` and you only need to read off one of those rows.

The (**recommended**) robust interface is available through the `RobustBayesianCorrelation` class. This is invoked identically to the basic class, and uses a multivariate T distribution to reduce the influence of outliers. As a result, there is an additional hyperparameter `nu`, the number of degrees of freedom. Like all hyperparameters in pybaycor, this is chosen to follow a weakly-informative prior. The other methods (`fit`, `summarise`, `plot_trace` and `plot_data`) work identically and transparantly in the robust case as well. However, at present the `plot_data()` method only works for 2-dimensional correlations.

If your data has uncertain measurements, however, these classes are not appropriate. For that purpose, pybaycor implements hierarchical equivalents that perform joint inference on the data and the correlation to determine the distribution of *true* correlations given the diluted, observed correlation. Once again, both robust and non-robust interfaces are available, and I recommend the robust interface although the runtime for `fit()` is roughly 5-times longer. This can be invoked as:

    baycor = pbc.HierarchicalRobustBayesianCorrelation(data=data, sigma=sigma) #where data and sigma are (n_points, n_dimensions) arrays or array_like
    baycor.fit(steps=1000, tune=1000)
    
Because this approach introduces `n_dimensions` parameters per data point, it can be difficult to read the summary. I'm working on improving the default formatting to make this useful, but you can always 

    summary = baycor.summarise()
    
to access the dataframe directly and extract useful parameters. At present, `plot_data()` does not give useful output for the hierarchical correlations, and should not be used. 

## Future work 

* Implement inference of correlations when only some features are uncertain. 
* Improve plotting and output

Community input is most welcome!


## Relevant citations:

```
@inbook{inbook,
author = {Gelman, Andrew and Hill, Jennifer},
year = {2006},
month = {11},
pages = {},
title = {Data Analysis Using Regression And Multilevel/Hierarchical Models},
volume = {3},
isbn = {0521867061},
journal = {Cambridge Universty Press},
doi = {10.1017/CBO9780511790942}
}

@article{doi:10.1152/jn.90727.2008,
author = {Behseta, Sam and Berdyyeva, Tamara and Olson, Carl R. and Kass, Robert E.},
title = {Bayesian Correction for Attenuation of Correlation in Multi-Trial Spike Count Data},
journal = {Journal of Neurophysiology},
volume = {101},
number = {4},
pages = {2186-2193},
year = {2009},
doi = {10.1152/jn.90727.2008},
    note ={PMID: 19129297},

URL = { 
        https://doi.org/10.1152/jn.90727.2008
    
},
eprint = { 
        https://doi.org/10.1152/jn.90727.2008
    
}
}

@article{10.1525/collabra.78,
    author = {Matzke, Dora and Ly, Alexander and Selker, Ravi and Weeda, Wouter D. and Scheibehenne, Benjamin and Lee, Michael D. and Wagenmakers, Eric-Jan},
    title = "{Bayesian Inference for Correlations in the Presence of Measurement Error and Estimation Uncertainty}",
    journal = {Collabra: Psychology},
    volume = {3},
    number = {1},
    year = {2017},
    month = {10},
    issn = {2474-7394},
    doi = {10.1525/collabra.78},
    url = {https://doi.org/10.1525/collabra.78},
    note = {25},
    eprint = {https://online.ucpress.edu/collabra/article-pdf/3/1/25/436268/78-1314-1-pb.pdf},
}
```
