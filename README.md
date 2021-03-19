# pybaycor

PyBayCor ("Pie Baker") is package for estimating Bayesian correlation coefficients with python. At present, it reimplements the ["Bayesian First Aid"](http://www.sumsar.net/blog/2014/03/bayesian-first-aid-pearson-correlation-test/) robust and non-robust Bayesian correlation coefficients in python using PyMC3. It should natively work for datasets with more than 2 features (although that hasn't been tested yet, and the plotting routines for this case are still work in progress).

Future work: Implement hierarchical inference of correlations in the presence of measurement uncertainty.
