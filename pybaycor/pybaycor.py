import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gs
import arviz as az
import xarray as xr

class BayesianCorrelation():
    """ A class to infer Bayesian correlation coefficients for multidimensional data without uncertainties

    Parameters
    ----------
    data : float, (n_points, n_dim) array_like
        The multidimensional dataset to infer correlations on.
        Either data OR both x & y should be passed as input.
    x : float, (n_points) array_like
        Array_like of x values (optional; only for 2-D datasets)
    y : float, (n_points array_like
        Array_like of y values (optional; only for 2-D datasets)
    ndim : int, optional
        The number of dimensions in the input data. 
        If not given, it will be inferred from the data
    mu_prior : length-2 or (2, ndim) iterable of floats, optional, default (0., 1000.)
        The mean and standard deviation of the Gaussian prior on the multivariate Normal distribution
    sigma_prior : scalar or (ndim) iterable of floats, optional, default 200.
        The prior on the scale parameter (beta) of the half-Cauchy prior on the standard deviations of the multivariate Normal distribution

    Attributes
    ----------
    None

    Methods
    --------
    fit : Fit the data assuming they are drawn from a multivariate Normal distribution

    summarise : summarise the results of the fit

    plot_trace : plot the trace and marginal distributions of the trace

    plot_data : plot the data overlaid with the ellipse described by the inferred correlated multivariate Normal

    plot_corner : plot the 1D and 2D marginal distributions of the inferred parameters.

    Examples
    --------

    Creating an instance is as simple as 

    >>> import pybaycor as pbc
    >>> bc = pbc.BayesianCorrelation(data=data)

    Once you have created the instances, the fit is run with 

    >>> bc.fit()

    or you can modify the length of burn-in and number of steps with

    >>> bc.fit(steps=2000, tune=2000)

    Once you are happy with the fit, you can get a tabular summary with

    >>> summary = bc.summarise()

    and visual summaries with

    >>> bc.plot_trace()
    >>> bc.plot_corner()
    >>> bc,plot_data()
        
    """
    def __init__(self,data=None, x=None, y=None, ndim = None, mu_prior=[0.0,1000.], sigma_prior=200.):
        
        
        #if ndim is None:
        self.fitted=False
        self.plot_trace_vars = ['mu', "chol_corr"]
        if data is None:
            if x is None and y is None:
                raise ValueError("Either data must be given as input, or x and y")
            else:
                self.ndim = 2
                self.data = np.column_stack((x,y))
        else:
            if ndim is None:
                self.ndim = data.shape[1]
            else:
                self.ndim = ndim
                if self.ndim != data.shape[1]:
                    raise ValueError("Data must have the same number of features and ndim")
            self.data = data

        self.model = pm.Model()
        with self.model:
            #we put weakly informative priors on the means and standard deviations of the multivariate normal distribution
            mu = pm.Normal("mu", mu=mu_prior[0], sigma=mu_prior[1], shape=self.ndim)
            sigma = pm.HalfCauchy.dist(sigma_prior)
            #and a prior on the covariance matrix which weakly penalises strong correlations
            chol, corr, stds = pm.LKJCholeskyCov("chol", n=self.ndim, eta=2.0, sd_dist=sigma, compute_corr=True)
            #the prior gives us the Cholesky Decomposition of the covariance matrix, so for completeness we can calculate that determinisitically
            cov = pm.Deterministic("cov", chol.dot(chol.T))

            #and now we can put our observed values into a multivariate normal to complete the model
            vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=self.data)
        pass



    def fit(self,steps=1000, tune=1000, summarise=False):
        """ Fit the model to infer the correlation coefficient

        Parameters
        ----------
        steps : int, optional, default 1000
            Number of MCMC steps per chain after burn-in
        tune : int, optional, default 1000
            Number of steps per chain for burn-in
        summarise : bool, default False
            Whether to produce the table summary (also available through summarise())

        """
        with self.model:
            self.trace = pm.sample(
                steps, tune=tune, target_accept=0.9, compute_convergence_checks=False,return_inferencedata=True
            )
            self.fitted=True
        if summarise:
            self.summary = az.summary(self.trace, var_names=["~chol"], round_to=2)
            #self.rho = [self.summary['hdi_3%'][chol_corr[1,0]],self.summary['mean'][chol_corr[1,0]],self.summary['hdi_97%'][chol_corr[1,0]]]
            print(self.summary)
            return self.trace, self.summary

        return self.trace

    def summarise(self):
        """ Summarise the results of the model

        Parameters
        ----------
        None
        """
        self.summary = az.summary(self.trace, var_names=["~chol"], round_to=2)
        print(self.summary)
        return self.summary
        

    def plot_trace(self,plotfile=None, show=False):
        """ Plot the trace of the MCMC run along with the marginal distributions of a subset of parameters

        Parameters
        ----------
        plotfile : str, optional
            Name of a file to write the plot to
        show : bool, optional, default False
            Whether to show the plot window

        """
        if not self.fitted:
            pass #raise an error here
        ax = az.plot_trace(
            self.trace,
            var_names=self.plot_trace_vars,
            #filter_vars="regex",
            compact=True,
            #lines=[
                #("mu", {}, mu),
                #("cov", {}, cov),
                #("chol_stds", {}, sigma),
                #("chol_corr", {}, rho),
            #],
        )
        if isinstance(plotfile, str):
            plt.save(plotfile)
        if show:
            plt.show()
        #elif plotfile is not None:
        #    plt.close()
        #should this also return the ax?
            

    def plot_data(self,plotfile=None, show=None):
        """ Plot the input data overlaid with the ellipse described by the inferred correlated multivariate distribution

        Parameters
        ----------
        plotfile : str, optional
            Name of a file to write the plot to
        show : bool, optional, default False
            Whether to show the plot window

        """
        #Currently only supports 2D correlations
        #if self.ndim != 2:
        #    raise NotImplementedError("This routine doesn't support plotting correlations in more than 2 dimensions yet!")
        if not self.fitted:
            raise RuntimeError("Please run fit() before attempting to plot the results")
        if self.ndim==np.int(2) and isinstance(self.ndim, int):
            blue, _, red, *_ = sns.color_palette()
            f, ax = plt.subplots(1, 1, figsize=(5, 4))#, gridspec_kw=dict(width_ratios=[4, 3]))

            sns.scatterplot(x=self.data[:,0], y=self.data[:,1])

            mu_post = self.trace.posterior["mu"].mean(axis=(0, 1)).data
    
            sigma_post = self.trace.posterior["cov"].mean(axis=(0, 1)).data
    
            var_post, U_post = np.linalg.eig(sigma_post)
            angle_post = 180.0 / np.pi * np.arccos(np.abs(U_post[0, 0]))

            e_post = Ellipse(
                mu_post,
                2 * np.sqrt(5.991 * var_post[0]),
                2 * np.sqrt(5.991 * var_post[1]),
                angle=angle_post,
            )
            e_post.set_alpha(0.5)
            e_post.set_facecolor(blue)
            e_post.set_zorder(10)
            ax.add_artist(e_post)
            rect_post = plt.Rectangle((0, 0), 1, 1, fc=blue, alpha=0.5)
            ax.legend(
                [rect_post],
                ["Estimated 95% density region"],
                loc=2,
            )
            #plt.show()

        elif self.ndim > 2 and isinstance(int, self.ndim) and np.isfinite(self.ndim):
            #raise NotImplementedError("This routine doesn't support plotting correlations in more than 2 dimensions yet!")
            rows = self.ndim - 1
            cols = self.ndim - 1
            fig = plt.figure()
            gs = fig.add_gridSpec(rows, cols,left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
            for i in range(self.ndim - 1):
                for j in range(i+1,self.ndim - 1):
                    ax = fig.add_subplot(gs[i,j])
                    #plot the data points
                    sns.scatterplot(self.data[:,i], self.data[:,j], ax=ax)
                    mu_post = self.trace.posterior["mu"].mean(axis=(i, j)).data
    
                    sigma_post = self.trace.posterior["cov"].mean(axis=(i, j)).data
                    
                    var_post, U_post = np.linalg.eig(sigma_post)
                    angle_post = 180.0 / np.pi * np.arccos(np.abs(U_post[0, 0]))
                    
                    e_post = Ellipse(
                        mu_post,
                        2 * np.sqrt(5.991 * var_post[0]),
                        2 * np.sqrt(5.991 * var_post[1]),
                        angle=angle_post,
                    )
                    e_post.set_alpha(0.5)
                    e_post.set_facecolor(blue)
                    e_post.set_zorder(10)
                    ax.add_artist(e_post)
                    
        else:
            raise ValueError("Ndim is either less than 2 or is not an integer!")
            
        if isinstance(plotfile, str):
            plt.save(plotfile)
        elif not show:
            raise TypeError("plotfile must be a string")
        if show:
            plt.show()
        #elif plotfile is not None:
        #    plt.close()


    
    def plot_corner(self, point_estimate='mean',plotfile=None,show=True):
        """ Plot the 1D and 2D marginal distributions of the inferred parameters

        Parameters
        ----------
        plotfile : str, optional
            Name of a file to write the plot to
        show : bool, optional, default False
            Whether to show the plot window

        """
        #For consistency's sake I'm going to re-invent the wheel here, and manually create a grid of plots from arviz, rather than letting corner do the work. This is because I want to make sure specific entries are plotted in a specific order.
        
        plot_vars = self.plot_trace_vars#[:-1]
        chol_coords = []
        if self.ndim == 2:
            #chol_coords.append(0)
            #chol_coords.append(1)
            chol_coords=(0,1)
            coords = {"chol_corr_dim_0":[0], "chol_corr_dim_1":[1]}
            #plot_vars.append("chol_corr[0,1]")
        else:
            coords = {"chol_corr_dim_0":[], "chol_corr_dim_1":[]}
            d0 = []
            d1 = []
            #raise NotImplementedError("Corner plots for data with more than 2 dimensions are not available yet!")
            for i in range(self.ndim - 1):
                for j in range(1,self.ndim - 1):
                    d0.append(i)
                    d1.append(j)
                    #print(i,j)
                    #chol_coords.append([i,j])#"chol_corr["+str(i)+","+str(j)+"]")

            coords["chol_corr_dim_0"] = xr.DataArray(d0, dims=['pointwise_sel'])
            coords["chol_corr_dim_1"] = xr.DataArray(d1, dims=['pointwise_sel'])
        #print(plot_vars)
        #coords = {"chol_corr":chol_coords}
        #print(coords)
        #corner = gs.GridSpec(rows, cols, figure=fig
        az.plot_pair(self.trace,
                     var_names = plot_vars,
                     coords = coords,
                     kind="kde",
                     marginals=True,
                     point_estimate=point_estimate,
                     show=show,
            )

        if isinstance(plotfile, str) and not show:
            plt.save(plotfile)
        elif not show:
            raise TypeError("plotfile must be a string")
        #pass
        

class RobustBayesianCorrelation(BayesianCorrelation):
    """ A class to infer robust Bayesian correlation coefficients for multidimensional data without uncertainties

    Parameters
    ----------
    data : float, (n_points, n_dim) array_like
        The multidimensional dataset to infer correlations on.
        Either data OR both x & y should be passed as input.
    x : float, (n_points) array_like
        Array_like of x values (optional; only for 2-D datasets)
    y : float, (n_points array_like
        Array_like of y values (optional; only for 2-D datasets)
    ndim : int, optional
        The number of dimensions in the input data. 
        If not given, it will be inferred from the data
    mu_prior : length-2 or (2, ndim) iterable of floats, optional, default (0., 1000.)
        The mean and standard deviation of the Gaussian prior on the multivariate t distribution
    sigma_prior : scalar or (ndim) iterable of floats, optional, default 200.
        The prior on the scale parameter (beta) of the half-Cauchy prior on the standard deviations of the multivariate t distribution

    Attributes
    ----------
    None

    Methods
    --------
    fit : Fit the data assuming they are drawn from a multivariate Normal distribution

    summarise : summarise the results of the fit

    plot_trace : plot the trace and marginal distributions of the trace

    plot_data : plot the data overlaid with the ellipse described by the inferred correlated multivariate Normal

    plot_corner : plot the 1D and 2D marginal distributions of the inferred parameters.

    Examples
    --------

    Creating an instance is as simple as 

    >>> import pybaycor as pbc
    >>> bc = pbc.BayesianCorrelation(data=data)

    Once you have created the instances, the fit is run with 

    >>> bc.fit()

    or you can modify the length of burn-in and number of steps with

    >>> bc.fit(steps=2000, tune=2000)

    Once you are happy with the fit, you can get a tabular summary with

    >>> summary = bc.summarise()

    and visual summaries with

    >>> bc.plot_trace()
    >>> bc.plot_corner()
    >>> bc,plot_data()

    """
    def __init__(self,data=None, x=None, y=None, ndim = None, mu_prior=[0.0,1000.], sigma_prior=200.):
        #if ndim is None:
        self.fitted=False
        self.plot_trace_vars = ['mu', "nu", "chol_corr"] #, "~nu-1", "~cov", "~chol_stds", "~chol"]
        if data is None:
            if x is None and y is None:
                raise ValueError("Either data must be given as input, or x and y")
            else:
                self.ndim = 2
                self.data = np.column_stack((x,y))
        else:
            if ndim is None:
                self.ndim = data.shape[1]
            else:
                self.ndim = ndim
                if self.ndim != data.shape[1]:
                    raise ValueError("Data must have the same number of features and ndim")
            self.data = data

        self.model = pm.Model()
        with self.model:
            #we put weakly informative priors on the means and standard deviations of the multivariate normal distribution
            mu = pm.Normal("mu", mu=mu_prior[0], sigma=mu_prior[1], shape=self.ndim)
            sigma = pm.HalfCauchy.dist(sigma_prior)
            #and a prior on the covariance matrix which weakly penalises strong correlations
            chol, corr, stds = pm.LKJCholeskyCov("chol", n=self.ndim, eta=2.0, sd_dist=sigma, compute_corr=True)
            #the prior gives us the Cholesky Decomposition of the covariance matrix, so for completeness we can calculate that determinisitically
            cov = pm.Deterministic("cov", chol.dot(chol.T))

            nuMinusOne = pm.Exponential('nu-1', lam=1./29.)
            nu = pm.Deterministic('nu', nuMinusOne + 1)

            #and now we can put our observed values into a multivariate t distribution to complete the model
            vals = pm.MvStudentT('vals', nu = nu, mu=mu, chol=chol, observed=self.data)


class HierarchicalBayesianCorrelation(BayesianCorrelation):
    """A class to infer Bayesian correlation coefficients for uncertain multidimensional data

    Parameters
    ----------
    data : float, (n_points, n_dim) array_like
        The multidimensional dataset to infer correlations on.
    sigma : float, (n_points, n_dim) array_like
        The uncertainties of the multidimensional dataset to infer correlations on.
    mu_prior : length-2 or (2, ndim) iterable of floats, optional, default (0., 1000.)
        The mean and standard deviation of the Gaussian prior on the multivariate Normal distribution
    sigma_prior : scalar or (ndim) iterable of floats, optional, default 200.
        The prior on the scale parameter (beta) of the half-Cauchy prior on the standard deviations of the multivariate Normal distribution

    Attributes
    ----------
    None

    Methods
    --------
    fit : Fit the data assuming they are drawn from a multivariate Normal distribution

    summarise : summarise the results of the fit

    plot_trace : plot the trace and marginal distributions of the trace

    plot_data : plot the data overlaid with the ellipse described by the inferred correlated multivariate Normal

    plot_corner : plot the 1D and 2D marginal distributions of the inferred parameters.

    Examples
    --------

    Creating an instance is as simple as 

    >>> import pybaycor as pbc
    >>> bc = pbc.BayesianCorrelation(data=data)

    Once you have created the instances, the fit is run with 

    >>> bc.fit()

    or you can modify the length of burn-in and number of steps with

    >>> bc.fit(steps=2000, tune=2000)

    Once you are happy with the fit, you can get a tabular summary with

    >>> summary = bc.summarise()

    and visual summaries with

    >>> bc.plot_trace()
    >>> bc.plot_corner()
    >>> bc,plot_data()

    """
    def __init__(self, data, sigma, mu_prior=[0.0,1000.], sigma_prior=200.):

        self.fitted=False
        if np.any(sigma <=0.):
            raise ValueError("Uncertainties must be positive real numbers!")
        self.plot_trace_vars = ['mu', "chol_corr"]
        if data is None:
            raise ValueError("Either data must be given as input, or x and y")
        else:
            self.ndim = data.shape[1]
            self.npoints = data.shape[0]
            self.data = data
            if data.shape != sigma.shape:
                raise RuntimeError("data and sigma must have the same shape!")
            self.sigma = sigma

        self.model = pm.Model()
        with self.model:
            #we put weakly informative hyperpriors on the means and standard deviations of the multivariate normal distribution
            mu = pm.Normal("mu", mu=mu_prior[0], sigma=mu_prior[1], shape=self.ndim)
            sigma = pm.HalfCauchy.dist(sigma_prior)
            #and a hyperprior on the covariance matrix which weakly penalises strong correlations
            chol, corr, stds = pm.LKJCholeskyCov("chol", n=self.ndim, eta=2.0, sd_dist=sigma, compute_corr=True)
            #the hyperprior gives us the Cholesky Decomposition of the covariance matrix, so for completeness we can calculate that determinisitically
            cov = pm.Deterministic("cov", chol.dot(chol.T))

            #and now we can construct our multivariate normals to complete the prior
            prior = pm.MvNormal('vals', mu=mu, chol=chol, shape=(self.npoints,self.ndim)) #, observed=self.data)
            #print(prior)
            #help(prior)
            mu1s = prior[:,0]

            datavars = []
            datavars = pm.Normal("data", mu = prior, sigma = self.sigma, observed = self.data)
            #Finally, we need to define our data
            #for i in range(self.ndim):
            #    datavars.append(pm.Normal("data_"+str(i), mu=prior[:,i], sigma = self.sigma[:,i], observed=self.data[:,i]))

            print(datavars)

    def data_summary(self, printout=True):
        """

        """
        #if self.summary is None:
        self.summary_data = az.summary(self.trace, var_names=["vals"], filter_vars="like", round_to=2)
        if printout:
            print(self.summary_data)
        return self.summary_data



    def model_summary(self):
        """

        """
        if self.summary is None:
            self.summary = az.summary(self.trace, var_names=["~chol","~vals"], round_to=2)
        pass

    def plot_data(self, plot_input=True, plot_fitted=True,plotfile=None, show=None):
        """Plot the input data overlaid with the ellipse described by the inferred correlated multivariate distribution

        Parameters
        ----------
        plot_input : bool, default True
            Whether to plot the input data and their uncertainties
        plot_fitted : bool, default True
            Whether to plot the inferred data and their inferred uncertainties
        plotfile : str, optional
            Name of a file to write the plot to
        show : bool, optional, default False
            Whether to show the plot window

        """
        if not self.fitted:
            raise RuntimeError("Please run fit() before attempting to plot the results")

        fitted_data = self.data_summary(printout=False)
        fitted_mean = fitted_data['mean'].to_numpy().reshape((self.npoints,self.ndim))
        print(fitted_mean.shape)
        fitted_sigma = fitted_data['sd'].to_numpy().reshape((self.npoints,self.ndim))
        if self.ndim==np.int(2) and isinstance(self.ndim, int):
            blue, _, red, *_ = sns.color_palette()
            f, ax = plt.subplots(1, 1, figsize=(5, 4))#, gridspec_kw=dict(width_ratios=[4, 3]))

            sns.scatterplot(x=self.data[:,0], y=self.data[:,1])
            if plot_input:
                ax.errorbar(x=self.data[:,0], y=self.data[:,1],
                            xerr=self.sigma[:,0], yerr=self.sigma[:,1],fmt='o',label='input data')
            
            if plot_fitted:
                ax.errorbar(x=fitted_mean[:,0], y=fitted_mean[:,1],
                            xerr=fitted_sigma[:,0], yerr=fitted_sigma[:,1],fmt='o',label='inferred data')
            
            mu_post = self.trace.posterior["mu"].mean(axis=(0, 1)).data
    
            sigma_post = self.trace.posterior["cov"].mean(axis=(0, 1)).data
    
            var_post, U_post = np.linalg.eig(sigma_post)
            angle_post = 180.0 / np.pi * np.arccos(np.abs(U_post[0, 0]))

            e_post = Ellipse(
                mu_post,
                2 * np.sqrt(5.991 * var_post[0]),
                2 * np.sqrt(5.991 * var_post[1]),
                angle=angle_post,
            )
            e_post.set_alpha(0.5)
            e_post.set_facecolor(blue)
            e_post.set_zorder(10)
            ax.add_artist(e_post)
            rect_post = plt.Rectangle((0, 0), 1, 1, fc=blue, alpha=0.5)
            ax.legend(
                [rect_post],
                ["Estimated 95% density region"],
                loc=2,
            )
            #plt.show()

        elif self.ndim > 2 and isinstance(int, self.ndim) and np.isfinite(self.ndim):
            #raise NotImplementedError("This routine doesn't support plotting correlations in more than 2 dimensions yet!")
            rows = self.ndim - 1
            cols = self.ndim - 1
            fig = plt.figure()
            gs = fig.add_gridSpec(rows, cols,left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
            for i in range(self.ndim - 1):
                for j in range(i+1,self.ndim - 1):
                    ax = fig.add_subplot(gs[i,j])
                    #plot the data points
                    sns.scatterplot(self.data[:,i], self.data[:,j], ax=ax)
                    if plot_input:
                        ax.errorbar(x=self.data[:,i], y=self.data[:,j],
                                    xerr=self.sigma[:,i], yerr=self.sigma[:,j])
            
                    if plot_fitted:
                        ax.errorbar(x=fitted_mean[:,i], y=fitted_mean[:,j],
                                    xerr=fitted_sigma[:,i], yerr=fitted_sigma[:,j])
                    
                    mu_post = self.trace.posterior["mu"].mean(axis=(i, j)).data
    
                    sigma_post = self.trace.posterior["cov"].mean(axis=(i, j)).data
                    
                    var_post, U_post = np.linalg.eig(sigma_post)
                    angle_post = 180.0 / np.pi * np.arccos(np.abs(U_post[0, 0]))
                    
                    e_post = Ellipse(
                        mu_post,
                        2 * np.sqrt(5.991 * var_post[0]),
                        2 * np.sqrt(5.991 * var_post[1]),
                        angle=angle_post,
                    )
                    e_post.set_alpha(0.5)
                    e_post.set_facecolor(blue)
                    e_post.set_zorder(10)
                    ax.add_artist(e_post)
                    
        else:
            raise ValueError("Ndim is either less than 2 or is not an integer!")
            
        if isinstance(plotfile, str):
            plt.save(plotfile)
        elif not show:
            raise TypeError("plotfile must be a string")
        if show:
            plt.show()
        elif plotfile is not None:
            plt.close()


            

            
class HierarchicalRobustBayesianCorrelation(HierarchicalBayesianCorrelation):
    """A class to infer robust Bayesian correlation coefficients for uncertain multidimensional data

    Parameters
    ----------
    data : float, (n_points, n_dim) array_like
        The multidimensional dataset to infer correlations on.
    sigma: float, (n_points, n_dim) array_like
        The uncertainties of the multidimensional dataset to infer correlations on.
    mu_prior : length-2 or (2, ndim) iterable of floats, optional, default (0., 1000.)
        The mean and standard deviation of the Gaussian prior on the multivariate t distribution
    sigma_prior : scalar or (ndim) iterable of floats, optional, default 200.
        The prior on the scale parameter (beta) of the half-Cauchy prior on the standard deviations of the multivariate t distribution

    Attributes
    ----------
    None

    Methods
    --------
    fit : Fit the data assuming they are drawn from a multivariate Normal distribution

    summarise : summarise the results of the fit

    plot_trace : plot the trace and marginal distributions of the trace

    plot_data : plot the data overlaid with the ellipse described by the inferred correlated multivariate Normal

    plot_corner : plot the 1D and 2D marginal distributions of the inferred parameters.

    Examples
    --------

    Creating an instance is as simple as 

    >>> import pybaycor as pbc
    >>> bc = pbc.BayesianCorrelation(data=data)

    Once you have created the instances, the fit is run with 

    >>> bc.fit()

    or you can modify the length of burn-in and number of steps with

    >>> bc.fit(steps=2000, tune=2000)

    Once you are happy with the fit, you can get a tabular summary with

    >>> summary = bc.summarise()

    and visual summaries with

    >>> bc.plot_trace()
    >>> bc.plot_corner()
    >>> bc,plot_data()
    """
    def __init__(self, data, sigma, mu_prior=[0.0,1000.], sigma_prior=200.):

        self.fitted=False
        if np.any(sigma <=0.):
            raise ValueError("Uncertainties must be positive real numbers!")
        self.plot_trace_vars = ['mu', "nu", "chol_corr"]
        if data is None:
            raise ValueError("Either data must be given as input, or x and y")
        else:
            self.ndim = data.shape[1]
            self.npoints = data.shape[0]
            self.data = data
            if data.shape != sigma.shape:
                raise RuntimeError("data and sigma must have the same shape!")
            self.sigma = sigma

        self.model = pm.Model()
        with self.model:
            #we put weakly informative hyperpriors on the means and standard deviations of the multivariate normal distribution
            mu = pm.Normal("mu", mu=mu_prior[0], sigma=mu_prior[1], shape=self.ndim)
            sigma = pm.HalfCauchy.dist(sigma_prior)
            #and a hyperprior on the covariance matrix which weakly penalises strong correlations
            chol, corr, stds = pm.LKJCholeskyCov("chol", n=self.ndim, eta=2.0, sd_dist=sigma, compute_corr=True)
            #the hyperprior gives us the Cholesky Decomposition of the covariance matrix, so for completeness we can calculate that determinisitically
            cov = pm.Deterministic("cov", chol.dot(chol.T))

            nuMinusOne = pm.Exponential('nu-1', lam=1./29.)
            nu = pm.Deterministic('nu', nuMinusOne + 1)

            #and now we can construct our multivariate t distribituions to complete the prior
            prior = pm.MvStudentT('vals', nu = nu, mu=mu, chol=chol, shape=(self.npoints,self.ndim)) #, observed=self.data)
            #print(prior)
            #help(prior)
            mu1s = prior[:,0]

            #Finally, we need to define our data
            for i in range(self.ndim):
                pm.Normal("data_"+str(i), mu=prior[:,i], sigma = self.sigma[:,i], observed=self.data[:,i])

            
