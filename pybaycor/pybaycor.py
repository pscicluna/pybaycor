import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import arviz as az


class BayesianCorrelation():

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
        with self.model:
            self.trace = pm.sample(
                steps, tune=tune, target_accept=0.9, compute_convergence_checks=False,return_inferencedata=True
            )
            self.fitted=True
        if summarise:
            self.summary = az.summary(self.trace, var_names=["~chol"], round_to=2)
            #self.rho = [self.summary['hdi_3%'][chol_corr[1,0]],self.summary['mean'][chol_corr[1,0]],self.summary['hdi_97%'][chol_corr[1,0]]]
            print(self.summary)

    def summarise(self):
        self.summary = az.summary(self.trace, var_names=["~chol"], round_to=2)
        print(self.summary)
        

    def plot_trace(self,plotfile=None, show=False):
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
        elif plotfile is not None:
            plt.close()
        #should this also return the ax?
            

    def plot_data(self,plotfile=None, show=None):
        #Currently only supports 2D correlations
        if self.ndim != 2:
            raise NotImplementedError("This routine doesn't support plotting correlations in more than 2 dimensions yet!")
        if not self.fitted:
            pass #raise an error here
        blue, _, red, *_ = sns.color_palette()
        f, ax = plt.subplots(1, 1, figsize=(5, 4))#, gridspec_kw=dict(width_ratios=[4, 3]))

        #ax.plot(data[:,0], data[:,1], 'o')
        sns.scatterplot(x=self.data[:,0], y=self.data[:,1])

        mu_post = self.trace.posterior["mu"].mean(axis=(0, 1)).data
        #(1 - mu_post / mu).round(2)
    
        sigma_post = self.trace.posterior["cov"].mean(axis=(0, 1)).data
        #(1 - sigma_post / sigma).round(2)

    
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

        if isinstance(plotfile, str):
            plt.save(plotfile)
        if show:
            plt.show()
        elif plotfile is not None:
            plt.close()
        
        #pass



class RobustBayesianCorrelation(BayesianCorrelation):
    def __init__(self,data=None, x=None, y=None, ndim = None, mu_prior=[0.0,1000.], sigma_prior=200.):
        #if ndim is None:
        self.fitted=False
        self.plot_trace_vars = ['mu', "chol_corr", "nu"] #, "~nu-1", "~cov", "~chol_stds", "~chol"]
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

            #and now we can put our observed values into a multivariate T distribution to complete the model
            vals = pm.MvStudentT('vals', nu = nu, mu=mu, chol=chol, observed=self.data)

        print("mu shape: ",mu.dshape)
        help(vals)
        print("vals shape: ",vals.shape)


class HierarchicalBayesianCorrelation(BayesianCorrelation):
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

            print("mu shape: ",mu.dshape)
            print("chol shape: ",chol.shape)
            print("chol size: ",chol.size)
            #help(self.npoints)

            #and now we can construct our multivariate normal to complete the prior
            prior = pm.MvNormal('vals', mu=mu, chol=chol, shape=(self.npoints,self.ndim)) #, observed=self.data)
            print(prior)
            help(prior)
            mu1s = prior[:,0]

            #Finally, we need to define our data
            for i in range(self.ndim):
                pm.Normal("data_"+str(i), mu=prior[:,i], sigma = self.sigma[:,i], observed=self.data[:,i])

            

            
class HierarchicalRobustBayesianCorrelation(BayesianCorrelation):
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

            nuMinusOne = pm.Exponential('nu-1', lam=1./29.)
            nu = pm.Deterministic('nu', nuMinusOne + 1)

            print("mu shape: ",mu.dshape)
            print("chol shape: ",chol.shape)
            print("chol size: ",chol.size)
            #help(self.npoints)

            #and now we can construct our multivariate normal to complete the prior
            prior = pm.MvStudentT('vals', nu = nu, mu=mu, chol=chol, shape=(self.npoints,self.ndim)) #, observed=self.data)
            print(prior)
            help(prior)
            mu1s = prior[:,0]

            #Finally, we need to define our data
            for i in range(self.ndim):
                pm.Normal("data_"+str(i), mu=prior[:,i], sigma = self.sigma[:,i], observed=self.data[:,i])

            
