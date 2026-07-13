### Python implementation of MHA algorithm
### From: https://github.com/piomonti/MHA

# Imports
from MHA_backend import * 
from nilearn import plotting
import numpy as np 


class MHA(object):
    """ Class for MHA object
    """
    def __init__(self, Shat, k, diagG=False):
        self.Shat = Shat
        self.k = k
        self.diagG = diagG 
        self.W = None
        self.G = None
        self.iter = None
        self.logLik = None

    def __repr__(self):
        mes = 'MHA object\n'
        mes += 'Number of subjects: ' + str(len(self.Shat)) + '\n'
        mes += 'Latent variable dim: ' + str(self.k) + '\n'
        if self.diagG:
            mes += 'Diagonal latent variable covariance'
        else:
            mes += 'Full (ie non-diagonal) latent variable covariance'
        return mes

    def fit(self, lagParam=1, tol=.01, alphaArmijo=0.5, maxIter=1000):
        """ Estimate loading matrix and latent variable covariances.
        """
        res = nonNegativeCovFactor_LagrangeMult(
            Shat=self.Shat, k=self.k, diagG=self.diagG, lagParam=lagParam,
            tol=tol, alphaArmijo=alphaArmijo, maxIter=maxIter)
        self.W = res['W']
        self.G = res['G']
        self.iter = res['iter']
        self.logLik = res['logLik']

    def transform(self, Xnew):
        """ Apply projection matrix, W, to new data.

        Parameters
        ----------
        Xnew: list of numpy array
            each entry should be an n by p array of n observations for p
            random variables.
        """
        ProjXnew = [X.dot(self.W) for X in Xnew]
        return ProjXnew

    def get_ll(self, Shat):
        """ Get the log-likelihood of unseen data.

        Parameters
        ----------
        Shat: list of numpy array
            each entry should be an n by p array of n observations for p
            random variables.
        """
        p = Shat[0].shape[0] 
        nSub = len(Shat)
        if self.diagG:
            G = [np.diag(np.diag(self.W.T.dot(Shat[i]).dot(self.W))) - np.eye(self.k)
                 for i in range(nSub)]
        else:
            G = [self.W.T.dot(Shat[i]).dot(self.W) - np.eye(self.k)
                 for i in range(nSub)]
        PresHat, logLik = [], []
        for i in range(nSub):
            PresHat.append(
                np.eye(p) - self.W.dot(G[i]).dot(np.linalg.pinv(
                    G[i] + np.eye(self.k))).dot(self.W.T))
            logLik.append(
                0.5 * np.log(np.linalg.det(PresHat[i])) - 0.5 *
                np.sum(np.diag(Shat[i].dot(PresHat[i]))))
        return logLik

    def plot(self, ROIcoord, clusterID, title, ax=None, fig=None):
        """ Display estimated networks

        Parameters
        ----------
        ROIcoord: array
            ROI template coordinates.
        clusterID: int
            which network should we plot.
        """
        ii = np.where(self.W[:, clusterID] != 0)[0]
        # this is just a place holder, we will not plot any of it!
        RandomMat = np.cov(np.random.random((10, len(ii))).T)
        # we just plot the result
        plotting.plot_connectome(
            RandomMat, ROIcoord[ii, :], node_color='black', annotate=False,
            display_mode='ortho', edge_kwargs={'alpha': 0}, node_size=50,
            title=title, axes=ax, figure=fig)

