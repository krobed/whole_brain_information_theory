# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2024
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import warnings 
warnings.filterwarnings("ignore")
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from MHA import MHA
from utils import extract_centroids
import matplotlib.pyplot as plt
from plotting import plot_networks, plot_networks_activity, plot_mosaic



def exp_main(meta_file, data_file, atlas_file, outdir, k=5, diag=False,
             early_stopping=False, exclude=None, trange=None):
    """ MHA experiment.

    Parameters
    ----------
    meta_file: str
        path to the 'participants.tsv' file.
    data_file: str
        path to the file that containes the ROI averaged timeseries.
    atlas_file: str
        path to the template defining the ROIs.
    outdir: str
        the destination folder.
    k: int, default 5
        the number of networks.
    diag: bool, default False
        should latent variables have diagonal covariance structure.
    early_stopping: bool, default False
        should we stop the code after training the model.
    exclude: list of str, default None
        the list of subjects to be excluded.
    trange: 2-uplet, default None
        contains the start and stop timeserie cut locations.

    References
    ----------
    .. [1] A Unified Probabilistic Model for Learning Latent Factors and
    Their Connectivities from High-Dimensional Data, UAI, 2018.
    """

    #########################################################################
    # Load data
    # ---------
    #
    # Load the preprocessed dataset

    print("Loading data...")
    timeseries = np.load(data_file)
    if trange is not None:
        timeseries = timeseries[:, trange[0]: trange[1]]
        print("- troncated timeseries: {0}".format(timeseries.shape))
    df = pd.read_csv(meta_file, sep="\t")
    if exclude is not None:
        train_mask = ~df["sub"].isin(exclude).values
        test_mask = df["sub"].isin(exclude).values
        timeseries_exclude = timeseries[test_mask]
        timeseries_include = timeseries[train_mask]
        # df = df[~df["sub"].isin(exclude)]
    else:
        timeseries_exclude = None
        timeseries_include = timeseries
    conditions = df["cond"].values
    n_conditions = len(np.unique(conditions))
    n_rois = timeseries.shape[-1]
    subjects = df["sub"].values
    atlas_centroids = extract_centroids(atlas_file, affine=None)
    print("- ROIS:", atlas_centroids.shape)
    print("- timeseries: {0}".format(timeseries_include.shape))
    print("- subjects: {0}".format(len(subjects)))
    print("- conditions: {0} - {1}".format(
        n_conditions, np.unique(conditions)))

    #########################################################################
    # Projection
    # ----------
    #
    # Perform constrain projection using MHA.

    name = "mha"
    if not diag:
        name = "f" + name
    model_file = os.path.join(outdir, "{}_k{}.pkl".format(name, k))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isfile(model_file):
        for data in timeseries_include:
            if np.isnan(data).sum() != 0:
                raise ValueError("Please clean your input timeseries first!")
        shat = [np.cov(scale(data).T) for data in timeseries_include
                if np.isnan(data).sum() == 0]
        print("- collected data for {} samples".format(len(shat)))
        model = MHA(Shat=shat, k=k, diagG=diag)
        model.fit(alphaArmijo=1)
        with open(model_file, "wb") as of:
            pickle.dump(model, of)
    else:
        with open(model_file, "rb") as of:
            model = pickle.load(of)
    print("- log likelihood:", model.logLik)
    if early_stopping:
        return model, timeseries_exclude
    fig = plot_networks(model, k, atlas_centroids, show=False)
    fig.savefig(os.path.join(outdir, "components_k{}.png".format(k)))
    for cidx in range(k):
        fig = plt.figure()
        model.plot(atlas_centroids, clusterID=cidx, title="", fig=fig)
        fig.savefig(os.path.join(outdir, "components{}_k{}.png".format(
            cidx + 1, k)))
    print("- networks distribution:", np.sum(model.W > 0, axis=0))
    if exclude is not None:
        for data in timeseries:
            if np.isnan(data).sum() != 0:
                raise ValueError("Please clean your input timeseries first!")
        shat = [np.cov(scale(data).T) for data in timeseries
                if np.isnan(data).sum() == 0]
        nsub = len(shat)
        if model.diagG:
            G = [(np.diag(np.diag(model.W.T.dot(shat[i]).dot(model.W))) -
                  np.eye(model.k)) for i in range(nsub)]
        else:
            G = [model.W.T.dot(shat[i]).dot(model.W) - np.eye(model.k)
                 for i in range(nsub)]
    else:
        G = model.G
    data = plot_networks_activity(model, G, k, conditions, show=False)
    stat_file = os.path.join(outdir, "{}_stats_k{}.pkl".format(name, k))
    if not os.path.isfile(stat_file):
        with open(stat_file, "wb") as of:
            pickle.dump(data, of)

    #########################################################################
    # Connectivity
    # ------------
    #
    # Study connectivity between these networks.

    conn = np.asarray(model.G)
    fig = plot_mosaic(conn, size=5, dim=(5, 5), show=False)
    fig.savefig(os.path.join(outdir, "g_k{}.png".format(k))) 
    conn = up_tri(conn)
    print("- network connectivity:", conn.shape)


def up_tri(x, k=0):
    """ Return the vectorized upper triangular elements only.
    """
    size = x.shape[-1]
    indices = np.triu_indices(size, k=k)
    return x[:, indices[0], indices[1]] 


if __name__ == "__main__":

    import fire
    fire.Fire(exp_main)
