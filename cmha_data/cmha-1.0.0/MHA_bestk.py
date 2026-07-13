# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022-2024
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Imports
import os
import json
import numpy as np
from sklearn.preprocessing import scale
from MHA_fit import exp_main


def multi_exp_main(meta_file, data_file, atlas_file, outdir, exclude,
                   all_k=(2, 11), diag=False):
    """ MHA multi experiments.

    Parameters
    ----------
    meta_file: str
        path to the 'participants.tsv' file.
    datadir: str,
        path to the file that containes the ROI averaged timeseries.
    atlas: str
        path to the template defining the ROIs.
    outdir: str
        the destination folder.
    exclude: list of str
        the list of subjects to be excluded.
    all_k: 2-uplet, default (2, 9)
        the number of networks.
    diag: bool, default False
        should latent variables have diagonal covariance structure.

    References
    ----------
    .. [1] A Unified Probabilistic Model for Learning Latent Factors and
    Their Connectivities from High-Dimensional Data, UAI, 2018.
    """
    ll = {}
    print("=" * 15)
    print(f"data: {data_file}")
    print(f"atlas: {atlas_file}")
    print(f"outdir: {outdir}")
    print("=" * 15)
    for k in range(*all_k):
        print(f"- k: {k}")
        model, timeseries = exp_main(
            meta_file=meta_file, data_file=data_file, atlas_file=atlas_file,
            outdir=outdir, k=k, diag=diag, early_stopping=True,
            exclude=exclude)
        print(f"- timeseries [validation]: {timeseries.shape}")
        for data in timeseries:
            if np.isnan(data).sum() != 0:
                raise ValueError("Please clean your input timeseries first!")
        shat = [np.cov(scale(data).T) for data in timeseries
                if np.isnan(data).sum() == 0]
        all_ll = model.get_ll(shat)
        print(f"- LL [validation]: {all_ll}")
        ll[k] = float(np.mean(all_ll))
        print(f"- mean log likelihood: {ll[k]}")
    with open(os.path.join(outdir, "ll.json"), "wt") as of:
        json.dump(ll, of, indent=2)    
    

if __name__ == "__main__":

    import fire
    fire.Fire(multi_exp_main)
