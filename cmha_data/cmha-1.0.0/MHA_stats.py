# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2024
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import os
import pickle
import pandas as pd
import pingouin as pg
from plotting import plot_stats


def exp_main(meta_file, model_file, outdir, alpha=0.01):
    """ MHA statistic experiments.

    Parameters
    ----------
    metafile: str
        path to the 'participants.tsv' file.
    model_file: str
        path to the file that containes the fitted model.
    outdir: str
        the destination folder.
    alpha: float, default 0.01
        the significance level.

    References
    ----------
    .. [1] A Unified Probabilistic Model for Learning Latent Factors and
    Their Connectivities from High-Dimensional Data, UAI, 2018.
    """
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    df = pd.read_csv(meta_file, sep="\t")
    subjects = df["sub"]
    with open(model_file, "rb") as of:
        data = pickle.load(of)
    for k, df in data.items():
        print("- component {}:".format(k))
        df["sub"] = subjects
        print(pg.normality(data=df, dv="activity", group="condition",
                           method="shapiro", alpha=0.05))
        print(pg.homoscedasticity(data=df, dv="activity", group="condition",
                                  method="levene", alpha=0.05))
        stats = pg.pairwise_ttests(
            data=df, dv="activity", between="condition", parametric=False,
            subject="sub", alpha=alpha, alternative="two-sided",
            padjust="fdr_by").round(6)
        stats = stats.sort_values("p-corr")
        print(stats)
        stats.to_csv(os.path.join(outdir, "stat_component{}.tsv".format(k + 1)),
                     sep="\t", index=False)
        fig = plot_stats(data=df, stats=stats, alpha=alpha, show=False)
        fig.savefig(os.path.join(outdir, "stat_component{}.png".format(k + 1))) 


if __name__ == "__main__":

    import fire
    fire.Fire(exp_main)
