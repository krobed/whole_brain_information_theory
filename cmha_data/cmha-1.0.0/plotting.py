# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2024
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import pandas as pd
import numpy as np
from nilearn import plotting
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from utils import _get_civmr_json_and_transform


def plot_stats(data, stats, alpha, show=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    sns.histplot(data, x="activity", hue="condition", element="step",
                 kde=True, ax=ax2)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    sns.boxplot(data=data, x="condition", y="activity", ax=ax1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    selection = stats[stats["p-corr"] < alpha]
    pairs = [(row.A, row.B) for _, row in selection.iterrows()]
    pvalues = selection["p-corr"].values
    if len(pairs) > 0:
        annotator = Annotator(ax1, pairs, data=data, x="condition",
                              y="activity")
        annotator.set_pvalues(pvalues)
        annotator.annotate()
    if show:
        plt.show()
    return fig


def plot_networks(model, k, atlas_centroids, show=True):
    plotting.glass_brain._get_json_and_transform = _get_civmr_json_and_transform
    n_rows, rest = divmod(k, 2)
    if rest != 0:
        n_rows += 1
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 20))
    flat_axs = axs.flat
    for ax in flat_axs:
        ax.axis("off")
    for idx in range(k):
        model.plot(atlas_centroids, clusterID=idx, title="", ax=flat_axs[idx],
                   fig=fig)
    if show:
        plt.show()
    return fig


def plot_networks_activity(model, G, k, labels, show=True):
    fig, axs = plt.subplots(k, figsize=(15, 20))
    all_df = {}
    for idx in range(k):
        data = [sigma[idx, idx] for sigma in G]
        df = pd.DataFrame.from_dict({"activity": data, "condition": labels})
        all_df[idx] = df
        sns.violinplot(x="condition", y="activity", data=df, ax=axs[idx])
        axs[idx].yaxis.set_ticks_position("left")
        axs[idx].xaxis.set_ticks_position("bottom")
        if idx < (k - 1):
            axs[idx].get_xaxis().set_visible(False)
    if show:
        plt.show()
    return all_df
    

def plot_mosaic(x, size=5, dim=(82, 82), show=True):
    fig, axs = plt.subplots(size, size, figsize=(5, 5))
    for idx, ax in enumerate(axs.flat):
        im = x[idx]
        ax.imshow(im, cmap="jet", vmin=0)
        ax.axis("off")
    if show:
        plt.show()
    return fig
