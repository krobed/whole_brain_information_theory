# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from utils import extract_centroids, get_mapping, cloud_distance


# Global parameters
REGEX = "./data/*/outputs/stats/stat_component*.png"
ATLAS = {}
for name in ("CoCoMac", "CIVMR", "DictLearn"):
    ATLAS[name] = f"./data/{name}/atlas.nii.gz"


# Organize models associated lls
data = {}
for path in glob.glob(REGEX):
    split = path.split(os.sep)
    key = split[-4]
    k = split[-1].split("_")[1].split(".")[0].replace("component", "")
    k = int(k)
    _path = glob.glob(os.path.join(os.path.dirname(os.path.dirname(path)),
                                   "components{}_k*.png".format(k)))[0]
    data.setdefault(key, {})[k] = (
        os.path.abspath(_path), os.path.abspath(path))
print(data)


# Order components using dMoC
clouds = {}
for name, atlas in ATLAS.items():
    atlas_centroids = extract_centroids(atlas, affine=None)
    path = data[name][1][0]
    basename = os.path.basename(path).split("_")[1].replace(".png", ".pkl")
    basename = "fmha_" + basename
    path = os.path.join(os.path.dirname(path), basename)
    with open(path, "rb") as of:
        model = pickle.load(of)
    for idx in range(model.k):
        ii = np.where(model.W[:, idx] != 0)[0]
        clouds.setdefault(name, []).append(atlas_centroids[ii])
mapping = get_mapping(clouds, cloud_distance, flatten=False)


# Display all components
fig = plt.figure(figsize=(50, 20))
max_components = max([len(item) for item in mapping.values()])
spec = fig.add_gridspec(3, max_components, left=0.05, right=0.95, hspace=0.05,
                        wspace=0.05, top=0.95, bottom=0.05)
for idx1, name in enumerate(["CoCoMac", "DictLearn", "CIVMR"]):
    n_components = len(data[name])
    _data = data[name]
    order = [mapping[name][idx] + 1 for idx in range(n_components)]
    for idx2, k in enumerate(order):
        ax = fig.add_subplot(spec[idx1, idx2])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title("BN{}".format(k), fontweight="bold", loc="left",
                     fontsize=15)
        path, _ = _data[k]
        with cbook.get_sample_data(path) as of:
            image = plt.imread(of)
        ax.imshow(image)

plt.show()
