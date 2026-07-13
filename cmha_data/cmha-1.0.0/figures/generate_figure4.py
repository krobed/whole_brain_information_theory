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


# Display stats wrt atlases
for name, _data in data.items():
    n_components = len(_data)
    fig = plt.figure(figsize=(55, 25))
    spec = fig.add_gridspec(3, 26, left=0.05, right=0.95, hspace=0.05,
                            wspace=0.05, top=0.95, bottom=0.05)
    order = [mapping[name][idx] + 1 for idx in range(n_components)]
    for idx, k in enumerate(order):
        offset, ridx = divmod(idx, 3)
        if offset > 1:
            continue
        path1, path2 = _data[k]
        ax1 = fig.add_subplot(spec[ridx, offset * 13: offset * 13 + 4])
        ax1.set_title("BN{}".format(k), fontweight="bold",
                      loc="left", fontsize=25)
        with cbook.get_sample_data(path1) as of:
            image = plt.imread(of)
        ax1.imshow(image)
        ax1.axis("off") 
        ax2 = fig.add_subplot(spec[ridx, offset * 13 + 5: offset * 13 + 13])
        with cbook.get_sample_data(path2) as of:
            image = plt.imread(of)
        ax2.imshow(image)
        ax2.axis("off")
plt.show()
