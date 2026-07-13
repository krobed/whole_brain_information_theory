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
import glob
import copy
import pickle
import nibabel
import numpy as np
import pandas as pd
from utils import extract_centroids


# Global parameters
REGEX = "./data/CoCoMac/outputs/stats/stat_component*.png"
ATLAS = f"./data/CoCoMac/atlas.nii.gz"
LABEL = "./data/CoCoMac/labels.tsv"


# Organize models associated lls
name = "CoCoMac"
data = {}
for path in glob.glob(REGEX):
    split = path.split(os.sep)
    key = split[-4]
    k = split[-1].split("_")[1].split(".")[0].replace("component", "")
    k = int(k)
    _path = glob.glob(os.path.join(os.path.dirname(os.path.dirname(path)),
                                   f"components{k}_k*.png"))[0]
    data.setdefault(key, {})[k] = (_path, path)
print(data)


# List all components
clouds = {}
clouds_meta = {}
clouds_weights = {}
atlas = ATLAS
atlas_centroids = extract_centroids(atlas, affine=None)
im = nibabel.load(atlas)
arr = im.get_fdata()
background_label = 0
label_vals = np.asarray(
    sorted(set(np.unique(arr)) - {background_label}))
path = data[name][1][0]
basename = os.path.basename(path).split("_")[1].replace(".png", ".pkl")
basename = "fmha_" + basename
path = os.path.join(os.path.dirname(path), basename)
with open(path, "rb") as of:
    model = pickle.load(of)
for idx in range(model.k):
    ii = np.where(model.W[:, idx] != 0)[0]
    clouds.setdefault(name, []).append(atlas_centroids[ii])
    clouds_meta.setdefault(name, []).append(label_vals[ii])
    clouds_weights.setdefault(name, []).append(model.W[ii, idx])


# Load CoCoMac labels
labels = pd.read_csv(LABEL, sep="\t")
print(labels)


# Check CoCoMac atlas GNW
n_components = len(clouds[name])
for k in range(n_components):
    data = {}
    n_rois = len(clouds_meta[name][k])
    print(f"Number of ROIs in BN{k + 1}: {n_rois}")
    for point, idx, weight in zip(clouds[name][k],
                                  clouds_meta[name][k],
                                  clouds_weights[name][k]):
        data.setdefault("x", []).append(point[0])
        data.setdefault("y", []).append(point[1])
        data.setdefault("z", []).append(point[2])
        data.setdefault("index", []).append(idx)
        data.setdefault("weight", []).append(weight)
        row = labels[labels["label"] == idx]
        data.setdefault("label", []).append(row.short_name.item())
        data.setdefault("hemi", []).append(row.hemi.item().lower())
        data.setdefault("location", []).append(row.location.item())
        data.setdefault("name", []).append(row.name.item())
    df = pd.DataFrame.from_dict(data)
    print(df)
