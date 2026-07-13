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
import json
import glob
import pickle
import nibabel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, plot_roi


# Global parameters
REGEX = "./data/*/outputs/bestk/ll.json"
ATLAS = {}
for name in ("CoCoMac", "CIVMR", "DictLearn"):
    ATLAS[name] = f"./data/{name}/atlas.nii.gz"
ANATFILE = "./data/anat.nii.gz"


# Organize models associated lls
lls, templates, components = [], [], []
for path in glob.glob(REGEX):
    split = path.split(os.sep)
    with open(path, "rt") as of:
        ll = json.load(of)
    n_components = len(ll)
    templates.extend([split[-4]] * n_components)
    components.extend(list(ll.keys())) 
    lls.extend(list(ll.values()))
data = {"ll": lls, "template": templates, "k": components}
df = pd.DataFrame.from_dict(data)
print(df)
df["ll"] = df.groupby("template")["ll"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()))
df["norm ll"] = df["ll"]


# Display lls wrt atlases
sns.set_theme(style="whitegrid")
plt.rcParams["mathtext.fontset"] = "stix"
fig = plt.figure(figsize=(23, 10))
spec = fig.add_gridspec(3, 3, left=0.05, right=0.95, hspace=0.05,
                        wspace=0.15, top=0.95, bottom=0.05)
ax1 = fig.add_subplot(spec[:, 1:])
line = sns.lineplot(
    data=df, palette="tab10", linewidth=2.5, x="k", y="norm ll",
    hue="template", marker="o", markersize=13, ax=ax1,
    hue_order=sorted(ATLAS.keys()))
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])
sns.move_legend(ax1, "lower center", bbox_to_anchor=(.5, -0.15), ncol=3,
                title=None, frameon=False, prop={"weight": "bold", "size": 18})
ax1.grid(False)
ax1.axvline(x=2, color="orange", linestyle="--", linewidth=3)
ax1.axvline(x=4, color="green", linestyle="--", linewidth=3)
ax1.axvline(x=5, color="b", linestyle="--", linewidth=3)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.set_xlabel("k", fontsize=16)
ax1.set_ylabel(r"$\mathcal{L}_{0-1}$", fontsize=16)
for idx, (name, path) in enumerate(ATLAS.items()):
    im = nibabel.load(path)
    ax = fig.add_subplot(spec[idx, 0])
    if im.ndim == 4:
        plot_prob_atlas(path, bg_img=ANATFILE, draw_cross=False, title=name,
                        axes=ax, cmap="Paired")
    else:
        plot_roi(path, bg_img=ANATFILE, draw_cross=False, title=name,
                 axes=ax, cmap="Paired")
plt.show()
