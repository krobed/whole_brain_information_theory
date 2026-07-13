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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Global parameters
REGEX = "./data/CoCoMac/outputs/predictions/svcrbf_*_importances.tsv"

# Get data
data = {}
xmin, xmax = (0, 0)
for path in glob.glob(REGEX):
    split = os.path.basename(path).split("_")
    name = split[1]
    data[name] = pd.read_csv(path, sep="\t")
    arr = data[name].values
    if arr.min() < xmin:
        xmin = arr.min()
    if arr.max() > xmax:
        xmax = arr.max()


# Display importances
fig, axs = plt.subplots(3)
for idx, (name, df) in enumerate(data.items()):
    print(name, df)
    df.columns = [col.replace("network", "BN") for col in df.columns]
    data = df.values.T
    sorted_idx = np.mean(data, axis=1).argsort()
    axs[idx].boxplot(
        data[sorted_idx].T, vert=False, labels=df.columns[sorted_idx])
    axs[idx].set_title(name, fontweight="bold")
    axs[idx].spines["right"].set_visible(False)
    axs[idx].spines["top"].set_visible(False)
    axs[idx].set_xlim(xmin, xmax)
fig.tight_layout()
plt.show()

