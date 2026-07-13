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
import pandas as pd


# Global parameters
REGEX = "./data/*/outputs/predictions/*_summary.tsv"


# Get data
data = {}
for path in glob.glob(REGEX):
    name = path.split(os.sep)[-4]
    split = os.path.basename(path).split("_")
    method = split[0]
    df = pd.read_csv(path, sep="\t")
    print(f"{method} - {name}")
    print(df)
