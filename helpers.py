from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hoi.metrics import RedundancyphiID, AtomsPhiID
from hoi.utils import get_nbest_mult
from itertools import combinations
from tqdm import tqdm
from hoi.metrics import DTC, TC, GradientOinfo, Oinfo, Sinfo


def get_Oinfo(subject_ts, minsize=2,maxsize=2):
    model = Oinfo(subject_ts)

    oi_values = model.fit(minsize=minsize, maxsize=maxsize)
    return oi_values,model

def phiid_byatoms(subject_ts, atoms=['sts'], interaction_size=2):
    model_group = AtomsPhiID(subject_ts, verbose=False)
    res = model_group.fit(atoms=atoms, minsize=interaction_size, maxsize=interaction_size)

    return res, model_group

def redundancy_phiid(subject_ts, interaction_size):
        
    # Slice data using the combination indices (cast tuple to list for proper indexing)
    model_group = RedundancyphiID(subject_ts, verbose=False)
    res = model_group.fit(minsize=interaction_size, maxsize=interaction_size)
    # Save results using the combination tuple as the key

    return res, model_group

def plot_2d_phiid(results,model,n_features):
    matrix_to_plot = np.zeros((n_features, n_features))
    for i, tup in enumerate(model.get_combinations(minsize=2, maxsize=2)[0]):
        m, g = tup
        matrix_to_plot[m, g] = results[i][0]

    plt.imshow(matrix_to_plot + matrix_to_plot.T, aspect="auto", interpolation="none")
    plt.colorbar()
    plt.show()
    plt.close()

def fit_model(model, max_size=2):
    hoi = model.fit(method="gc", minsize=2, maxsize=max_size)
    return hoi


import hoi

def compute_phiid_measures(subject_ts, ATOM_INDEX, MEASURES, n_features=None):
    """
    subject_ts : (T, N)
    returns    : dict {measure_name: scalar}
    """

    phiid = hoi.metrics.AtomsPhiID(subject_ts, verbose= False)

    # 1. Compute ALL atoms once
    atom_values = {}
    for atom in ATOM_INDEX:
        # Enforce flat 1D arrays immediately using .ravel()
        atom_values[atom] = phiid.fit(
            method="gc",          # Gaussian copula entropy
            minsize=2, maxsize=2, # Only pairwise combinations allowed
            atoms=[atom]
        ).ravel()

    # Dynamic length allocation (avoids hardcoded "rtr")
    first_key = list(atom_values.keys())[0]
    N = len(atom_values[first_key])
    
    # 2. Extract and prepare coordinate indices for vectorization
    multiplets = np.array(phiid.multiplets)  # Shape: (N, 2)
    i_idx = multiplets[:, 0]
    j_idx = multiplets[:, 1]

    # Automatically infer matrix dimensions if not provided
    if n_features is None:
        n_features = int(multiplets.max() + 1)

    results = {}
    interaction_matrix = {}

    # 3. Compute measures and assign to symmetric matrices using advanced indexing
    for measure, spec in MEASURES.items():
        M = np.zeros(N)

        for atom in spec["add"]:
            M += atom_values[atom]

        for atom in spec["sub"]:
            M -= atom_values[atom]

        results[measure] = M
        
        # Initialize the square matrix
        mat = np.zeros((n_features, n_features))
        
        # Vectorized symmetric assignment (no Python loop needed)
        mat[i_idx, j_idx] = M
        mat[j_idx, i_idx] = M
        
        interaction_matrix[measure] = mat

    return results, interaction_matrix
