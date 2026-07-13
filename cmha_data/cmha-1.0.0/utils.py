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
import nibabel
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment as linear_assignment


def extract_centroids(atlas, background_label=0, affine=None):
    """ Extracts the centroids of the atlas regions.

    Parameters
    ----------
    atlas: str
        region definitions, as one image of labels (3d or 4d image).
    background_label: int, optional
        label used in to represent background.
    affine: array (4, 4), optional
        transform the centroid by applying this transformation.

    Returns
    -------
    coords: array (n_rois, 3)
        the ROI center coordinates in world space.
    """
    im = nibabel.load(atlas)
    arr = im.get_data()
    if arr.ndim == 3:
        labels = sorted(set(np.unique(arr)) - {background_label})
        centroids = [np.argwhere(arr == lab).mean(axis=0) for lab in labels]
    else:
        arr = np.transpose(arr, (3, 0, 1, 2))
        centroids = [np.argwhere(tarr != background_label).mean(axis=0)
                     for tarr in arr]
    if affine is None:
        affine = np.eye(4)
    trf = np.dot(im.affine, affine)
    return apply_affine(np.asarray(centroids), trf)


def apply_affine(coords, affine):
    """ Apply an affine transformation on each coordiantes.

    Parameters
    ----------
    coords: array (n_points, 3)
        the coordiantes.
    affine: array (4, 4)
        an affine transformation to applied.

    Returns
    -------
    new_coords: array (n_points, 3)
        the new coordiantes.
    """
    nb_points, _ = coords.shape
    ones = np.ones((nb_points, 1), dtype=coords.dtype)
    homogenous_coords = np.concatenate((coords, ones), axis=1)
    new_coords = np.dot(affine, homogenous_coords.T).T[..., :3]
    return new_coords


def cloud_distance(sample1, sample2, metric=np.min, symetric=True):
    """ Calculate the euclidean distance for every point in sample point
    cloud to the closest point in the reference point cloud (number of
    columns must match).

    Parameters
    ----------
    sample1: array (m, n)
        a n-dim cloud of point.
    sample2: array (p, n)
        a n-dim cloud of point.
    metric: @callable
        compute min or max distances between points.
    symetric: bool, default True
        reverse the reference / sample role and average to get final distance.

    Returns
    -------
    dist: float
        the min distance.
    """
    if len(sample1) < len(sample2):
        sample = sample1
        reference = sample2
    else:
        sample = sample2
        reference = sample1
    dist = distance.cdist(sample, reference, "euclidean")
    dist = np.mean(metric(dist, axis=1))
    if symetric:
        dist_sym = distance.cdist(reference, sample, "euclidean")
        dist_sym = np.mean(metric(dist_sym, axis=1))
        dist = (dist + dist_sym) * 0.5
    return dist


def match(y1, y2, dist_fct, flatten=False, exclude=None):
    """ Align components using a linear assigment.

    Parameters
    ----------
    y1, y2: array (n_components, *)
        the features to be aligned.
    dist_fct: @callable
        a function to compute the distance between two features.
    flatten: bool, default False
        wether to flatten the features before computing the metric.
    exclude: 2-uplet with list of int, default None
        wether to exclude some features.

    Returns
    -------
    mapping: dict
        a linear mapping between the components.
    """
    n_elems = max(len(y1), len(y2))
    gain = np.ones((n_elems, n_elems), dtype=np.float32)
    gain *= np.finfo(np.float32).max
    for idx1 in range(len(y1)):
        for idx2 in range(len(y2)):
            if flatten:
                gain[idx1, idx2] = dist_fct(y1[idx1].flatten(),
                                            y2[idx2].flatten())
            else:
                gain[idx1, idx2] = dist_fct(y1[idx1], y2[idx2])
    if exclude is not None:
        gain[exclude[0], :] = np.finfo(np.float32).max
        gain[:, exclude[1]] = np.finfo(np.float32).max
    ind = np.vstack(linear_assignment(gain)).T
    return dict(ind)


def get_mapping(data, dist_fct, flatten=False):
    """ Get a mapping between components.

    Parameters
    ----------
    data: dict with sets as array (n_components, *)
        the data for the matching.
    dist_fct: @callable
        a function to compute the distance between two features.
    flatten: bool, default False
        wether to flatten the features before computing the metric.
    """
    n_components = dict((key, len(val)) for key, val in data.items())
    shift = n_components["CoCoMac"]
    cocomac_civmr_map = match(data["CoCoMac"], data["CIVMR"], dist_fct,
                              flatten=flatten)
    cocomac_dictlearn_map = match(data["CoCoMac"], data["DictLearn"], dist_fct,
                                  flatten=flatten)
    dictlearn_exclude = [cocomac_dictlearn_map[idx]
               for idx in range(n_components["CoCoMac"])]
    civmr_exclude = [cocomac_civmr_map[idx]
               for idx in range(n_components["CoCoMac"])]
    delta_dictlearn_civmr = match(
        data["DictLearn"], data["CIVMR"], dist_fct, flatten=flatten,
        exclude=(dictlearn_exclude, civmr_exclude))
    for idx in range(n_components["CoCoMac"], n_components["DictLearn"]):
        _idx = cocomac_dictlearn_map[idx]
        cocomac_civmr_map[idx] = delta_dictlearn_civmr[_idx]
    civmr_inuse = [cocomac_civmr_map[idx]
              for idx in range(n_components["DictLearn"])]
    cocomac_civmr_map[6] = list(
        set(range(n_components["CIVMR"])) - set(civmr_inuse))[0]
    mapping = {
        "CoCoMac": dict((idx, idx) for idx in range(n_components["CoCoMac"])),
        "CIVMR": cocomac_civmr_map,
        "DictLearn": cocomac_dictlearn_map}
    return mapping


def _get_civmr_json_and_transform(direction):
    """Returns the json filename and and an affine transform, which has
    been tweaked by hand to fit the CIVMR template.
    """
    from matplotlib import transforms
    direction_to_view_name = {'x': 'side',
                              'y': 'back',
                              'z': 'top',
                              'l': 'side',
                              'r': 'side'}
    direction_to_transform_params = {
        'x': [0.73, 0, 0, 0.7, -49, -19],
        'y': [0.7, 0, 0, 0.7, -28, -17],
        'z': [0.7, 0, 0, 0.72, -29, -36],
        'l': [0, 0, 0, 0, 0, 0],
        'r': [0, 0, 0, 0, 0, 0]}

    dirname = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(dirname, 'glass_brain_files')
    direction_to_filename = dict([
        (_direction, os.path.join(
            dirname,
            'brain_schematics_{0}.json'.format(view_name)))
        for _direction, view_name in direction_to_view_name.items()])

    direction_to_transforms = dict([
        (_direction, transforms.Affine2D.from_values(*params))
        for _direction, params in direction_to_transform_params.items()])

    direction_to_json_and_transform = dict([
        (_direction, (direction_to_filename[_direction],
                      direction_to_transforms[_direction]))
        for _direction in direction_to_filename])

    filename_and_transform = direction_to_json_and_transform.get(direction)

    if filename_and_transform is None:
        message = ("No glass brain view associated with direction '{0}'. "
                   "Possible directions are {1}").format(
                       direction,
                       list(direction_to_json_and_transform.keys()))
        raise ValueError(message)

    return filename_and_transform
