# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2024
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import warnings 
warnings.filterwarnings("ignore")
import os
import pickle
import pandas as pd
import numpy as np
import mord
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


def exp_main(meta_file, model_file, outdir, exclude, full=False):
    """ MHA acquisition setting prediction experiment.

    Parameters
    ----------
    metafile: str
        path to the 'participants.tsv' file.
    model_file: str
        path to the file that containes the fitted model.
    outdir: str
        the destination folder.
    exclude: list of str
        the list of subjects to be excluded.
    full: bool, default False
        wether to use the diagonal or the upper triangular elements.
    """
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    df = pd.read_csv(meta_file, sep="\t")
    subjects = df["sub"]
    train_mask = ~df["sub"].isin(exclude).values
    test_mask = df["sub"].isin(exclude).values
    print(df)
    if full:
        with open(model_file.replace("_stats", ""), "rb") as of:
            model = pickle.load(of)
        X = []
        for mat in model.G:
            X.append(mat[np.triu_indices(len(mat), k=0)])
        X = np.asarray(X)
    else:
        with open(model_file, "rb") as of:
            data = pickle.load(of)
        print(df)
        X = []
        for k, _df in data.items():
            print("  read component:", k)
            X.append(_df["activity"].values)
        X = np.asarray(X).T
    y = df["cond"].values
    y = LabelEncoder().fit_transform(y)
    y1 = df["cond"].map({"deep-propofol": "deep",
                         "ketamine": "deep",
                         "moderate-propofol": "moderate",
                         "moderate-sevoflurane": "moderate",
                         "deep-sevoflurane": "deep",
                         "awake": "awake"})
    y1 = y1.map({"awake": 0, "moderate": 1, "deep": 2}).values
    y2 = df["cond"].map({"deep-propofol": "anesthesia",
                         "ketamine": "anesthesia",
                         "moderate-propofol": "anesthesia",
                         "moderate-sevoflurane": "anesthesia",
                         "deep-sevoflurane": "anesthesia",
                         "awake": "awake"})
    y2 = y2.map({"awake": 0, "anesthesia": 1}).values
    print("  data:", X.shape, y.shape)
    X_train = X[train_mask]
    X_test = X[test_mask]
    pipelines = [
        make_pipeline(StandardScaler(),
                      SVC(kernel="rbf", C=1)),
        make_pipeline(StandardScaler(),
                      mord.LogisticIT(alpha=1.))]
    for pname, pipeline in zip(["svcrbf", "logiticit"], pipelines):
        all_scores = []
        for name, _y in [("All", y), ("DeepModerate", y1), ("Anesthesia", y2)]:
            print("- ", name, ":", pname)
            y_train = _y[train_mask]
            y_test = _y[test_mask]
            print("  train data:", X_train.shape, y_train.shape)
            print("  test data:", X_test.shape, y_test.shape)
            scores, importances = acquisition_regression(
                pipeline, X_train, y_train, X_test, y_test)
            print(scores)
            importances.to_csv(
                os.path.join(outdir, "{}_{}_importances.tsv".format(
                    pname, name)),
                sep="\t", index=False)
            scores = scores.round(2).astype(str)
            scores[name] = scores[["bacc", "bacc_std"]].agg("+/-".join, axis=1)
            all_scores.append(scores[[name]])
        summary = pd.concat(all_scores, axis=1)
        print(summary)
        summary.to_csv(os.path.join(outdir, "{}_summary.tsv".format(pname)),
                       sep="\t")


def acquisition_regression(pipeline, X, y, X_test, y_test):
    """ Basic acquisition prediction.

    Parameters
    ----------
    pipeline: sklearn Pipeline
        a base estimator.
    X: array (N, P)
        the input features.
    y: array (N, )
        the age information.
    X_test: array (M, P)
        the test features.
    y_test: array (M, )
        the age information.

    Returns
    -------
    scores: pandas DataFrame
        the regressions scores.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    estimator = BaggingClassifier(estimator=pipeline,
                                  n_estimators=10, random_state=0, n_jobs=-2)
    bacc = {}
    importances = []
    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        estimator.fit(X_train, y_train)
        importances.append(
            permutation_importance(
                estimator, X_test, y_test, n_repeats=100,
                random_state=0).importances)
        y_hat_train = estimator.predict(X_train)
        y_hat_val = estimator.predict(X_val)
        y_hat_test = estimator.predict(X_test)
        bacc.setdefault("train", []).append(
            balanced_accuracy_score(y_train, y_hat_train))
        bacc.setdefault("val", []).append(
            balanced_accuracy_score(y_val, y_hat_val))
        bacc.setdefault("test", []).append(
            balanced_accuracy_score(y_test, y_hat_test))
    importances = np.concatenate(importances, axis=1)
    labels = np.asarray(
        ["network {}".format(idx) for idx in range(1, X.shape[1] + 1)])
    importances = pd.DataFrame(importances.T, columns=labels)
    data = []
    for name in ("train", "val", "test"):
        arr = np.asarray(bacc[name])
        _bacc = arr.mean()
        _bacc_std = arr.std()
        data.append([name, _bacc, _bacc_std])
    scores = pd.DataFrame(data, columns=["data", "bacc", "bacc_std"])
    scores.set_index("data", inplace=True)
    return scores, importances


if __name__ == "__main__":

    import fire
    fire.Fire(exp_main)
