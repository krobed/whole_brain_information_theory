Revisiting the standard for modeling functional brain network activity: application to consciousnes
---------------------------------------------------------------------------------------------------


Interpretable brain activity prediction using linear latent variable models
of functional connectivity function conditionned on the consciousness levels.

The MHA model is an extension of traditional factor analytic and PCA models.
MHA enjoys the following benefits over traditional methods such as PCA and
factor analysis:

- MHA is built to facilitate the interpretation of results. In particular,
  MHA introduces non-negativity and orthonormality constraints which
  effectively serve to cluster observations (e.g., ROIs in the fMRI example
  below).
- MHA is designed to accomodate data across multiple subjects (e.g., fMRI
  data across a cohort of subjects as in the example below).
- MHA is a latent variable model were we explicity allow latent variables to
  have a full covariance structure. This is in contrast to traditional latent
  variable models such as PCA, where latent variables are assumed to be
  independent.
- Despite allowing latent variables to be correlated, MHA is identifiable -
  this means that the latent variables which generated the data can be
  recovered (up to some tolerable indeterminacies).


The original [MHA implementation from Monti](https://doi.org/10.1371/journal.pone.0232296)
is available in Python and R.

* Python: https://github.com/piomonti/MHA
* R: http://www.gatsby.ucl.ac.uk/~ricardom/FactorCovariance_ScoreMatch_PenalizeLagrange.R

We have copied the Python implementation in this repository.


### Requirements

The code has been developed in Python 3.8, but should work with newer
versions as well. The dependencies are listed below:

- numpy
- pandas
- scikit-learn
- nilearn
- pingouin
- matplotlib
- seaborn
- statannotations
- mord
- fire

### Timeseries computation

To average timeseries across Regions Of Interest (ROIs), one should first
define these locations. We use two kinds of approaches: i) reference atlases
previously defined on other structural or functional datasets (CoCoMac, CIVMR),
and ii) atlases directly learned from the data (DictLearn).
The question underlying atlas selection is whether different conditions lead
to a consistent choices, and whether genericity should be preferred to
adaptive strategies. As described in the paper, the three set of ROIs for the
three listed atlases are available in the repository.


### Data description

For each atlas (CoCoMac, CIVMR, DictLearn), the input ROI averaged timeseries
and the corresponding atlas are available in:

- data/<atlas>/timeseries.npy
- data/<atlas>/atlas.nii.gz

The input subjects description is available in:

- data/participants.tsv


### Infering networks using MHA

Let's first fit the MHA model to the CoCoMac, CIVMR and DictLearn atlases, and
determine the best number of compoents k for each atlas:

```
for ATLAS in CoCoMac CIVMR DictLearn
do
    OUTDIR=./data/$ATLAS/outputs/bestk
    mkdir -p $OUTDIR
    python3 MHA_bestk.py --meta-file ./data/participants.tsv --data-file ./data/$ATLAS/timeseries.npy --atlas-file ./data/$ATLAS/atlas.nii.gz --outdir $OUTDIR --exclude ['J']
done
```

Then, refit (without early stopping) the model for each atlas using the best k:

```
declare -A bestk
bestk[CoCoMac]=4
bestk[CIVMR]=7
bestk[DictLearn]=6
for ATLAS in CoCoMac CIVMR DictLearn
do
    OUTDIR=./data/$ATLAS/outputs
    python3 MHA_fit.py --meta-file ./data/participants.tsv --data-file ./data/$ATLAS/timeseries.npy --atlas-file ./data/$ATLAS/atlas.nii.gz --outdir ./data/$ATLAS/outputs --k ${bestk[$ATLAS]}
done
```

You can also choose to exclue 'J' during the model fitting to generate the
results of the appendix on the CoCoMac atlas (please adpat the path of the
following commands 'outputs_wJ' <-> 'outputs'):

```
ATLAS=CoCoMac
OUTDIR=./data/$ATLAS/outputs_wJ
python3 MHA_fit.py --meta-file ./data/participants.tsv --data-file ./data/$ATLAS/timeseries.npy --atlas-file ./data/$ATLAS/atlas.nii.gz --outdir ./data/$ATLAS/outputs --k ${bestk[$ATLAS]} --exclude ['J']

```

### BNA-based statistical inference

Group analysis is performed on the BNAs on an atlas basis to highlight the main 269
discrepancy between anesthetic conditions. Applying the Shapiro-Wilk test reveals that 270
the BNAs do not satisfy the normal assumptions. Therefore, pairwise nonparametric 271
Wilcoxon signed-rank tests are used between paired grouped BNAs:

```
for ATLAS in CoCoMac CIVMR DictLearn
do
    OUTDIR=./data/$ATLAS/outputs/stats
    mkdir -p $OUTDIR
    python3 MHA_stats.py --meta-file ./data/participants.tsv --model-file ./data/$ATLAS/outputs/fmha_stats_k${bestk[$ATLAS]}.pkl --outdir $OUTDIR --alpha 0.01
done
```

### BNA-based multivariate analysis

To determine the best template, we perform an SVC or LogitIT (ordinal) on
the brain activities to predict the acquisition settings, taking into account
the diagonal elements in G:

```
for ATLAS in CoCoMac CIVMR DictLearn
do
    OUTDIR=./data/$ATLAS/outputs/predictions
    mkdir -p $OUTDIR
    python3 MHA_preds.py --meta-file ./data/participants.tsv --model-file ./data/$ATLAS/outputs/fmha_stats_k${bestk[$ATLAS]}.pkl --outdir $OUTDIR --exclude ['J']
done
```

### Figures & tables

To generate the figures and tables of the paper, you can run the following
scripts:

```
for REF in 2 3 4 5
do
    python3 figures/generate_figure$REF.py
done

for REF in 1 2
do
    python3 tables/generate_table$REF.py
done
```

