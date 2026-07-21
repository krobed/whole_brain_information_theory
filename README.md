# Whole-brain Reduced Wong-Wang BOLD Optimization with different information measures

This repository uses TVB-Optim library and high order interactions to fit a whole-brain model. For this, we used a dataset available at this [article](https://doi.org/10.1371/journal.pone.0314598) (Grigis A. et. al.).

### Install dependencies

`conda env create -f environment.yml`

### Make sure that file has execute permissions

In linux:
`chmod +x run_rww.sh`
Then `./run_rww.sh` 

This script will use different information measures to fit the model. Results are available at `results/{measure_name}`

Make sure to have enough computing resources as JAX library uses GPU.

