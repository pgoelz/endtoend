This repository contains the simulation code for the paper
> Bailey Flanigan, Paul Gölz, Anupam Gupta, and Ariel D. Procaccia: Neutralizing Self-Selection Bias in Sampling for Sortition. 2020.

Code written by [Bailey Flanigan](http://baileyflanigan.com) and [Paul Gölz](https://paulgoelz.de).
The paper is freely available at https://paulgoelz.de/papers/endtoend.pdf

Content
=======
This directory contains two main experiment scripts:
- [pool_experiments.py](pool_experiments.py) estimates the β parameters and runs the column-generation algorithm.
- [end_to_end_experiments.py](end_to_end_experiments.py) samples many pools from a synthetic population and estimates end-to-end selection probabilities.

Other scripts are used for data cleaning and visualization of results.

Software Dependencies
=====================
We used the following software and libraries in the indicated versions:
- Python 3.7.6
- Gurobi 9.0.1 with python bindings
- PyTorch 1.5.0
- NumPy 1.17.3
- Pandas 1.0.3
- Seaborn 0.9.0
- Matplotlib 3.0.2
For academic use, Gurobi provides free licenses at
<http://www.gurobi.com/academia/for-universities>.

Reproduction of Results
=======================
STEP 1: Setup
-------------
1. Install dependencies
2. Replace the dummy file `data/UKrespondents.csv` with the data from the Climate Assembly (unavailable for release due to privacy considerations).
   The dummy file specifies the format of the table.
3. Download and set up the European Social Survey (ESS) data: 
    - Download the raw data (2016 Edition 2.1, UK country data, STATA format) from https://www.europeansocialsurvey.org/file/download?f=ESS8GB.stata.zip&c=GB&y=2016 (free registration required) and extract the file `ESS8GB.dta` into `data`.
    - Run `python3 clean_ESS_data.py`, which produces an output file containing all features and values for each agent in the ESS data, as well as each agent's post-stratification weight. Creates `data/cleaned_ESS_data_missingsdropped.csv` with SHA256 hash `e4d463b7972f61fb4be1488a118beb0535e3245227806a9626dcbd31e0c9836a`.
    - Run `python3 get_household_size.py`, which computes the average number of eligible agents per household. Prints average to console, which should be `2.00` for the data that we used. In the following commands, replace `<household size>` by this number.

STEP 2: Run Experiments
-----------------------
### For results without climate concern feature, run:
- `python3 pool_experiments.py --dropother --dropclimate --run --householdsize <household size> without_climate`
- `python3 end_to_end_experiments.py --r 10000 15000 60000 --steps 100000 --seed 0 without_climate`
- `python3 end_to_end_experiments.py --r 11000 12000 13000 14000 --steps 100000 --seed 0 without_climate`
- `python3 make_figures.py 0`

| Figure in paper                                | File path                                  |
|------------------------------------------------|--------------------------------------------|
| Figure 1                                       | figures/realized_representation.pdf        |
| Body: End-to-End Probabilities                 | figures/end_to_end.pdf                     |
| App. D.4: Pool and Background Data Composition | figures/data_composition.pdf               |
| App. D.4: Estimates of β                       | figures/betas.pdf                          |
| App. D.4: Estimates of qᵢ                      | figures/qis.pdf                            |
| App. D.4: Test for Calibration of qᵢ Estimates | figures/qis_calibration.pdf                |
| App. D.4: Comparison of Realized Pool …        | figures/realvhypothetical.pdf              |
| App. D.4: Testing Model Capture of 2-Corr.…    | figures/twocorrelations.pdf                |
| App. D.5: End-to-End Fairness Results for …    | figures/end_to_end_largerange.pdf          |

### For results with climate concern feature, run:
- `python3 pool_experiments.py --dropother --run --householdsize <household size> with_climate`
- `python3 end_to_end_experiments.py --r 10000 15000 600000 --steps 100000 --seed 0 with_climate`
- `python3 make_figures.py 1`

This sequence generates the same files as the version without climate concern, with the following differences:
- All files end in "_withclimate.pdf".
- There is no equivalent to `end_to_end.pdf`.
- `end_to_end_largerange_withclimate.pdf` is the plot for a much larger value of r.
