This directory contains benchmarks against existing MetaLearner
implementations from `econml` and `causalml`.

In order to recreate the benchmarks you can run the following:

```
$ git clone https://github.com/Quantco/metalearners.git
$ cd metalearners
$ micromamba env create -f benchmarks/environment.yml
$ micromamba activate benchmarks
$ pip install -e .
$ python benchmarks/benchmark.py
```

After this, you can investigate `comparison.json`, e.g. by running

```
$ cat comparison.json
```

## Results

Recents results look as such, where each cell corresponds to an RMSE
on ground truth CATEs:

| T-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0458966 |    0.0456347 |        0.0458966 |  0.0456347 |              0.0467864 |        0.0456347 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |          0.0212419 |    0.0215154 |              nan |        nan |               0.021512 |        0.0215154 |
| twins_pandas                                                  |           0.308362 |     0.345602 |              nan |        nan |               0.354783 |         0.348551 |
| twins_numpy                                                   |           0.308362 |     0.345602 |              nan |        nan |               0.349543 |         0.345602 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0621115 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0759047 |         0.075295 |

| S-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |            14.5706 |      14.6248 |          14.5706 |    14.6248 |                14.5729 |          14.6248 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.229101 |     0.228616 |              nan |        nan |               0.229231 |           0.2286 |
| twins_pandas                                                  |           0.314253 |     0.318554 |              nan |        nan |               0.371613 |         0.319028 |
| twins_numpy                                                   |           0.314253 |     0.318554 |              nan |        nan |               0.361345 |         0.318554 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |                nan |          nan |          14.1468 |     14.185 |                14.1478 |          14.1853 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |                nan |          nan |        0.0110779 |  0.0110778 |              0.0104649 |       0.00897915 |

| X-learner                                                    | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :----------------------------------------------------------- | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te |          0.0458966 |    0.0456347 |        0.0458966 |  0.0456347 |               0.046185 |        0.0456347 |
| synthetic_data_binary_outcome_binary_treatment_linear_te     |           0.304592 |     0.301882 |              nan |        nan |               0.304634 |         0.301833 |
| twins_pandas                                                 |           0.325027 |     0.335259 |              nan |        nan |               0.334088 |          0.33426 |
| twins_numpy                                                  |           0.325027 |     0.335259 |              nan |        nan |               0.330992 |         0.330445 |

| R-learner                                                    | causalml_in_sample | causalml_oos | metalearners_in_sample | metalearners_oos |
| :----------------------------------------------------------- | -----------------: | -----------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te |          0.0469718 |    0.0462968 |              0.0499107 |        0.0473927 |
| synthetic_data_binary_outcome_binary_treatment_linear_te     |           0.304553 |     0.301832 |               0.304672 |         0.301835 |
| twins_pandas                                                 |           0.322255 |     0.344373 |               0.353968 |         0.349625 |
| twins_numpy                                                  |           0.322681 |      0.34876 |               0.350513 |         0.336281 |

| DR-learner                                                   | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :----------------------------------------------------------- | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te |          0.0463883 |    0.0454509 |         0.248504 |   0.248071 |              0.0479302 |        0.0454753 |
| synthetic_data_binary_outcome_binary_treatment_linear_te     |                nan |          nan |         0.304584 |   0.301864 |               0.304651 |         0.301819 |
| twins_pandas                                                 |                nan |          nan |              nan |        nan |               0.384664 |         0.371556 |
| twins_numpy                                                  |                nan |          nan |              nan |        nan |               0.365246 |         0.355291 |
