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
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.021464 |      0.02174 |              nan |        nan |              0.0216776 |          0.02174 |
| twins_pandas                                                  |           0.308362 |     0.345602 |              nan |        nan |               0.354783 |         0.348551 |
| twins_numpy                                                   |           0.308362 |     0.345602 |              nan |        nan |               0.349543 |         0.345602 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0621115 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0759047 |         0.075295 |

| S-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |            14.5706 |      14.6248 |          14.5706 |    14.6248 |                14.5729 |          14.6248 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.229108 |     0.228622 |              nan |        nan |                0.22925 |         0.228624 |
| twins_pandas                                                  |           0.314253 |     0.318554 |              nan |        nan |               0.371613 |         0.319028 |
| twins_numpy                                                   |           0.314253 |     0.318554 |              nan |        nan |               0.361345 |         0.318554 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |                nan |          nan |          14.1466 |    14.1853 |                14.1478 |          14.1853 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |                nan |          nan |       0.00897915 | 0.00897915 |              0.0104649 |       0.00897915 |

| X-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0458966 |    0.0456347 |        0.0458966 |  0.0456347 |               0.046185 |        0.0456347 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304592 |     0.301882 |              nan |        nan |               0.304635 |         0.301832 |
| twins_pandas                                                  |           0.325027 |     0.335259 |              nan |        nan |               0.334088 |          0.33426 |
| twins_numpy                                                   |           0.325027 |     0.335259 |              nan |        nan |               0.330992 |         0.330445 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0616481 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0754751 |         0.075295 |

| R-learner                                                     | causalml_in_sample | causalml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0465057 |    0.0471772 |              0.0499243 |        0.0474053 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304553 |     0.301832 |               0.304672 |         0.301835 |
| twins_pandas                                                  |           0.318512 |      0.34811 |               0.353968 |         0.349625 |
| twins_numpy                                                   |           0.323612 |     0.347497 |               0.350504 |         0.336285 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |            8.22679 |      8.22137 |               0.287706 |         0.278268 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |            1.33371 |      1.33346 |              0.0854205 |        0.0816067 |

| DR-learner                                                    | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0451979 |    0.0459405 |         0.253659 |   0.255802 |               0.047929 |        0.0454741 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |                nan |          nan |         0.304585 |   0.301865 |               0.304653 |         0.301818 |
| twins_pandas                                                  |                nan |          nan |              nan |        nan |               0.384664 |         0.371556 |
| twins_numpy                                                   |                nan |          nan |              nan |        nan |               0.365253 |         0.355291 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |           0.064108 |     0.063737 |         0.361528 |   0.359169 |               0.065019 |         0.062008 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |          0.0754271 |    0.0783726 |        0.0762091 |  0.0763029 |              0.0788084 |        0.0757352 |
