This directory contains benchmarks against existing MetaLearner
implementations from `econml` and `causalml`.

In order to recreate the benchmarks you can run the following:

```
$ git clone https://github.com/Quantco/metalearners.git
$ cd metalearners
$ pixi run -e benchmark postinstall
$ pixi run benchmark
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
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |            14.5706 |      14.6248 |          14.5706 |    14.6248 |                14.5707 |          14.6248 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.229101 |     0.228616 |              nan |        nan |               0.229201 |           0.2286 |
| twins_pandas                                                  |           0.314253 |     0.318554 |              nan |        nan |               0.322171 |         0.319028 |
| twins_numpy                                                   |           0.314253 |     0.318554 |              nan |        nan |               0.322132 |         0.318554 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |                nan |          nan |          14.1468 |     14.185 |                 14.147 |          14.1853 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |                nan |          nan |        0.0110779 |  0.0110778 |              0.0101122 |       0.00897915 |

| X-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0458966 |    0.0456347 |        0.0458966 |  0.0456347 |               0.046185 |        0.0456347 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304592 |     0.301882 |              nan |        nan |               0.304634 |         0.301833 |
| twins_pandas                                                  |           0.325027 |     0.335259 |              nan |        nan |               0.334088 |          0.33426 |
| twins_numpy                                                   |           0.325027 |     0.335259 |              nan |        nan |               0.330992 |         0.330445 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0616481 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0754751 |         0.075295 |

| R-learner                                                     | causalml_in_sample | causalml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |           0.045502 |    0.0460119 |              0.0502378 |        0.0477203 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304553 |     0.301835 |               0.304671 |         0.301833 |
| twins_pandas                                                  |           0.320526 |     0.347428 |               0.354841 |         0.352163 |
| twins_numpy                                                   |           0.321604 |     0.348827 |               0.349479 |         0.339678 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |            8.22625 |      8.22012 |               0.287132 |          0.27762 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |            1.33364 |       1.3333 |              0.0848038 |        0.0809661 |

| DR-learner                                                    | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0465054 |    0.0454393 |         0.252997 |   0.254672 |              0.0477179 |         0.045259 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |                nan |          nan |         0.304585 |   0.301862 |               0.304651 |         0.301819 |
| twins_pandas                                                  |                nan |          nan |              nan |        nan |               0.382051 |         0.371518 |
| twins_numpy                                                   |                nan |          nan |              nan |        nan |               0.367528 |         0.354263 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0649009 |    0.0648888 |         0.357959 |   0.362171 |              0.0651789 |        0.0621714 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075477 |    0.0763162 |        0.0760458 |  0.0760873 |              0.0788384 |        0.0757601 |
