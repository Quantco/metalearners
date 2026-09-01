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
| synthetic_data_binary_outcome_binary_treatment_linear_te      |          0.0213081 |    0.0215812 |              nan |        nan |              0.0215817 |        0.0215812 |
| twins_pandas                                                  |           0.308362 |     0.345602 |              nan |        nan |               0.354783 |         0.348551 |
| twins_numpy                                                   |           0.308362 |     0.345602 |              nan |        nan |               0.349543 |         0.345602 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0621115 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0759047 |         0.075295 |

| S-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |            14.5706 |      14.6248 |          14.5706 |    14.6248 |                14.5707 |          14.6248 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |            0.22908 |     0.228594 |              nan |        nan |               0.229195 |          0.22861 |
| twins_pandas                                                  |           0.314253 |     0.318554 |              nan |        nan |               0.321511 |         0.318397 |
| twins_numpy                                                   |           0.314253 |     0.318554 |              nan |        nan |               0.321511 |         0.318397 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |                nan |          nan |          14.1466 |    14.1853 |                 14.147 |          14.1853 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |                nan |          nan |       0.00897915 | 0.00897915 |              0.0101122 |       0.00897915 |

| X-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0458966 |    0.0456347 |        0.0458966 |  0.0456347 |               0.046185 |        0.0456347 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304592 |     0.301882 |              nan |        nan |               0.304634 |         0.301832 |
| twins_pandas                                                  |           0.325027 |     0.335259 |              nan |        nan |               0.334088 |          0.33426 |
| twins_numpy                                                   |           0.325027 |     0.335259 |              nan |        nan |               0.330992 |         0.330445 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0616481 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0754751 |         0.075295 |

| R-learner                                                     | causalml_in_sample | causalml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0462255 |     0.046374 |               0.050264 |        0.0477462 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304553 |     0.301833 |               0.304671 |         0.301833 |
| twins_pandas                                                  |           0.325384 |      0.35071 |               0.354841 |         0.352163 |
| twins_numpy                                                   |           0.328911 |     0.349626 |               0.349479 |         0.339678 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |           0.278926 |     0.278768 |               0.287077 |         0.277564 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |          0.0815243 |    0.0816331 |              0.0848195 |         0.080982 |

| DR-learner                                                    | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0439431 |     0.109736 |         0.249413 |   0.255546 |              0.0477176 |        0.0452587 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |                nan |          nan |         0.304583 |   0.301863 |               0.304652 |         0.301817 |
| twins_pandas                                                  |                nan |          nan |              nan |        nan |               0.382051 |         0.371518 |
| twins_numpy                                                   |                nan |          nan |              nan |        nan |               0.367528 |         0.354263 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |           0.473745 |      0.48371 |         0.356675 |   0.359222 |              0.0651804 |        0.0621731 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.449959 |     0.276953 |        0.0759822 |  0.0759191 |               0.078836 |        0.0757576 |
