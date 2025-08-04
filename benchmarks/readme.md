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
| synthetic_data_binary_outcome_binary_treatment_linear_te      |          0.0214902 |    0.0217761 |              nan |        nan |              0.0216828 |        0.0217761 |
| twins_pandas                                                  |           0.308362 |     0.345602 |              nan |        nan |               0.354783 |         0.348551 |
| twins_numpy                                                   |           0.308362 |     0.345602 |              nan |        nan |               0.349543 |         0.345602 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |          0.0615009 |     0.061717 |        0.0615009 |   0.061717 |              0.0621115 |         0.061717 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.075331 |     0.075295 |         0.075331 |   0.075295 |              0.0759047 |         0.075295 |

| S-learner                                                     | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |            14.5706 |      14.6248 |          14.5706 |    14.6248 |                14.5707 |          14.6248 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.229104 |     0.228609 |              nan |        nan |                 0.2292 |         0.228605 |
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
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0460793 |    0.0469546 |              0.0502287 |        0.0477101 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |           0.304553 |     0.301835 |               0.304671 |         0.301833 |
| twins_pandas                                                  |           0.325398 |     0.349962 |               0.354841 |         0.352163 |
| twins_numpy                                                   |           0.327066 |     0.352357 |               0.349479 |         0.339678 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |           0.278909 |     0.279225 |               0.287116 |         0.277606 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |          0.0818863 |    0.0817913 |              0.0848569 |        0.0810205 |

| DR-learner                                                    | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :------------------------------------------------------------ | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome_binary_treatment_linear_te  |          0.0946389 |     0.146014 |         0.248725 |     0.2467 |              0.0477205 |        0.0452617 |
| synthetic_data_binary_outcome_binary_treatment_linear_te      |                nan |          nan |         0.304583 |   0.301861 |               0.304652 |         0.301818 |
| twins_pandas                                                  |                nan |          nan |              nan |        nan |               0.382051 |         0.371518 |
| twins_numpy                                                   |                nan |          nan |              nan |        nan |               0.367528 |         0.354263 |
| synthetic_data_continuous_outcome_multi_treatment_linear_te   |           0.419679 |     0.608385 |         0.358621 |   0.361715 |              0.0651796 |        0.0621723 |
| synthetic_data_continuous_outcome_multi_treatment_constant_te |           0.482103 |     0.606256 |         0.076316 |   0.075765 |              0.0788408 |        0.0757627 |
