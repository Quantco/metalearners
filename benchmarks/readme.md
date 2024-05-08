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

|                                   | causalml_in_sample | causalml_oos | econml_in_sample | econml_oos | metalearners_in_sample | metalearners_oos |
| :-------------------------------- | -----------------: | -----------: | ---------------: | ---------: | ---------------------: | ---------------: |
| synthetic_data_continuous_outcome |          0.0121381 |      0.01212 |        0.0121381 |    0.01212 |              0.0124729 |          0.01212 |
| synthetic_data_binary_outcome     |          0.0149216 |    0.0148903 |              nan |        nan |              0.0149779 |        0.0148903 |
| twins_pandas                      |            0.34843 |     0.362315 |              nan |        nan |               0.354783 |         0.348551 |
| twins_numpy                       |           0.308362 |     0.345602 |              nan |        nan |               0.349543 |         0.345602 |
