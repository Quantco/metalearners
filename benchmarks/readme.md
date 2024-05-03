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
| synthetic_data_continuous_outcome |          0.0153079 |    0.0152885 |        0.0153079 |  0.0152885 |              0.0155772 |        0.0152885 |
| synthetic_data_binary_outcome     |          0.0156171 |    0.0155887 |              nan |        nan |              0.0157499 |        0.0155887 |
| twins_pandas                      |            0.35937 |     0.366364 |              nan |        nan |               0.350377 |         0.342073 |
| twins_numpy                       |           0.300211 |     0.338768 |              nan |        nan |               0.345016 |         0.338768 |
