# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd
from git_root import git_root

x = pd.read_csv(
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv"
)
y = pd.read_csv(
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv"
)
t = pd.read_csv(
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv"
)

# _0 denotes features specific to the lighter twin and _1 denotes features specific to the heavier twin
lighter_columns = [
    "pldel",
    "birattnd",
    "brstate",
    "stoccfipb",
    "mager8",
    "ormoth",
    "mrace",
    "meduc6",
    "dmar",
    "mplbir",
    "mpre5",
    "adequacy",
    "orfath",
    "frace",
    "birmon",
    "gestat10",
    "csex",
    "anemia",
    "cardiac",
    "lung",
    "diabetes",
    "herpes",
    "hydra",
    "hemo",
    "chyper",
    "phyper",
    "eclamp",
    "incervix",
    "pre4000",
    "preterm",
    "renal",
    "rh",
    "uterine",
    "othermr",
    "tobacco",
    "alcohol",
    "cigar6",
    "drink5",
    "crace",
    "data_year",
    "nprevistq",
    "dfageq",
    "feduc6",
    "infant_id_0",
    "dlivord_min",
    "dtotord_min",
    "bord_0",
    "brstate_reg",
    "stoccfipb_reg",
    "mplbir_reg",
]
heavier_columns = [
    "pldel",
    "birattnd",
    "brstate",
    "stoccfipb",
    "mager8",
    "ormoth",
    "mrace",
    "meduc6",
    "dmar",
    "mplbir",
    "mpre5",
    "adequacy",
    "orfath",
    "frace",
    "birmon",
    "gestat10",
    "csex",
    "anemia",
    "cardiac",
    "lung",
    "diabetes",
    "herpes",
    "hydra",
    "hemo",
    "chyper",
    "phyper",
    "eclamp",
    "incervix",
    "pre4000",
    "preterm",
    "renal",
    "rh",
    "uterine",
    "othermr",
    "tobacco",
    "alcohol",
    "cigar6",
    "drink5",
    "crace",
    "data_year",
    "nprevistq",
    "dfageq",
    "feduc6",
    "infant_id_1",
    "dlivord_min",
    "dtotord_min",
    "bord_1",
    "brstate_reg",
    "stoccfipb_reg",
    "mplbir_reg",
]

# Since data has pair property,processing the data to get separate row for each twin so that each child can be treated as an instance
data = []

for i in range(len(t.values)):

    # select only if both <=2kg
    if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
        continue

    this_instance_lighter = list(x.iloc[i][lighter_columns].values)
    this_instance_heavier = list(x.iloc[i][heavier_columns].values)

    # adding weight
    this_instance_lighter.append(t.iloc[i].values[1])
    this_instance_heavier.append(t.iloc[i].values[2])

    # adding treatment, is_heavier
    this_instance_lighter.append(0)
    this_instance_heavier.append(1)

    # adding the outcome
    this_instance_lighter.append(y.iloc[i].values[1])
    this_instance_heavier.append(y.iloc[i].values[2])
    data.append(this_instance_lighter)
    data.append(this_instance_heavier)

cols = [
    "pldel",
    "birattnd",
    "brstate",
    "stoccfipb",
    "mager8",
    "ormoth",
    "mrace",
    "meduc6",
    "dmar",
    "mplbir",
    "mpre5",
    "adequacy",
    "orfath",
    "frace",
    "birmon",
    "gestat10",
    "csex",
    "anemia",
    "cardiac",
    "lung",
    "diabetes",
    "herpes",
    "hydra",
    "hemo",
    "chyper",
    "phyper",
    "eclamp",
    "incervix",
    "pre4000",
    "preterm",
    "renal",
    "rh",
    "uterine",
    "othermr",
    "tobacco",
    "alcohol",
    "cigar6",
    "drink5",
    "crace",
    "data_year",
    "nprevistq",
    "dfageq",
    "feduc6",
    "infant_id",
    "dlivord_min",
    "dtotord_min",
    "bord",
    "brstate_reg",
    "stoccfipb_reg",
    "mplbir_reg",
    "wt",
    "treatment",
    "outcome",
]
df = pd.DataFrame(columns=cols, data=data)

df.to_csv(
    git_root("data/twins.zip"),
    compression={"method": "zip", "archive_name": "twins.csv"},
    index=False,
)
