#!/bin/bash

set -e

jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags no-convert --to script docs/examples/*.ipynb &&
for file in docs/examples/*.txt;
do mv -- "$file" "${file%.txt}.py";
done
