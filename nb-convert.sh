#!/bin/bash

set -euo pipefail

jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags no-convert --to script docs/examples/*.ipynb
shopt -s nullglob # Prevents the loop from running if there are no .txt files
for file in docs/examples/*.txt;
do mv -- "$file" "${file%.txt}.py";
done
