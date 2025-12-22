#!/bin/bash

python submit-umbrella-sampling.py 2cht-rcsb-aligned      5667 130185 $(readlink -f smd/2cht-rcsb-aligned-130185-smd) 50 100000 --base_folder  $(pwd) --protein
python submit-umbrella-sampling.py 2cht-rcsb-aligned      5667 260822 $(readlink -f smd/2cht-rcsb-aligned-130185-smd) 50 100000 --base_folder  $(pwd) --protein
python submit-umbrella-sampling.py 2cht-rcsb-aligned      5667 290988 $(readlink -f smd/2cht-rcsb-aligned-130185-smd) 50 100000 --base_folder  $(pwd) --protein