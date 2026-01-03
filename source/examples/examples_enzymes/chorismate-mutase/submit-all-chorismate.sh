#!/bin/bash

python submit-umbrella-sampling.py chorismate                      1 130185 $(readlink -f smd/charge/chorismate-130185-smd) 50 100000 --base_folder  $(pwd)
python submit-umbrella-sampling.py chorismate                      1 260822 $(readlink -f smd/charge/chorismate-130185-smd) 50 100000 --base_folder  $(pwd)
python submit-umbrella-sampling.py chorismate                      1 290988 $(readlink -f smd/charge/chorismate-130185-smd) 50 100000 --base_folder  $(pwd)