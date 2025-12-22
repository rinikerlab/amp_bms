#!/bin/bash
python smd.py 2cht-rcsb-aligned      5667 2cht-rcsb-aligned  --ligand chorismate-aligned --fix --minimization 10
python smd.py chorismate             1                       --ligand chorismate-aligned --fix --minimization 10