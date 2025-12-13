# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import json
import os
import numpy as np
from helper_functions import write_config_yaml
from pathlib import Path

HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent
CALIBRATION_DIR = EXAMPLES_DIR.parent
DATA_DIR = CALIBRATION_DIR / "data"
write_yaml_py = CALIBRATION_DIR / "write_yaml.py"
get_energy_py = CALIBRATION_DIR / "HFE_get_lambda_energies_ELE_LJ.py"


lambda_ele = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6666666666666667, 0.5, 0.33333333333333337, 0.16666666666666674, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lambda_lj  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3999999999999999, 0.29999999999999993, 0.19999999999999996, 0.09999999999999998, 0.0]
mol_i = 12
base_path = f"/path/to/main/simulation/folder" # adjust the path as necessary, same base_path as in 'HFE_ELE_LJ_mol_012.py'
pdb_path = DATA_DIR / "MOL_012_solvated.pdb"
in_xml_path = DATA_DIR / "MOL_012_equilibrated_state.xml"
amp_parameters_path = DATA_DIR / "PARAMETERS_MIN_LRv2_tip4pfb.yaml"
weights_path = DATA_DIR / "MIN_state_dict"
cache_path = DATA_DIR / "forcefield_parameters_HFE_molecules.json"
molecules_path = DATA_DIR / "molecules_for_HFE.txt"
csv_json_path = DATA_DIR / "default_csv_parameters.json"
forcefield_path = DATA_DIR / "ff_default_tip4pfb.json"
box_dim = 3.0 # in nm
cutoff_nb = 0.9 # in nm
step_size = 0.0005 # in ps
production_steps = 4200000 # 2.1 ns 
prod_read_freq = 100 # in steps
temp = 298.15 # Kelvin
barostat_frequency = 2000 # in steps
ff_name = "openff_unconstrained-2.2.0" 
    
for key in range(len(lambda_ele)):

  dir_path = os.path.join(base_path, f"lambda_{key:03d}")
      
  yaml_path = os.path.join(dir_path, f"lambda_{key:03d}_postprocess.yaml")

  dcd_path = os.path.join(dir_path, "production", f"production_trajectory.dcd")

  npy_path = os.path.join(dir_path, "energies.npy")
      
  write_config_yaml(yaml_path=yaml_path,
                        molecules_file=molecules_path,
                        pdb_path=pdb_path,
                        dcd_path=dcd_path,
                          npy_path=npy_path,
                          box_dimension=box_dim,
                          cutoff_nb=cutoff_nb,
                          nonbondedMethod="PME",
                          forcefield=forcefield_path,
                          cache_path=cache_path,
                          ff_name=ff_name,
                          AMP_parameters_path=amp_parameters_path,
                          weights_path=weights_path,
                          device_ml="cuda",
                          mol_charge=0,
                          qm_zone_resnames=["UNL"],
                          mm_zone_resnames=["HOH"],
                          scaling_charges=1.0,
                          tip4p=True,
                          lambdas_lj=lambda_lj,
                          lambdas_coulomb=lambda_ele,
                          platform_name="CUDA")

  command = f"python {get_energy_py} {yaml_path}"
  os.system(command=command)


