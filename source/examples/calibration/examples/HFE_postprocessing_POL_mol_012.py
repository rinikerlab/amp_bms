# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import json
import os
import numpy as np
from helper_functions import write_config_yaml
import yaml
from pathlib import Path

def load_yaml(path:str):
    
    parameters = yaml.safe_load(Path(path).read_text())
     
    return parameters


HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent
CALIBRATION_DIR = EXAMPLES_DIR.parent
DATA_DIR = CALIBRATION_DIR / "data"
write_yaml_py = CALIBRATION_DIR / "write_yaml.py"
get_energy_py = CALIBRATION_DIR / "HFE_get_lambda_energies_POL.py"

lambda_pol = [1.0, 0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.2, 0.0]
mol_i = 12
base_path = f"/path/to/main/simulation/folder/MOL_{mol_i:03d}" # adjust the path as necessary
pdb_path = DATA_DIR / "MOL_012_solvated.pdb"
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
prod_read_freq = 200 # in steps
temp = 298.15 # in Kelvin
barostat_frequency = 2000 # in steps
ff_name = "openff_unconstrained-2.2.0"
  
for key in range(len(lambda_pol)):
      
  for pol in lambda_pol:

    dir_path = os.path.join(base_path, f"lambda_{key:03d}")

    yaml_path = os.path.join(dir_path, f"lambda_{key:03d}_postprocess_pol_{round(pol, 3)}.yaml")
        
    params = load_yaml(amp_parameters_path)
    params["pol_scaling"] = float(pol)
    amp_parameters_path_new = os.path.join(dir_path, f"amp_parameters_postprocess_pol_{round(pol, 3)}.yaml")
        
    with open(amp_parameters_path_new, 'w') as file:
        yaml.dump(params, file)

    dcd_path = os.path.join(dir_path, "production", f"production_trajectory.dcd")

    npy_path = os.path.join(dir_path, f"energies_pol_{round(pol, 3)}.npy")

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
                      AMP_parameters_path=amp_parameters_path_new,
                      weights_path=weights_path,
                      device_ml="cuda",
                      mol_charge=0,
                      qm_zone_resnames=["UNL"],
                      mm_zone_resnames=["HOH"],
                      scaling_charges=1.0,
                      tip4p=True,
                      lambdas_lj=1.0,
                      lambdas_coulomb=1.0,
                      platform_name="CUDA")

  command = f"python {get_energy_py} {yaml_path}"
  os.system(command=command)


