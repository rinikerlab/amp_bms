# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import json
import os
import numpy as np
from pathlib import Path


def write_qm_mm_json(path:str):

    parameters = {}
    parameters["qm_zone_resnames"] = ["UNL"]
    parameters["mm_zone_resnames"] = ["HOH"]

    with open(path, "w") as file:
        
        json.dump(parameters,file,indent=4)

def write_alchemical_FE_json(path:str, lambda_coulomb:float, lambda_lj:float):

    parameters = {}
    parameters["scaling_factor_coulomb_qmmm"] = lambda_coulomb
    parameters["scaling_factor_lj_qmmm"] = lambda_lj

    with open(path, "w") as file:
        
        json.dump(parameters,file,indent=4)
        

HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent
CALIBRATION_DIR = EXAMPLES_DIR.parent
DATA_DIR = CALIBRATION_DIR / "data"
write_yaml_py = CALIBRATION_DIR / "write_yaml.py"
main_calibration_py = CALIBRATION_DIR / "main_calibration.py"


lambda_ele = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.66666, 0.5, 0.33333, 0.16666, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lambda_lj  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
mol_i = 12
charge = 0 # charge of QM zone
base_path = f"/path/to/main/simulation/folder" # adjust the path as necessary
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
prod_read_freq = 200 # in steps
temp = 298.15 # in Kelvin
barostat_frequency = 2000 # in steps
ff_name = "openff_unconstrained-2.2.0"
        
for key in range(len(lambda_ele)):

    dir_path = os.path.join(base_path, f"lambda_{key:03d}")
    os.makedirs(dir_path, exist_ok=True)
    yaml_path = os.path.join(dir_path, f"lambda_{key:03d}_config.yaml")
    qm_mm_zone_json_path = os.path.join(dir_path, "qm_mm_zone_definition.json")
    alchemical_json_path = os.path.join(dir_path, "alchemical_parameters.json")
    xml_save_path = os.path.join(dir_path, "npt.xml")
    write_qm_mm_json(qm_mm_zone_json_path)
    write_alchemical_FE_json(alchemical_json_path, lambda_coulomb=lambda_ele[key], lambda_lj=lambda_lj[key])
        
    #write parameters YAML file
    command = f"""python {write_yaml_py} {yaml_path} {dir_path} {f'lambda_{key:03d}'} {pdb_path} {molecules_path} \
                    --cache_path {cache_path} --barostat_temperature {temp} --barostat_frequency {barostat_frequency} \
                    --box_dimension {box_dim} --integrator_step_size {step_size} \
                    --not_modify_internal_forces --cutoff_nb {cutoff_nb} --add_eq_csv_reporter --add_prod_dcd_reporter \
                    --integrator_temp {temp} --add_prod_csv_reporter  --prod_csv_parameters {csv_json_path} \
                    --production_steps {production_steps} --prod_readout_frequency {prod_read_freq} \
                    --save_final_state --out_state_xml_path {xml_save_path} --forcefield {forcefield_path} \
                    --use_AMP --AMP_parameters_path {amp_parameters_path} \
                    --qm_mm_zones_definition {qm_mm_zone_json_path} --weights_path {weights_path} --mol_charge {charge} \
                    --alchemical_FE --alchemical_FE_definition {alchemical_json_path} --tip4p \
                    --continue_simulation --in_state_xml_path {in_xml_path} --ff_name {ff_name} --not_minimize \
                    """
                    
    os.system(command=command)
    command = f"python {main_calibration_py} {yaml_path}"
    os.system(command=command)

