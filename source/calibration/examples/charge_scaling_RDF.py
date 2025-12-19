# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import json
import os
import numpy as np
from pathlib import Path

def write_qm_mm_json(path):

    parameters = {}
    parameters["qm_zone_resnames"] = ["UNL"]
    parameters["mm_zone_resnames"] = ["HOH"]

    with open(path, "w") as file:
        
        json.dump(parameters,file,indent=4)

HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent
CALIBRATION_DIR = EXAMPLES_DIR.parent
DATA_DIR = CALIBRATION_DIR / "data"
write_yaml_py = CALIBRATION_DIR / "write_yaml.py"
main_calibration_py = CALIBRATION_DIR / "main_water_in_water_calibration.py"

base_path = "/path/to/simulation/folder" # adjust the path as necessary
scaling_factor_charge = 1.0 # scaling of charges (lambda_ch in manuscript) 
charge = 0 # charge of QM zone
pdb_path = DATA_DIR / "input_water.pdb"
amp_parameters_path = DATA_DIR / "PARAMETERS_MIN_LRv2_tip4pfb.yaml"
weights_path = DATA_DIR / "MIN_state_dict"
cache_path = DATA_DIR / "forcefield_water_off.json"
molecules_path = DATA_DIR / "molecules_water_in_water.txt"
csv_json_path = DATA_DIR / "default_csv_parameters.json"
forcefield_path = DATA_DIR / "ff_default_tip4pfb.json"
box_dim = 3.0 # in nm
cutoff_nb = 0.9 # in nm
step_size = 0.0005 # in ps
production_steps = 2000000 # 1 ns 
prod_read_freq = 200 # in steps
temp = 298.15 # in Kelvin
barostat_frequency = 2000 # in steps
ff_name = "openff-2.2.0" # force field to parametrize the QM zone to obtain LJ parameters

dir_path = os.path.join(base_path, "md")
os.makedirs(dir_path, exist_ok=True)
yaml_path = os.path.join(dir_path, f"config.yaml")
qm_mm_zone_json_path = os.path.join(dir_path, "qm_mm_zone_definition.json")
xml_save_path = os.path.join(dir_path, "npt.xml")
write_qm_mm_json(qm_mm_zone_json_path)

#write parameters YAML file
command = f"""python {write_yaml_py} {yaml_path} {dir_path} md {pdb_path} {molecules_path} \
                --cache_path {cache_path} --barostat_temperature {temp} \
                --barostat_frequency {barostat_frequency} --box_dimension {box_dim} \
                --integrator_step_size {step_size} --not_modify_internal_forces --cutoff_nb {cutoff_nb} \
                --add_eq_csv_reporter --add_prod_dcd_reporter \
                --integrator_temp {temp} --add_prod_csv_reporter  --prod_csv_parameters {csv_json_path} \
                --production_steps {production_steps} --prod_readout_frequency {prod_read_freq} \
                --save_final_state --out_state_xml_path {xml_save_path} --forcefield {forcefield_path} \
                --use_AMP --AMP_parameters_path {amp_parameters_path} --not_minimize \
                --qm_mm_zones_definition {qm_mm_zone_json_path} --weights_path {weights_path} --mol_charge {charge} \
                --scaling_charges {scaling_factor_charge} --ff_name {ff_name} --tip4p \
                """
os.system(command=command)
command = f"python {main_calibration_py} {yaml_path}"
os.system(command=command)
