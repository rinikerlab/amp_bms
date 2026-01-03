# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import json
import os
import numpy as np
import yaml
from pathlib import Path


def write_qm_mm_json(path:str):

    parameters = {}
    parameters["qm_zone_resnames"] = ["UNL"]
    parameters["mm_zone_resnames"] = ["HOH"]

    with open(path, "w") as file:
        
        json.dump(parameters,file,indent=4)

def load_yaml(path:str):
    
    parameters = yaml.safe_load(Path(path).read_text())
     
    return parameters

HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent
CALIBRATION_DIR = EXAMPLES_DIR.parent
DATA_DIR = CALIBRATION_DIR / "data"
write_yaml_py = CALIBRATION_DIR / "write_yaml.py"
main_calibration_py = CALIBRATION_DIR / "main_water_in_water_calibration.py"



charge = 0 # charge of QM zone
base_path = "/base/path/to/simulation/folder" # adjust the path as necessary
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
production_steps = 4000000 # 2 ns 
prod_read_freq = 200 # in steps
temp = 298.15 # in Kelvin
barostat_frequency = 2000 # in steps
ff_name = "openff-2.2.0"
scaling_factor = 0.35

dir_path = os.path.join(base_path, f"scaling_index_035")
os.makedirs(dir_path, exist_ok=True)
yaml_path = os.path.join(dir_path, f"scaling_index_best_035_config.yaml")
qm_mm_zone_json_path = os.path.join(dir_path, "qm_mm_zone_definition.json")
xml_save_path = os.path.join(dir_path, "npt.xml")
params = load_yaml(amp_parameters_path)
params["pol_scaling"] = scaling_factor
amp_parameters_path_new = os.path.join(dir_path, "amp_parameters.yaml")
with open(amp_parameters_path_new, 'w') as file:
    yaml.dump(params, file)

write_qm_mm_json(qm_mm_zone_json_path)

#write parameters YAML file
command = f"""python {write_yaml_py} {yaml_path} {dir_path} scaling_index_035 {pdb_path} {molecules_path} \
                --cache_path {cache_path} --barostat_temperature {temp} --barostat_frequency {barostat_frequency} \
                --box_dimension {box_dim} --integrator_step_size {step_size} \
                --not_modify_internal_forces --cutoff_nb {cutoff_nb} --add_eq_csv_reporter --add_prod_dcd_reporter \
                --integrator_temp {temp} --add_prod_csv_reporter  --prod_csv_parameters {csv_json_path} \
                --production_steps {production_steps} --prod_readout_frequency {prod_read_freq} \
                --save_final_state --out_state_xml_path {xml_save_path} --forcefield {forcefield_path} \
                --use_AMP --AMP_parameters_path {amp_parameters_path_new} --qm_mm_zones_definition {qm_mm_zone_json_path} \
                --weights_path {weights_path} --mol_charge {charge} \
                --ff_name {ff_name} --tip4p --not_minimize \
                """
os.system(command=command)
command = f"python {main_calibration_py} {yaml_path}"
os.system(command=command)