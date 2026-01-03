# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import yaml
import argparse
import json
from typing import Dict, List

def read_molecules_file(path:str)->List[str]:
    
    with open(path, "r") as file:
        
        all_lines = file.readlines()
        
        molecule_definitions = [line.strip() for line in all_lines if line.strip()!=""]
            
    return molecule_definitions

def jsonfile2dict(path:str)->Dict:
    
    with open(path, "r") as file:
    
        data = json.load(file)

    return data

def readjsonfile(path:str):
    
    with open(path, "r") as file:
    
        data = json.load(file)

    return data
        
def main():
    
    #positional
    parser = argparse.ArgumentParser(description='This is a script that creates YAML file for OpenMM simulation.')
    parser.add_argument("file_path", type=str, help="Output YAML file path.")
    parser.add_argument("base_path", type=str, help='Path to folder, where the simulation folder will be created.')
    parser.add_argument("simulation_name", type=str, help='Name of the folder that should be created for simulation.')
    parser.add_argument("pdb_path", type=str, help="The path to PDB file for simulation start.")
    parser.add_argument('molecules_file', type=str, help="Path to the file with all molecules in SMILES or SDF paths format.")
    
    #optional
    parser.add_argument('--not_set_logger', action='store_false', help='Do not use logger.')
    parser.add_argument('--forcefield', type=str, default="/cluster/project/igc/igordiy/amp_overpolarization_benchmark/data/ff_default.json")
    parser.add_argument('--cache_path', type=str, default=None, help="Path to the file that will store parameters after FF parametrization.")
    parser.add_argument('--ff_name', type=str, default="openff_unconstrained-2.1.0", help="Forcefield name for parametrization.")
    parser.add_argument('--integrator_type', type=str, default="LMI", help="Alias for OpenMM integrator.")
    parser.add_argument('--integrator_temp', type=float, default=298.15, help="Temperature in Kelvin for integrator.")
    parser.add_argument('--integrator_fric_coeff', type=float, default=1.0, help="Friction coefficient in inverse picoseconds for integrator.")
    parser.add_argument('--integrator_step_size', type=float, default=0.002, help="Step size in picoseconds for integrator.")
    parser.add_argument('--not_use_barostat', action='store_false', help='Do not use barostat.')
    parser.add_argument('--barostat_type', type=str, default="MCB", help="Alias for OpenMM barostat.")
    parser.add_argument('--barostat_pressure', type=float, default=1.0, help="Barostat pressure in bar.")
    parser.add_argument('--barostat_temperature', type=float, default=298.15, help="Barostat temperature in kelvin.")
    parser.add_argument('--barostat_frequency', type=int, default=25, help="Barostat frequency in time steps.")
    parser.add_argument('--not_minimize', action='store_false', help='Do not system minimization.')
    parser.add_argument('--max_iterations', type=int, default=0, help='Number of maximum minimization steps. If 0 - unlimited.')
    parser.add_argument('--box_dimension', type=float, default=85.0, help='Cubic box dimension in Angstrom.')
    parser.add_argument('--cutoff_nb', type=float, default=1.0, help='LJ cutoff in nm.')
    parser.add_argument('--nonbondedMethod', type=str, default="PME", help="Alias for nonbonded method in OpenMM.")
    parser.add_argument('--platform_name', type=str, default="CUDA", help="Platform name to run simulation on. Can be one of: Reference, CUDA, CPU, OpenCL.")
    parser.add_argument('--no_rigidWater', action='store_false', help='Do not make water rigid in simulation.')
    parser.add_argument('--constraints', type=str, default="", help="Alias for constraints in OpenMM.")
    parser.add_argument('--not_modify_internal_forces', action='store_false', help='Do not modify standard forces in the system.')
    parser.add_argument('--forces_to_modify', type=str, help="Path to json file in which the forces for modification are specified.")
    parser.add_argument('--add_restraint_position', action='store_true', help='Add positional restraining for COM of group of atoms.')
    parser.add_argument('--pos_res_definition', type=str, help='Path to JSON file with all parameters needed for positional restraint definition.')
    parser.add_argument('--equilibration', action='store_true', help='Do equilibration run before production.')
    parser.add_argument('--equilibration_step_size', type=float, default=0.002, help="Step size in picoseconds for integrator for equilibration run.")
    parser.add_argument('--eq_readout_frequency', type=int, default=500, help="Frequency of readout for equilibration run.")
    parser.add_argument('--initial_topology_name', type=str, default="init_topology.pdb", help="Name of file for writing out topology before simulation.")
    parser.add_argument('--add_eq_csv_reporter', action='store_true', help='Add CSV reporter for equilibration run.')
    parser.add_argument('--add_eq_hdf5_reporter', action='store_true', help='Add HDF5 reporter for equilibration run.')
    parser.add_argument('--add_eq_dcd_reporter', action='store_true', help='Add DCD reporter for equilibration run.')
    parser.add_argument('--add_eq_chk_reporter', action='store_true', help='Add CHK reporter for equilibration run.')
    parser.add_argument('--eq_csv_name', type=str, default="equilibration_properties_trajectory.csv", help="Name of file for writing out CSV file from reporter in equilibration run.")
    parser.add_argument('--eq_csv_parameters', type=str, help="Path to JSON file with all parameters for CSV reporter.")
    parser.add_argument('--eq_dcd_name', type=str, default="equilibration_trajectory.dcd", help="Name of file for writing out DCD file from reporter in equilibration run.")
    parser.add_argument('--eq_chk_name', type=str, default="equilibration_checkpoint.chk", help="Name of file for writing out CHK file from reporter in equilibration run.")
    parser.add_argument('--eq_hdf5_name', type=str, default="equilibration_trajectory.h5", help="Name of file for writing out HDF5 file from reporter in equilibration run.")
    parser.add_argument('--eq_hdf5_parameters', type=str, help="Path to JSON file with all parameters for HDF5 reporter.")
    parser.add_argument('--equilibration_steps', type=int, default=50000, help="Number of equilibration steps.")
    parser.add_argument('--prod_readout_frequency', type=int, default=1000, help="Frequency of redout for production run.")
    parser.add_argument('--not_write_nonbonded_parameters', action='store_false', help='Do not write the file with all nonbonded parameters before production simulation.')
    parser.add_argument('--simulation_phase', type=str, default="production", help='Add simulation phase name (e.g. NVT, NPT, prod).')
    parser.add_argument('--add_prod_csv_reporter', action='store_true', help='Add CSV reporter for production run.')
    parser.add_argument('--add_prod_hdf5_reporter', action='store_true', help='Add HDF5 reporter for production run.')
    parser.add_argument('--add_prod_dcd_reporter', action='store_true', help='Add DCD reporter for production run.')
    parser.add_argument('--add_prod_chk_reporter', action='store_true', help='Add CHK reporter for production run.')
    parser.add_argument('--prod_csv_name', type=str, default="production_properties_trajectory.csv", help="Name of file for writing out CSV file from reporter in equilibration run.")
    parser.add_argument('--prod_csv_parameters', type=str, help="Path to JSON file with all parameters for CSV reporter.")
    parser.add_argument('--prod_dcd_name', type=str, default="production_trajectory.dcd", help="Name of file for writing out DCD file from reporter in production run.")
    parser.add_argument('--prod_chk_name', type=str, default="production_checkpoint.chk", help="Name of file for writing out CHK file from reporter in production run.")
    parser.add_argument('--prod_hdf5_name', type=str, default="production_trajectory.h5", help="Name of file for writing out HDF5 file from reporter in production run.")
    parser.add_argument('--prod_hdf5_parameters', type=str, help="Path to JSON file with all parameters for HDF5 reporter.")
    parser.add_argument('--production_steps', type=int, default=500000, help="Number of production steps.")
    parser.add_argument('--use_AMP', action='store_true', help='Do QM/MM MD with electrostatic embedding using AMP3 as QM engine.')
    parser.add_argument('--AMP_parameters_path', type=str, help='Path to YAML file with all parameters needed for AMP3 definition.')
    parser.add_argument('--qm_mm_zones_definition', type=str, help='Path to JSON file with all parameters needed for QM and MM zones definition.')
    parser.add_argument('--device_ml', type=str, default="cuda", help='Add desription here.')
    parser.add_argument('--weights_path', type=str, help='Add desription here.')
    parser.add_argument('--continue_simulation', action='store_true', help='Start the simulation from the state XML file.')
    parser.add_argument('--save_final_state', action='store_true', help='After simulation is complete save final state as XML file for later restart.')
    parser.add_argument('--in_state_xml_path', type=str, default=None, help='Path to XML file of the simulation state, from which to continue.')
    parser.add_argument('--out_state_xml_path', type=str, default=None, help='Path to XML file of the simulation state. After simulation is complete the final state is saved as XML file.')
    parser.add_argument('--set_to_temperature', action='store_true', help='Assign the velocities to particles according to the temperature of the integrator.')
    parser.add_argument('--mol_charge', default=0, type=int, help='Charge of the QM zone.')
    parser.add_argument('--scaling_charges', default=1.0, type=float, help='Scaling factor to scale all MM charges for QM-MM interactions.')
    parser.add_argument('--tip4p', action='store_true', help='Flag to simulate with TIP4P-FB water model.')
    ###########Parameters for Alchemical FE##########
    parser.add_argument('--alchemical_FE', action='store_true', help='Flag to check if we want to do alchemical free energy calculation.')
    parser.add_argument('--alchemical_FE_definition', type=str, help='Path to JSON file with all parameters needed for alchemical free energy calculation.')
    
    args = parser.parse_args()
    
    all_parameters = dict()
    
    all_parameters["base_path"] = args.base_path
    all_parameters["simulation_name"] = args.simulation_name
    all_parameters["pdb_path"] = args.pdb_path
    all_parameters["molecules"] = read_molecules_file(args.molecules_file)
   
    all_parameters["simulation_phase"] = args.simulation_phase 
    all_parameters["set_logger"] = args.not_set_logger
    
    all_parameters["forcefield"] = jsonfile2dict(args.forcefield)
    all_parameters["cache_path"] = args.cache_path
    all_parameters["ff_name"] = args.ff_name
    all_parameters["integrator_type"] = args.integrator_type
    all_parameters["integrator_parameters"] = {"temperature":args.integrator_temp,
                                               "friction_coefficient":args.integrator_fric_coeff,
                                               "step_size":args.integrator_step_size}
    
    all_parameters["use_barostat"] = args.not_use_barostat
    
    if all_parameters["use_barostat"]:
        
        all_parameters["barostat_type"] = args.barostat_type
        
        all_parameters["barostat_parameters"] = {"pressure":args.barostat_pressure,
                                                 "temperature":args.barostat_temperature,
                                                 "frequency":args.barostat_frequency}
        
    
    all_parameters["set_to_temperature"] = args.set_to_temperature
    all_parameters["minimize"] = args.not_minimize
    all_parameters["maxIterations"] = args.max_iterations
    all_parameters["box_dimension"] = args.box_dimension
    all_parameters["cutoff_nb"] = args.cutoff_nb
    all_parameters["nonbondedMethod"] = args.nonbondedMethod
    all_parameters["rigidWater"] = args.no_rigidWater
    all_parameters["constraints"] = args.constraints
    all_parameters["modify_internal_forces"] = args.not_modify_internal_forces
    all_parameters["platform_name"] = args.platform_name
    all_parameters["continue_simulation"] = args.continue_simulation
    all_parameters["in_state_xml_path"] = args.in_state_xml_path
    all_parameters["save_final_state"] = args.save_final_state
    all_parameters["out_state_xml_path"] = args.out_state_xml_path


    all_parameters["use_AMP"] = args.use_AMP

    if (all_parameters["use_AMP"]):

        all_parameters["AMP_parameters_path"] = args.AMP_parameters_path
        
        qm_mm_zone_dict = jsonfile2dict(args.qm_mm_zones_definition)

        all_parameters["qm_zone_resnames"] = qm_mm_zone_dict["qm_zone_resnames"]

        all_parameters["mm_zone_resnames"] = qm_mm_zone_dict["mm_zone_resnames"]

        all_parameters["device_ml"] = args.device_ml
        all_parameters["weights_path"] = args.weights_path
        all_parameters["mol_charge"] = args.mol_charge
        all_parameters["scaling_charges"] = args.scaling_charges
        all_parameters["tip4p"] = args.tip4p
    
    if args.not_modify_internal_forces:
        
        all_parameters["forces_to_modify"] = jsonfile2dict(args.forces_to_modify)
    
    all_parameters["do_equilibration"] = args.equilibration
    
    if args.equilibration:
        
        all_parameters["equilibration_readout_frequency"] = args.eq_readout_frequency
        all_parameters["add_eq_csv_reporter"] = args.add_eq_csv_reporter
        all_parameters["add_eq_hdf5_reporter"] = args.add_eq_hdf5_reporter
        all_parameters["add_eq_dcd_reporter"] = args.add_eq_dcd_reporter
        all_parameters["add_eq_chk_reporter"] = args.add_eq_chk_reporter
        
        if args.add_eq_csv_reporter:
            
            all_parameters["equilibration_csv_name"] = args.eq_csv_name
            all_parameters["equilibration_csv_parameters"] = jsonfile2dict(args.eq_csv_parameters)
            
        if args.add_eq_hdf5_reporter:
            
            all_parameters["equilibration_hdf5_name"] = args.eq_hdf5_name
            all_parameters["equilibration_hdf5_parameters"] = jsonfile2dict(args.eq_hdf5_parameters)
            
        if args.add_eq_dcd_reporter:
            
            all_parameters["equilibration_dcd_name"] = args.eq_dcd_name

        if args.add_eq_chk_reporter:
            
            all_parameters["equilibration_chk_name"] = args.eq_chk_name
            
        all_parameters["equilibration_steps"] = args.equilibration_steps
        all_parameters["equilibration_step_size"] = args.equilibration_step_size
    
    
    all_parameters["add_prod_csv_reporter"] = args.add_prod_csv_reporter
    all_parameters["add_prod_hdf5_reporter"] = args.add_prod_hdf5_reporter
    all_parameters["add_prod_dcd_reporter"] = args.add_prod_dcd_reporter
    all_parameters["add_prod_chk_reporter"] = args.add_prod_chk_reporter
    all_parameters["production_steps"] = args.production_steps
    all_parameters["write_nonbonded_parameters"] = args.not_write_nonbonded_parameters
    all_parameters["initial_topology_name"] = args.initial_topology_name
    all_parameters["production_readout_frequency"] = args.prod_readout_frequency
    
    if args.add_prod_csv_reporter:
        all_parameters["production_csv_name"] = args.prod_csv_name
        all_parameters["production_csv_parameters"] = jsonfile2dict(args.prod_csv_parameters)
        
    if args.add_prod_hdf5_reporter:
        all_parameters["production_hdf5_name"] = args.prod_hdf5_name
        all_parameters["production_hdf5_parameters"] = jsonfile2dict(args.prod_hdf5_parameters)

    if args.add_prod_dcd_reporter:
        
        all_parameters["production_dcd_name"] = args.prod_dcd_name

    all_parameters["alchemical_FE"] = args.alchemical_FE
    
    if (args.alchemical_FE):

        all_parameters["alchemical_FE_definition"] = jsonfile2dict(args.alchemical_FE_definition)

    with open(args.file_path, "w") as file:
    
        yaml.dump(all_parameters,file)
           
if __name__ == '__main__':
    
    main()
    
