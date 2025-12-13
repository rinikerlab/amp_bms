# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import argparse
import yaml
import logging
from Simulator_calibration import (ForcefieldBuilder, SimulationBuilder, ReporterAdder,
                                    SimulationRunner, AmpConfigurator, SystemBuilder,
                                    PDBReader, BarostatAdder)

import openmm as mm
from openmm.openmm import XmlSerializer
from openmm.app import ForceField
from openff.toolkit import Molecule
import os
from typing import Iterable, Dict
from pathlib import Path
from openmm import unit as u
import numpy as np

def load_yaml(path:str):
    
    parameters = yaml.safe_load(Path(path).read_text())
     
    return parameters

def create_logger(path:str):
   
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler(path)

    file_handler.setLevel(logging.DEBUG) 

    f_format = logging.Formatter('%(asctime)s - %(message)s')

    file_handler.setFormatter(f_format)

    logger.addHandler(file_handler)

    return logger

def read_molecules(mol_definitions:Iterable[str]):
    
    molecules = []
    
    for mol_definition in mol_definitions:
        
        if (mol_definition.endswith(".sdf")):
            
            molecule = Molecule(mol_definition)
        
        else:
            
            molecule = Molecule.from_smiles(mol_definition)
            
        molecules.append(molecule)
        
    return molecules

def set_integrator(integrator_type:str, integrator_parameters:Dict):
    
    match integrator_type:
        
        case "LMI":
            
            temp = integrator_parameters["temperature"]*u.kelvin
            fric_coeff = integrator_parameters["friction_coefficient"]/u.picosecond
            step_size =  integrator_parameters["step_size"]*u.picoseconds
            
            integrator = mm.openmm.LangevinMiddleIntegrator(temp, fric_coeff, step_size)
    
    return integrator

def set_constraint(constraint_type:str):
    
    match constraint_type:
        
        case "HBonds":
            
            constraint = mm.app.forcefield.HBonds
        
        case "":

            constraint = None
            
    return constraint

def set_nonbonded_method(nonbonded_method_type:str):
    
    match nonbonded_method_type:
        
        case "PME":
            
            nonbonded_method = mm.app.forcefield.PME
        
        case "NoCutoff":

            nonbonded_method = mm.app.forcefield.NoCutoff
        
        case "CutoffNonPeriodic":

            nonbonded_method = mm.app.forcefield.CutoffNonPeriodic

        case "CutoffPeriodic":

            nonbonded_method = mm.app.forcefield.CutoffPeriodic
        
        case "Ewald":

            nonbonded_method = mm.app.forcefield.Ewald
        
        case "LJPME":

            nonbonded_method = mm.app.forcefield.LJPME
    
    return nonbonded_method

def read_system_xml(path, logger=None):

    with open(path) as input:

        system = XmlSerializer.deserialize(input.read())

    if(logger):
            
        message = f"""The system has been read from XML file: {path}"""

        logger.info(message)
    
    return system

def write_system_xml(path, system, logger=None):

    with open(path, 'w') as output:
    
        output.write(XmlSerializer.serialize(system))
    
    if(logger):
            
        message = f"""The system has been written to XML file: {path}"""

        logger.info(message)

def main():
    
    #Read-in YAML file from command-line input 
    parser = argparse.ArgumentParser(description='This is a script that parses YAML file and performs the OpenMM simulation.')
    parser.add_argument("parameters", type=str, help='Path to YAML file with all parameters required for simulation.')
    args = parser.parse_args()
    
    #Read all parameters from YAML file
    parameters = load_yaml(args.parameters)

    #Create simulation folder
    simulation_folder_path = parameters["base_path"]
    os.makedirs(simulation_folder_path, exist_ok=True)
        
    #Set logger
    if (parameters["set_logger"]):
        
        logger = create_logger(os.path.join(simulation_folder_path, "allInfo.log"))
    
    else:
        
        logger=None
    
    #Create a forcefield for all molecules present in simulation
    forcefield = ForceField(*parameters["forcefield"]["default"])
        
    molecules = read_molecules(parameters["molecules"])
    
    if molecules != []:

        ff_builder = ForcefieldBuilder(forcefield=forcefield, logger=logger)

        ff_builder.parametrize_molecules_smirnoff(molecules=molecules, cache_path=parameters["cache_path"], ff_name=parameters["ff_name"])

        forcefield = ff_builder.build_forcefield()
        
    else:
        if logger:
            
            message = "The molecules list is empty. Assume the all residues are known by the forcefield"
            logger.info(message)

    integrator = set_integrator(integrator_type=parameters["integrator_type"], integrator_parameters=parameters["integrator_parameters"])
    
    pdb_reader = PDBReader(pdb_path=parameters["pdb_path"], box_dimension=parameters["box_dimension"], logger=logger)

    modeller, pdbfile = pdb_reader.get_modeller()
    
    if (parameters["tip4p"]):
        modeller.addSolvent(forcefield, model="tip4pew", padding=2.0*u.nanometer, neutralize=False)
    else:
        modeller.addSolvent(forcefield, model="tip3p", padding=2.0*u.nanometer, neutralize=False)

    #create system
    system_builder = SystemBuilder(topology=modeller.topology,
                                    forcefield=forcefield,
                                    cutoff_nb = parameters["cutoff_nb"]*u.nanometer,
                                    nonbondedMethod = set_nonbonded_method(parameters["nonbondedMethod"]),
                                    rigidWater = parameters["rigidWater"],
                                    constraints = set_constraint(parameters["constraints"]),
                                    logger=logger
                                    )

    system = system_builder.build_system()

    if(parameters["use_barostat"]):

        barostat_adder = BarostatAdder(system=system,
                                       barostat_type=parameters["barostat_type"],
                                       barostat_parameters=parameters["barostat_parameters"],
                                       logger=logger
                                       )
        
        system = barostat_adder.modify_forces()

    #Modify forces to use AMP3 (do QM/MM with QM engine AMP3)
    if (parameters["use_AMP"]):

        #Modify forces to use AMP3 (do QM/MM with QM engine AMP3)
        parameters_ml = load_yaml(parameters["AMP_parameters_path"])
        
        amp_configurator = AmpConfigurator(system=system,
                                               topology=modeller.topology,
                                               qm_zone_definition=parameters["qm_zone_resnames"],
                                               mm_zone_definition=parameters["mm_zone_resnames"],
                                               eps_rf=parameters_ml["eps_rf"],
                                               cutoff_nb=parameters["cutoff_nb"]*u.nanometer,
                                               params_path=parameters["AMP_parameters_path"],
                                               weights_path=parameters["weights_path"],
                                               device_ml=parameters["device_ml"],
                                               mol_charge=parameters["mol_charge"],
                                               scaling_charges=parameters["scaling_charges"],
                                               tip4p=parameters["tip4p"],
                                               logger=logger)

        system = amp_configurator.configure()
    
    platform_properties = None
        
    simulation_builder = SimulationBuilder(simulation_name=parameters["simulation_name"],
                                               forcefield=forcefield,
                                               integrator=integrator,
                                               system=system,
                                               modeller=modeller,
                                               platform_name=parameters["platform_name"],
                                               continue_simulation=parameters["continue_simulation"],
                                               state_xml_path=parameters["in_state_xml_path"],
                                               platform_properties=platform_properties,
                                               logger = logger
                                               )
        
    simulation = simulation_builder.build_simulation()

    ##Do energy minimization
    if(parameters["minimize"] and not parameters["continue_simulation"]):

        simulation.minimizeEnergy(maxIterations=parameters["maxIterations"])
        
        if(logger):
            
            message = f"""The system has been successfully minimized!!!"""

            logger.info(message)    
        
    #Do equilibration run
    if(parameters["do_equilibration"] and not parameters["continue_simulation"]):
        
        equilibration_folder_path = os.path.join(simulation_folder_path, "equilibration")
        
        os.makedirs(equilibration_folder_path, exist_ok=True)
        
        reporter_adder = ReporterAdder(simulation=simulation,
                                       out_folder_path=equilibration_folder_path,
                                       readout_frequency=parameters["equilibration_readout_frequency"],
                                       logger=logger)
        
        
        reporter_adder.write_initial_topology(file_name=parameters["initial_topology_name"])
        
        if(parameters["add_eq_csv_reporter"]):
            
            simulation = reporter_adder.add_csv_reporter(file_name=parameters["equilibration_csv_name"],
                                        parameters=parameters["equilibration_csv_parameters"])
            
        if(parameters["add_eq_hdf5_reporter"]):
            
            simulation = reporter_adder.add_hdf5_reporter(file_name=parameters["equilibration_hdf5_name"],
                                        parameters=parameters["equilibration_hdf5_parameters"])
        
        if(parameters["add_eq_dcd_reporter"]):
            
            simulation = reporter_adder.add_dcd_reporter(file_name=parameters["equilibration_dcd_name"])
        
        
        simulation.integrator.setStepSize(parameters["equilibration_step_size"])
            
        simulation_runner = SimulationRunner(simulation=simulation,
                                             steps=parameters["equilibration_steps"],
                                             run_type="equilibration",
                                             logger=logger)
        
        simulation_runner.run_simulation()
        
        #After Equilibration is completed set step to 0 and clear reporters
        simulation.currentStep = 0
        
        simulation.reporters.clear()
        
    #Do production run
    production_folder_path = os.path.join(simulation_folder_path, parameters["simulation_phase"])
        
    os.makedirs(production_folder_path, exist_ok=True)

    reporter_adder = ReporterAdder(simulation=simulation,
                                       out_folder_path=production_folder_path,
                                       readout_frequency=parameters["production_readout_frequency"],
                                       logger=logger)

    reporter_adder.write_initial_topology(file_name=parameters["initial_topology_name"])

    if(parameters["add_prod_csv_reporter"]):
        simulation = reporter_adder.add_csv_reporter(file_name=parameters["production_csv_name"],
                                        parameters=parameters["production_csv_parameters"])
    if(parameters["add_prod_hdf5_reporter"]):
        simulation = reporter_adder.add_hdf5_reporter(file_name=parameters["production_hdf5_name"],
                                        parameters=parameters["production_hdf5_parameters"],
                                        topology=simulation.topology)
    if(parameters["add_prod_dcd_reporter"]):
        simulation = reporter_adder.add_dcd_reporter(file_name=parameters["production_dcd_name"])
    
    simulation.integrator.setStepSize(parameters["integrator_parameters"]["step_size"])

    simulation_runner = SimulationRunner(simulation=simulation,
                                         steps=parameters["production_steps"],
                                         run_type="production",
                                         save_final_state = parameters["save_final_state"],
                                         state_xml_path=parameters["out_state_xml_path"],
                                         logger=logger)
    
    simulation_runner.run_simulation()
    

if __name__ == '__main__':
    
    main()