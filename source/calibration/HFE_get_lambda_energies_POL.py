# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import argparse
import yaml
from Simulator_calibration import (ForcefieldBuilder, SimulationBuilder, AmpConfigurator, SystemBuilder, PDBReader)
import openmm as mm
from openmm.unit import *
from openmm.app import ForceField
from openff.toolkit import Molecule
from typing import Iterable, Dict
from pathlib import Path
from openmm import unit as u
import numpy as np
import mdtraj as md
import tqdm


def load_yaml(path:str):
    
    parameters = yaml.safe_load(Path(path).read_text())
     
    return parameters

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
            
            temp = integrator_parameters["temperature"]*kelvin
            fric_coeff = integrator_parameters["friction_coefficient"]/picosecond
            step_size =  integrator_parameters["step_size"]*picoseconds
            
            integrator = mm.openmm.LangevinMiddleIntegrator(temp, fric_coeff, step_size)
    
    return integrator

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

def get_potential_energies(pdb_path:str, dcd_path:str, simulation):
    # Load the trajectory using MDTraj
    traj = md.load_dcd(dcd_path, top=pdb_path)
    
    energies_AMP = []
    energies_LJ = []

    # Iterate over all frames in the trajectory
    for frame in tqdm.tqdm(traj):

        positions = frame.xyz[0] * u.nanometer  # MDTraj outputs in nm
        boxvectors = tuple(mm.vec3.Vec3(*traj[0].unitcell_vectors[0][i]) for i in range(3))
        simulation.context.setPositions(positions)
        simulation.context.setPeriodicBoxVectors(*boxvectors)
        U_AMP = simulation.context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        U_LJ = simulation.context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        energies_AMP.append(U_AMP)
        energies_LJ.append(U_LJ)

    return np.array(energies_AMP), np.array(energies_LJ)

def main():
    
    #Read-in YAML file from command-line input 
    parser = argparse.ArgumentParser(description='This is a script that parses YAML file and calculates potential energies for a given lambda.')
    parser.add_argument("parameters", type=str, help='Path to YAML file with all parameters required for reevaluation.')
    args = parser.parse_args()
    
    #Read all parameters from YAML file
    parameters = load_yaml(args.parameters)
    logger=None
    
    #Create a forcefield for all molecules present in simulation
    forcefield = ForceField(*parameters["forcefield"]["default"])
        
    molecules = read_molecules(parameters["molecules"])
    
    if molecules != []:
        ff_builder = ForcefieldBuilder(forcefield=forcefield, logger=logger)

        ff_builder.parametrize_molecules_smirnoff(molecules=molecules, cache_path=parameters["cache_path"], ff_name=parameters["ff_name"])

        forcefield = ff_builder.build_forcefield()

    integrator = mm.openmm.LangevinMiddleIntegrator(293.15*u.kelvin, 1/u.picosecond, 0.0005*u.picosecond)
    
    pdb_reader = PDBReader(pdb_path=parameters["pdb_path"], box_dimension=parameters["box_dimension"])

    modeller, pdbfile = pdb_reader.get_modeller()

    #create system
    system_builder = SystemBuilder(topology=modeller.topology,
                                    forcefield=forcefield,
                                    cutoff_nb = parameters["cutoff_nb"]*u.nanometer,
                                    nonbondedMethod = set_nonbonded_method(parameters["nonbondedMethod"]),
                                    )

    system = system_builder.build_system()

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
                                   scaling_lj_qm_mm=1.0,
                                   scaling_factor_alchemical_coulomb = 1.0,
                                   scaling_charges=parameters["scaling_charges"],
                                   softcore_lj_qm_mm=True,
                                   tip4p=parameters["tip4p"]
                                    )

    system = amp_configurator.configure()
    for i, f in enumerate(system.getForces()):
        if f.getName() == "AMP":
            f.setForceGroup(1)
        elif f.getName() == "lj_qm-mm":
            f.setForceGroup(2)
    
    simulation_builder = SimulationBuilder(simulation_name="reeval",
                                               forcefield=forcefield,
                                               integrator=integrator,
                                               system=system,
                                               modeller=modeller,
                                               platform_name=parameters["platform_name"],
                                               )
        
    
    simulation = simulation_builder.build_simulation()

    energies_AMP, energies_LJ = get_potential_energies(pdb_path=parameters["pdb_path"],
                                      dcd_path=parameters["dcd_path"],
                                      simulation=simulation)
    
    results = {
        "energies_AMP": energies_AMP,
        "energies_LJ": energies_LJ
    }
    
    np.save(parameters["npy_path"], results, allow_pickle=True)

if __name__ == '__main__':
    
    main()