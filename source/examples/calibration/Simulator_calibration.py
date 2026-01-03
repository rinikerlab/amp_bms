# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import json
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmm.app import PDBFile, Modeller, PME, Simulation
import sys
import os
from abc import abstractmethod
import time
import numpy as np

import openmm as mm
from openmm import unit as u
from helper_functions import get_atom_indices
import mdtraj as md

import torch
from openmmtorch import TorchForce
from AMP_calibration import AMP
from datastructures_calibration import Graph
from utilities_calibration import build_Rx2, load_parameters


class PDBReader():

    def __init__(self, pdb_path, box_dimension, logger=None):

        self.pdb_path = pdb_path
        self.box_dimension = box_dimension
        self.logger = logger

    def _read_pdb(self):

        self.pdbfile = PDBFile(self.pdb_path)

        if(self.logger):

            message = f"""
            The topology and atom positions are read from:  {self.pdb_path}\n
            Total number of read atoms:     {self.pdbfile.topology.getNumAtoms()}\n
            Total number of read positions:     {len(self.pdbfile.getPositions())}\n
            Total number of read residues:      {self.pdbfile.topology.getNumResidues()}\n
            """

            self.logger.info(message)

    def _prepare_modeller_from_pdbfile(self):

        self.modeller = Modeller(self.pdbfile.topology, self.pdbfile.positions)

        a = mm.vec3.Vec3(x=self.box_dimension, y=self.box_dimension*0, z=self.box_dimension*0)
        b = mm.vec3.Vec3(x=self.box_dimension*0, y=self.box_dimension, z=self.box_dimension*0)
        c = mm.vec3.Vec3(x=self.box_dimension*0, y=self.box_dimension*0, z=self.box_dimension)

        box_vectors = (a, b, c)

        self.modeller.topology.setPeriodicBoxVectors(box_vectors)

        box_vectors_with_units = self.modeller.topology.getPeriodicBoxVectors()

        if(self.logger):

            self.logger.info(f"The box dimensions are set to: \t a={box_vectors_with_units[0]} \t b={box_vectors_with_units[1]} \t c={box_vectors_with_units[2]}")

    def get_modeller(self):

        self._read_pdb()

        self._prepare_modeller_from_pdbfile()
    
        return self.modeller, self.pdbfile

class SystemBuilder():
    
    def __init__(self, topology, forcefield, nonbondedMethod=PME, cutoff_nb=10*u.angstrom, rigidWater=True, constraints=None, logger=None):
        
        self.topology = topology
        self.forcefield = forcefield
        self.nonbondedMethod = nonbondedMethod
        self.cutoff_nb = cutoff_nb
        self.rigidWater = rigidWater
        self.constraints = constraints
        self.logger = logger

    def build_system(self):

        system = self.forcefield.createSystem(self.topology, nonbondedMethod=self.nonbondedMethod,
                                            nonbondedCutoff=self.cutoff_nb, rigidWater=self.rigidWater, 
                                            constraints=self.constraints)
        
        if(self.logger):

            message = f"""The system is created with the following settings:  
            Nonbonded Method:   {self.nonbondedMethod}\n
            Nonbonded Cutoff:   {self.cutoff_nb}\n
            Rigid Water:    {self.rigidWater}\n
            Constraints:    {self.constraints}
            """

            self.logger.info(message)

        return system

class SimulationBuilder():

    def __init__(self,
        simulation_name:str,
        forcefield:mm.app.forcefield.ForceField,
        integrator,
        system,
        modeller,
        platform_name = "CUDA",
        continue_simulation:bool=False,
        state_xml_path = None,
        set_to_temperature = False,
        platform_properties=None,
        rank = None,
        alchemical_FE = False,
        scaling_lj_qm_mm = None,
        scaling_factor_alchemical_coulomb = None,
        logger = None
        ):
            
            self.simulation_name = simulation_name
            self.integrator = integrator
            self.forcefield = forcefield
            self.modeller = modeller
            self.platform_name = platform_name
            self.state_xml_path =state_xml_path
            self.continue_simulation = continue_simulation
            self.system = system
            self.set_to_temperature = set_to_temperature
            self.platform_properties = platform_properties
            self.rank = rank
            self.alchemical_FE = alchemical_FE
            self.scaling_lj_qm_mm = scaling_lj_qm_mm
            self.scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb
            self.logger = logger

            if self.logger:

                self.logger.info(f"The simulation object with name {self.simulation_name} is created!")
        
    def build_simulation(self):

        platform = mm.openmm.Platform.getPlatformByName(self.platform_name)

        self.simulation = Simulation(topology=self.modeller.topology,
                                         system=self.system,
                                         integrator=self.integrator,
                                         platform=platform,
                                         platformProperties=self.platform_properties,
                                         )

        if(self.continue_simulation and (not self.alchemical_FE)):

            self.simulation.loadState(self.state_xml_path)
            
            
            if(self.logger):

                message = f"""The simulation has been continued from the state xml file: {self.state_xml_path}
                """

                self.logger.info(message)

                
        elif (self.continue_simulation and self.alchemical_FE):
            
            self.simulation.loadState(self.state_xml_path)
            
            self.simulation.context.setParameter("scaling_lj_qm_mm", float(self.scaling_lj_qm_mm))
            
            self.simulation.context.setParameter("scaling_factor_alchemical_coulomb", float(self.scaling_factor_alchemical_coulomb))
        
            if(self.logger):

                message = f"""The alchemical FE calulation has been continued from: {self.state_xml_path}
                The scaling_lj_qm_mm is set to: {self.scaling_lj_qm_mm}
                The scaling_factor_alchemical_coulomb is set to: {self.scaling_factor_alchemical_coulomb}
                """
                
                self.logger.info(message)
        
        if (not self.continue_simulation):
            
            self.simulation.context.setPositions(self.modeller.positions)
        
            if(self.set_to_temperature):
                
                self.simulation.context.setVelocitiesToTemperature(self.integrator.getTemperature())
                
                if(self.logger):

                    message = f"""The velocities has been set to: {self.integrator.getTemperature()}
                    """

                    self.logger.info(message)
            
        return self.simulation

class SimulationRunner():
    
    def __init__(self, simulation, steps, run_type, save_final_state=False, state_xml_path=None, logger=None):
        
        self.simulation = simulation
        self.steps = steps
        self.run_type = run_type
        self.save_final_state = save_final_state
        self.state_xml_path = state_xml_path
        self.logger = logger

    def run_simulation(self):

        if self.logger:

            message = f"""Starting the {self.run_type} run.
            The simulation length is {self.simulation.integrator.getStepSize()*self.steps}
            """
                
            self.logger.info(message)

        start_time = time.time()
    
        self.simulation.step(self.steps)
            
        end_time = time.time()

        elapsed_time = end_time - start_time

        if self.logger:

            message = f"""The {self.run_type} run is successfull!
            The total time needed for the run:  {elapsed_time} s"""

            self.logger.info(message)
        
        if self.save_final_state:
            
            assert self.state_xml_path is not None
            
            self.simulation.saveState(self.state_xml_path)
            
            if self.logger:

                message = f"""The final state has been saved to XML file {self.state_xml_path}"""

                self.logger.info(message)
            
class ForcesModifier():
    
    @abstractmethod
    def modify_forces():
        pass

class BarostatAdder(ForcesModifier):

    def __init__(self, system, barostat_type, barostat_parameters, logger):
        
        self.system = system

        self.barostat_type = barostat_type

        self.barostat_parameters = barostat_parameters

        self.logger = logger

    def modify_forces(self):

        barostat = None
        
        match self.barostat_type:
        
            case "MCB":
            
                pressure = self.barostat_parameters["pressure"]*u.bar
                temp = self.barostat_parameters["temperature"]*u.kelvin
                frequency = self.barostat_parameters["frequency"]
            
                barostat = mm.openmm.MonteCarloBarostat(pressure, temp, frequency)
                
        self.system.addForce(barostat)

        if(self.logger):

            message = f"""Barostat used: {type(barostat)}\n
            Barostat pressure:  {barostat.getDefaultPressure()}\n
            Barostat temperature:   {barostat.getDefaultTemperature()}
            """
            self.logger.info(message)

        return self.system

class ReporterAdder():
    
    def __init__(self,
                 simulation,
                 readout_frequency,
                 out_folder_path,
                 logger = None):
        
        self.simulation = simulation
        self.readout_frequency = readout_frequency
        self.out_folder_path = out_folder_path
        self.logger = logger
        
    def write_initial_topology(self, file_name):
       
        init_topology_path = os.path.join(self.out_folder_path, file_name)
        
        positions = self.simulation.context.getState(getPositions=True).getPositions()

        PDBFile.writeModel(self.simulation.topology, positions, open(init_topology_path, "w"))
        
        PDBFile.writeFooter(self.simulation.topology, open(init_topology_path, "a"))
        
        if self.logger:

            message = f"""The initial topology is written to: {init_topology_path}"""

            self.logger.info(message)
            
    def add_dcd_reporter(self, file_name, append=False):
        
        self.simulation.reporters.append(mm.app.DCDReporter(os.path.join(self.out_folder_path, file_name), self.readout_frequency, append=append))

        if self.logger:

            message = f"""The DCD reporter is added to the simulation.\n
            The readout_frequency is: every {self.readout_frequency} steps.\n
            The DCD file is written to: {os.path.join(self.out_folder_path, file_name)}\n
            """

            self.logger.info(message)

        return self.simulation
        
    def add_csv_reporter(self, file_name, parameters, append=False):
        
        parameters["append"] = append

        self.simulation.reporters.append(mm.app.StateDataReporter(
                os.path.join(self.out_folder_path, file_name),
                self.readout_frequency,
                **parameters
            )                               
        )

        if self.logger:

            message = f"""The State Data Reporter is added to the simulation.\n
            The readout_frequency is: every {self.readout_frequency} steps.\n
            The CSV file is written to: {os.path.join(self.out_folder_path, file_name)}\n
            Properties to be written out: {[key for key, value in parameters.items() if value]}\n
            """

            self.logger.info(message)
        
        return self.simulation
    
    def add_hdf5_reporter(self, file_name, topology, parameters, residue_names_to_output=["CYC", "PYE"]):
        
        atom_dict = get_atom_indices(residue_names_to_output, topology)
        
        atomSubset = None
        
        for key in atom_dict.keys():
            
            if atomSubset is None:
                
                atomSubset = atom_dict[key]
                
            else:
                
                atomSubset.extend(atom_dict[key])
                
        atomSubset.sort()
        
        self.simulation.reporters.append(md.reporters.HDF5Reporter(
                file=os.path.join(self.out_folder_path, file_name),
                reportInterval=self.readout_frequency,
                atomSubset = atomSubset,
                **parameters
            )                               
        )
            
        if self.logger:

            message = f"""The HDF5 reporter from MDTraj is added to the simulation.\n
            The readout_frequency is: every {self.readout_frequency} steps.\n
            The .h5 file is written to: {os.path.join(self.out_folder_path, file_name)}\n
            The following atomSubset is written out: {atomSubset}\n
            """

            self.logger.info(message)
        
        return self.simulation
          
class ForcefieldBuilder():
    
    def __init__(self, forcefield, logger=None):
        
        self.forcefield = forcefield
        self.logger = logger
    
    def parametrize_molecules_smirnoff(self, molecules, cache_path, ff_name):
        
        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecules, cache=cache_path, forcefield=ff_name)
        
        self.forcefield.registerTemplateGenerator(smirnoff.generator)

        if self.logger:

            message = f"""The SMIRNOFFTemplateGenerator is registered.
            Molecular SMILES added to the template generator:    {[mol.to_smiles() for mol in molecules]}\n
            The forcefield used for template generator:     {ff_name}\n
            The forcefdield TemplateGenerator is cached to: {cache_path}\n
            """
            
            self.logger.info(message)
        
    def build_forcefield(self):
        
        return self.forcefield

class AmpForcesModifier(ForcesModifier):

    def __init__(self, system, topology, qm_zone, mm_zone, eps_rf, cutoff_nb, scaling_lj_qm_mm=1.0, softcore_lj_qm_mm=False, logger=None):

        self.topology = topology
        
        self.system = system
        
        self.logger = logger
        
        self.qm_zone = qm_zone
        
        self.mm_zone = mm_zone
        
        self.eps_rf = eps_rf
        
        self.cutoff_nb = cutoff_nb

        self.scaling_lj_qm_mm = scaling_lj_qm_mm

        self.softcore_lj_qm_mm = softcore_lj_qm_mm

        self.softcore_alpha = 0.5
        
        self._get_forces()
    
    def _get_forces(self):
        
        for idf, force in enumerate(self.system.getForces()):
            
            if isinstance(force, mm.openmm.HarmonicBondForce):
                self.bond_force = force
                self.id_bond_force = idf
            elif isinstance(force, mm.openmm.NonbondedForce):
                self.nb_force = force
                self.id_nb_force = idf
            elif isinstance(force, mm.openmm.PeriodicTorsionForce):
                self.torsion_force = force
                self.id_torsion_force = idf
            elif isinstance(force, mm.openmm.HarmonicAngleForce):
                self.angle_force = force
                self.id_angle_force = idf

    def _build_custom_nonbonded_mmmm(self):
        
        krf = ((self.eps_rf - 1) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb**3)
        ONE_4PI_EPS0 = 138.935456  # * u.kilojoules_per_mole*u.nanometer/(u.elementary_charge_base_unit*u.elementary_charge_base_unit)
        mrf = 4
        nrf = 6
        arfm = (3 * self.cutoff_nb ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * (
            (2 * self.eps_rf + nrf - 1) / (1 + 2 * self.eps_rf)
        )
        arfn = (3 * self.cutoff_nb ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * (
            (2 * self.eps_rf + mrf - 1) / (1 + 2 * self.eps_rf)
        )
        crf = (
            ((3 * self.eps_rf) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb)
            + arfm * self.cutoff_nb**mrf
            + arfn * self.cutoff_nb**nrf
        )
        
        crf_exp = "ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf);"
        crf_exp += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
        crf_exp += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
        crf_exp += "r6 = r2*r4;"
        crf_exp += "r4 = r2*r2;"
        crf_exp += "r2 = r*r;"
        crf_exp += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
        crf_exp += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
        crf_exp += "chargeprod = charge1*charge2;"
        crf_exp += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
        lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        lj_exp += "sigma_over_r = sigma/r;"
        lj_exp += "epsilon = sqrt(epsilon1*epsilon2);"
        lj_exp += "sigma = 0.5*(sigma1+sigma2);"        
        force_crf = mm.CustomNonbondedForce(crf_exp)
        force_crf.addPerParticleParameter("charge")
        force_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_crf.setCutoffDistance(self.cutoff_nb)
        force_crf.setName("coulomb_mm-mm")
        force_lj = mm.CustomNonbondedForce(lj_exp)
        force_lj.addPerParticleParameter("sigma")
        force_lj.addPerParticleParameter("epsilon")
        force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_lj.setCutoffDistance(self.cutoff_nb)
        force_lj.setName("lj_mm-mm")
        if (self.cutoff_nb<1.0*u.nanometer):
            force_lj.setUseLongRangeCorrection(True) # for TIP4P-FB
        else:
            force_lj.setUseLongRangeCorrection(False) # for TIP3P and OPENFF
        
        for index in range(self.nb_force.getNumParticles()):
            
            charge, sigma, epsilon = self.nb_force.getParticleParameters(index)
                     
            force_crf.addParticle([charge])
            force_lj.addParticle([sigma, epsilon])
        
        
        # Retain exceptions for mm-mm interactions
        for index in range(self.nb_force.getNumExceptions()):
            
            j, k, chargeprod, sigma, epsilon = self.nb_force.getExceptionParameters(index)
            
            if j in self.mm_zone and k in self.mm_zone:
                
                force_lj.addExclusion(j, k)
                force_crf.addExclusion(j, k)
                
        force_lj.addInteractionGroup(set(self.mm_zone), set(self.mm_zone))
        force_crf.addInteractionGroup(set(self.mm_zone), set(self.mm_zone))

        self.system.addForce(force_lj)
        self.system.addForce(force_crf)

        if(self.logger):

           message = f"""The long-range electrostatics method has been set to reaction field!
           The MM-MM LJ and custom RF interactions have been added to the system!"""

           self.logger.info(message)
    
    def _build_custom_nonbonded_qmmm(self):
        
        if self.softcore_lj_qm_mm:
            lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
            lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
            lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
            lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
            lj_exp += "sigma_over_r = sigma/reff_vdw;"
            lj_exp += "epsilon = sqrt(epsilon1*epsilon2)*scaling_lj_qm_mm;"
            lj_exp += f"reff_vdw = sigma*({self.softcore_alpha}*(1-scaling_lj_qm_mm) + (r/sigma)^6)^(1/6);"
            lj_exp += "sigma = 0.5*(sigma1+sigma2);"
            
        else:
            lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
            lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
            lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
            lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
            lj_exp += "sigma_over_r = sigma/r;"
            lj_exp += "epsilon = sqrt(epsilon1*epsilon2)*scaling_lj_qm_mm;"
            lj_exp += "sigma = 0.5*(sigma1+sigma2);"        
   
          
        force_lj = mm.CustomNonbondedForce(lj_exp)
        force_lj.addPerParticleParameter("sigma")
        force_lj.addPerParticleParameter("epsilon")
        force_lj.addGlobalParameter("scaling_lj_qm_mm", self.scaling_lj_qm_mm)
        force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_lj.setCutoffDistance(self.cutoff_nb)
        force_lj.setName("lj_qm-mm")
        if (self.cutoff_nb<1.0*u.nanometer):
            force_lj.setUseLongRangeCorrection(True) # for TIP4P-FB
        else:
            force_lj.setUseLongRangeCorrection(False) # for TIP3P and OPENFF
        
        for index in range(self.nb_force.getNumParticles()):
            
            _, sigma, epsilon = self.nb_force.getParticleParameters(index)

            force_lj.addParticle([sigma, epsilon])
        
        force_lj.addInteractionGroup(set(self.qm_zone), set(self.mm_zone))

        # Retain exceptions for mm-mm interactions
        for index in range(self.nb_force.getNumExceptions()):
            
            j, k, chargeprod, sigma, epsilon = self.nb_force.getExceptionParameters(index)
            
            if j in self.mm_zone and k in self.mm_zone:
                force_lj.addExclusion(j, k)
                
        
        self.system.addForce(force_lj)

        if(self.logger):

           message = f"""The QM-MM LJ interactions have been added to the system!"""

           self.logger.info(message)
    
    def _zero_bond_force_qm_zone(self):
        
        zero_force_constant = 0.0
        
        for bond_id in range(self.bond_force.getNumBonds()):

            bond_id_1, bond_id_2, eq_distance, k_constant = self.bond_force.getBondParameters(bond_id)

            if any((bond_id_1 not in self.qm_zone, bond_id_2 not in self.qm_zone)):

                    continue

            else:

                self.bond_force.setBondParameters(bond_id, bond_id_1, bond_id_2, eq_distance, zero_force_constant)

        if(self.logger):

            message = f"""Harmonic bond forces in the QM zone have been zeroed!"""

            self.logger.info(message)
            
    def _zero_angle_force_qm_zone(self):
        
        zero_force_constant = 0.0
    
        for angle_id in range(self.angle_force.getNumAngles()):

            angle_id_1, angle_id_2, angle_id_3, eq_distance, k_constant = self.angle_force.getAngleParameters(angle_id)

            if any((angle_id_1 not in self.qm_zone, angle_id_2 not in self.qm_zone, angle_id_3 not in self.qm_zone)):

                continue

            else:

                self.angle_force.setAngleParameters(angle_id, angle_id_1, angle_id_2, angle_id_3, eq_distance, zero_force_constant)

        
        if(self.logger):
            
            message = f"""Harmonic angle forces in the QM zone have been zeroed!"""
            self.logger.info(message)
    
    def _zero_torsion_force_qm_zone(self):
        
        zero_force_constant = 0.0
    
        for torsion_id in range(self.torsion_force.getNumTorsions()):

            torsion_id_1, torsion_id_2, torsion_id_3, torsion_id_4, periodicity, shift, k_constant = self.torsion_force.getTorsionParameters(torsion_id)

            
            if any((torsion_id_1 not in self.qm_zone, torsion_id_2 not in self.qm_zone,
                    torsion_id_3 not in self.qm_zone, torsion_id_4 not in self.qm_zone)):

                continue

            else:

                self.torsion_force.setTorsionParameters(torsion_id, torsion_id_1, torsion_id_2, torsion_id_3, torsion_id_4,
                                               periodicity, shift, zero_force_constant)
                
        if(self.logger):
            
            message = f"""Torsion forces in the QM zone have been zeroed!"""
            self.logger.info(message)
    
    def _remove_old_nonbonded_force(self):
        
        self.system.removeForce(self.id_nb_force)
        
        if(self.logger):
            
            message = f"""Old Nonbonded Force has been removed!"""
            self.logger.info(message)
    
    def modify_forces(self):
        
        self._zero_bond_force_qm_zone()
        self._zero_angle_force_qm_zone()
        self._zero_torsion_force_qm_zone()
        self._build_custom_nonbonded_mmmm()
        self._build_custom_nonbonded_qmmm()
        self._remove_old_nonbonded_force()
        
        return self.system
        
class AmpTorchForceAdder(ForcesModifier):
    
    def __init__(self, system, topology, qm_zone, charges_mm, mm_zone_charges, params_path,
                 weights_path, device_ml,scaling_factor_node_potential,
                scaling_factor_coulomb_qm, scaling_factor_coulomb_qmmm,
                scaling_factor_D4, scaling_factor_ZBL, mol_charge=0, scaling_charges=1.0, scaling_factor_alchemical_coulomb=1.0,
                logger=None):
        
        self.system = system
        self.topology = topology
        self.qm_zone = qm_zone
        self.charges_mm = charges_mm
        self.mm_zone_charges = mm_zone_charges
        self.params_path = params_path
        self.weights_path = weights_path
        self.device_ml = device_ml
        self.scaling_factor_node_potential = scaling_factor_node_potential
        self.scaling_factor_coulomb_qm = scaling_factor_coulomb_qm
        self.scaling_factor_coulomb_qmmm = scaling_factor_coulomb_qmmm
        self.scaling_factor_D4 = scaling_factor_D4
        self.scaling_factor_ZBL = scaling_factor_ZBL
        self.scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb
        self.mol_charge = mol_charge
        self.scaling_charges = scaling_charges
        self.logger = logger
        
    def _load_params(self):
        
        self.PARAMETERS = load_parameters(self.params_path)
        
    def _init_AMP_model(self):
        
        model = AMP(self.PARAMETERS).to(self.device_ml)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device_ml, weights_only=True))
        torch.jit.enable_onednn_fusion(True)        
        model_scripted = torch.jit.script(model)

        if (self.logger):

            message = f"""The AMP model has been initialized successfully!"""
            self.logger.info(message)
        
        
        force_module = ForceModule(amp=model_scripted, topology=self.topology, qm_zone=self.qm_zone,
                                   charges_mm=self.charges_mm, mm_zone_charges=self.mm_zone_charges,
                                   device=self.device_ml, mol_charge=self.mol_charge, scaling_charges=self.scaling_charges)
        
        module = torch.jit.script(force_module).to(self.device_ml)
        
        torch_force = TorchForce(module)
        
        torch_force.setUsesPeriodicBoundaryConditions(True)
        
        torch_force.addGlobalParameter('scaling_factor_node_potential', self.scaling_factor_node_potential)
        
        torch_force.addGlobalParameter('scaling_factor_coulomb_qm', self.scaling_factor_coulomb_qm)
        
        torch_force.addGlobalParameter('scaling_factor_coulomb_qmmm', self.scaling_factor_coulomb_qmmm)
        
        torch_force.addGlobalParameter('scaling_factor_D4', self.scaling_factor_D4)
        
        torch_force.addGlobalParameter('scaling_factor_ZBL', self.scaling_factor_ZBL)

        torch_force.addGlobalParameter('scaling_factor_alchemical_coulomb', self.scaling_factor_alchemical_coulomb)
        
        if (self.logger):

            message = f"""Running ML model on {self.device_ml} with single precision."""
            self.logger.info(message)

        return torch_force
    
    def modify_forces(self):
        
        self._load_params()
        torch_force = self._init_AMP_model()
        torch_force.setName("AMP")
        self.system.addForce(torch_force)
        
        return self.system        
        
class AmpConfigurator():
    
    def __init__(self, system, topology, qm_zone_definition, mm_zone_definition, eps_rf,
                 cutoff_nb, params_path, weights_path, device_ml, scaling_factor_node_potential=1.0,
                scaling_factor_coulomb_qm=1.0, scaling_factor_coulomb_qmmm=1.0,
                scaling_factor_D4=1.0, scaling_factor_ZBL=1.0, scaling_lj_qm_mm=1.0, scaling_factor_alchemical_coulomb=1.0,
                mol_charge=0, scaling_charges=1.0, tip4p=False, softcore_lj_qm_mm=False,
                logger=None):
    
        self.mm_zone_definition = mm_zone_definition
        self.qm_zone_definition = qm_zone_definition
        self.topology = topology
        self.system = system
        self.eps_rf = eps_rf
        self.cutoff_nb = cutoff_nb
        self.params_path = params_path
        self.weights_path = weights_path
        self.device_ml = device_ml
        self.scaling_factor_node_potential = scaling_factor_node_potential
        self.scaling_factor_coulomb_qm = scaling_factor_coulomb_qm
        self.scaling_factor_coulomb_qmmm = scaling_factor_coulomb_qmmm
        self.scaling_factor_D4 = scaling_factor_D4
        self.scaling_factor_ZBL = scaling_factor_ZBL
        self.scaling_lj_qm_mm = scaling_lj_qm_mm
        self.scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb
        self.mol_charge = mol_charge
        self.scaling_charges = scaling_charges
        self.tip4p = tip4p
        self.softcore_lj_qm_mm = softcore_lj_qm_mm
        self.logger = logger
    
    def _define_zones(self):
        
        qm_zone, mm_zone = [], []
        
        if all([isinstance(i, str) for i in self.qm_zone_definition]) and all([isinstance(i, str) for i in self.mm_zone_definition]):
            
            qm_data = get_atom_indices(self.qm_zone_definition, self.topology)
            
            mm_data = get_atom_indices(self.mm_zone_definition, self.topology)
            
            for key in qm_data.keys():
            
                if qm_zone is None:
                
                    qm_zone = qm_data[key]
                
                else:
                
                    qm_zone.extend(qm_data[key])
            
            for key in mm_data.keys():
            
                if mm_zone is None:
                
                    mm_zone = mm_data[key]
                
                else:
                    
                    mm_zone.extend(mm_data[key])
            
        self.qm_zone = np.array(sorted(qm_zone))
        self.mm_zone = np.array(sorted(mm_zone))
        
        if not self.tip4p:
            
            self.mm_zone_charges = np.array(sorted(mm_zone))
        else:
            
            tmp = []
            
            for atom in self.topology.atoms():
                
                if (atom.index in self.mm_zone) and (atom.name in ["H1", "H2", "M"]):
                    
                    tmp.append(atom.index)
                    
            self.mm_zone_charges = np.array(sorted(tmp))
        
        n_mm = len(self.mm_zone)
        n_qm = len(self.qm_zone)
        
        if self.logger:

            message = f"""Initialized systems with {n_qm} atoms in the QM zone and {n_mm} atoms in the MM zone."""

            self.logger.info(message)
   
    def _extract_mm_charges(self):
         
        for idf, force in enumerate(self.system.getForces()):
            
            if isinstance(force, mm.openmm.NonbondedForce) or isinstance(force, mm.openmm.CustomNonbondedForce):
                
                nb_force = force
        
        #assume that all atoms are charged in the mm zone, not applicable for TIP4P/TIP5P
        self.charges_mm = np.array([nb_force.getParticleParameters(index)[0]._value for index in self.mm_zone_charges])

    def _modify_old_forces(self):
        
        force_modifier = AmpForcesModifier(self.system, self.topology, self.qm_zone, self.mm_zone,
                                           self.eps_rf, self.cutoff_nb,
                                           self.scaling_lj_qm_mm, self.softcore_lj_qm_mm, self.logger)
        
        self.system = force_modifier.modify_forces()
    
    def _add_torch_force(self):
        
        amp_force_adder = AmpTorchForceAdder(system=self.system, topology=self.topology, qm_zone=self.qm_zone,
                                             charges_mm=self.charges_mm, mm_zone_charges=self.mm_zone_charges,
                                             params_path=self.params_path,
                                             weights_path=self.weights_path, device_ml=self.device_ml,
                                             scaling_factor_node_potential=self.scaling_factor_node_potential,
                                            scaling_factor_coulomb_qm=self.scaling_factor_coulomb_qm,
                                            scaling_factor_coulomb_qmmm=self.scaling_factor_coulomb_qmmm,
                                            scaling_factor_D4=self.scaling_factor_D4,
                                            scaling_factor_ZBL=self.scaling_factor_ZBL,
                                            scaling_factor_alchemical_coulomb=self.scaling_factor_alchemical_coulomb,
                                            mol_charge=self.mol_charge,
                                            scaling_charges=self.scaling_charges,
                                            logger=self.logger)
        
        self.system = amp_force_adder.modify_forces()
        
    def configure(self):
        
        self._define_zones()
        self._extract_mm_charges()
        self._modify_old_forces()
        self._add_torch_force()
        
        return self.system

class ForceModule(torch.nn.Module):
    def __init__(self, amp, topology, qm_zone, charges_mm, mm_zone_charges, mol_charge=0, n_nlist=64, pairlist_padding=4.0, chunk_size=10000000, block_size=4000, 
                 dtype=torch.float32, device=torch.device('cuda'), scaling_charges=1.0):
        super(ForceModule, self).__init__()        
        self.topology = topology
        self.device = device
        self.dtype = dtype
        Z = [atom.element.atomic_number for atom in self.topology.atoms() if atom.index in qm_zone]
        self.Z = torch.tensor(Z, device=self.device)
        self.register_buffer('nodes', amp.node_embedding(self.Z).detach())        
        self.register_buffer('charges_mm', torch.tensor(scaling_charges*charges_mm, device=self.device, dtype=self.dtype).unsqueeze(-1))
        self.register_buffer('mol_charge', torch.tensor(mol_charge, device=self.device, dtype=self.dtype))
        self.register_buffer('mol_size', torch.tensor([self.Z.shape[0]], device=self.device, dtype=torch.int64))
        self.register_buffer('mm_zone_charges', torch.tensor(mm_zone_charges, device=self.device))        
        self.register_buffer('qm_zone', torch.tensor(qm_zone, device=self.device))
        self.n_qm = self.qm_zone.shape[0]
        self.n_charges = self.mm_zone_charges.shape[0]
        self.register_buffer('cutoff', torch.tensor(amp.cutoff, device=self.device, dtype=self.dtype)) 
        self.register_buffer('cutoff_esp', torch.tensor(amp.cutoff_esp, device=self.device, dtype=self.dtype)) 
        self.register_buffer('cutoff_qmmm_esp', torch.tensor(amp.cutoff_qmmm_esp, device=self.device, dtype=self.dtype)) 
        self.register_buffer('cutoff_qmmm_pol', torch.tensor(amp.cutoff_qmmm_pol, device=self.device, dtype=self.dtype))
        self.register_buffer('cutoff_nlist', torch.tensor(amp.cutoff_esp + pairlist_padding, device=self.device, dtype=self.dtype))
        self.register_buffer('cutoff_qmmm_nlist', torch.tensor(amp.cutoff_qmmm_esp + pairlist_padding, device=self.device, dtype=self.dtype))
        self.register_buffer('chunk_size', torch.tensor(chunk_size, device=self.device, dtype=torch.int64))
        self.register_buffer('index_block_size', torch.tensor(block_size, device=self.device, dtype=torch.int64))
        self.register_buffer('step_count', torch.tensor(0, device=self.device, dtype=torch.int64))
        self.register_buffer('n_nlist', torch.tensor(n_nlist, device=self.device, dtype=torch.int64))
        self.register_buffer('nlist_qm', torch.tensor(0, device=self.device, dtype=torch.int64))
        self.register_buffer('nlist_mm', torch.tensor(0, device=self.device, dtype=torch.int64))
        self.register_buffer('nlist_senders', torch.tensor(0, device=self.device, dtype=torch.int64))
        self.register_buffer('nlist_receivers', torch.tensor(0, device=self.device, dtype=torch.int64))

        self.amp = amp.to(device=device, dtype=dtype).eval()

    def forward(self, positions, boxvectors, scaling_factor_node_potential,
                scaling_factor_coulomb_qm, scaling_factor_coulomb_qmmm,
                scaling_factor_D4, scaling_factor_ZBL, scaling_factor_alchemical_coulomb):
        
        scaling_factor_node_potential = scaling_factor_node_potential.to(dtype=self.dtype, device=self.device)
        scaling_factor_coulomb_qm = scaling_factor_coulomb_qm.to(dtype=self.dtype, device=self.device)
        scaling_factor_coulomb_qmmm = scaling_factor_coulomb_qmmm.to(dtype=self.dtype, device=self.device)
        scaling_factor_D4 = scaling_factor_D4.to(dtype=self.dtype, device=self.device)
        scaling_factor_ZBL = scaling_factor_ZBL.to(dtype=self.dtype, device=self.device)
        scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb.to(dtype=self.dtype, device=self.device)
        
        boxsize = torch.diag(boxvectors * 10).unsqueeze(0).to(dtype=self.dtype, device=self.device) # 
        positions = (positions * 10).to(dtype=self.dtype, device=self.device)
        graph = self._build_graph(positions, boxsize, scaling_factor_alchemical_coulomb)
        graph = self.amp(graph)
        node_potential = graph.V_nodes.squeeze() * scaling_factor_node_potential
        qm_coulomb = graph.V_coulomb_qm.squeeze() * scaling_factor_coulomb_qm
        qmmm_coulomb = graph.V_coulomb_qmmm.squeeze() * scaling_factor_coulomb_qmmm
        D4_potential = graph.V_D4.squeeze() * scaling_factor_D4
        ZBL_potential = graph.V_ZBL.squeeze() * scaling_factor_ZBL
        self.step_count = self.step_count + 1
        return node_potential+qm_coulomb+qmmm_coulomb+D4_potential+ZBL_potential
    
    def _build_graph(self, positions, boxsize, scaling_factor_alchemical_coulomb):         
        coords_qm = positions[self.qm_zone]
        coords_mm = positions[self.mm_zone_charges]     
        if self.step_count % self.n_nlist == 0:
            self.nlist_qm, self.nlist_mm = self.build_nlist_qmmm_iteratively(coords_qm, coords_mm, boxsize)
            trius = torch.triu_indices(self.n_qm, self.n_qm, offset=1, dtype=torch.long, device=self.device)
            senders_qm, receivers_qm = trius[0], trius[1]
            self.nlist_senders, self.nlist_receivers = self.build_nlist(coords_qm, coords_qm, boxsize, 
                                                                        senders_qm, receivers_qm)  
        R1_qm, Rx1_qm, senders_qm, receivers_qm = self.prepare_distances_qm(coords_qm, boxsize, self.nlist_senders, self.nlist_receivers)
        R1, R2, Rx1, Rx2, senders, receivers = self.prepare_qm_indices(R1_qm, Rx1_qm, senders_qm, receivers_qm)
        R1_esp, R2_esp, Rx1_esp, Rx2_esp, senders_esp, receivers_esp = self.prepare_esp_indices(R1_qm, Rx1_qm, senders_qm, receivers_qm)
        R1_qmmm, Rx1_qmmm, indices_qm, indices_mm = self.prepare_distances_qmmm(coords_qm, coords_mm, boxsize, self.nlist_qm, self.nlist_mm)
        R1_qmmm_esp, Rx1_qmmm_esp, Rx2_qmmm_esp, indices_qm_esp, indices_mm_esp,\
            R1_qmmm_pol, Rx1_qmmm_pol, Rx2_qmmm_pol, indices_qm_pol, indices_mm_pol =\
            self.prepare_qmmm_indices(R1_qmmm, Rx1_qmmm, indices_qm, indices_mm)
        mm_monos_esp, mm_monos_pol = self.charges_mm[indices_mm_esp]*scaling_factor_alchemical_coulomb, self.charges_mm[indices_mm_pol]*scaling_factor_alchemical_coulomb
        graph = Graph(Z=self.Z, nodes=self.nodes, coords_qm=coords_qm, 
                      mm_monos_esp=mm_monos_esp, mm_monos_pol=mm_monos_pol, 
                      mol_charge=self.mol_charge, mol_size=self.mol_size,
                      R1=R1, R2=R2, Rx1=Rx1, Rx2=Rx2, 
                      senders=senders, receivers=receivers,
                      R1_esp=R1_esp, R2_esp=R2_esp,
                      senders_esp=senders_esp, receivers_esp=receivers_esp, batch_index_esp=torch.empty(0),
                      R1_qmmm_esp=R1_qmmm_esp, Rx1_qmmm_esp=Rx1_qmmm_esp, Rx2_qmmm_esp=Rx2_qmmm_esp, 
                      receivers_qmmm_esp=indices_qm_esp, qm_indices_qmmm_esp=torch.empty(0),
                      R1_qmmm_pol=R1_qmmm_pol, Rx1_qmmm_pol=Rx1_qmmm_pol, Rx2_qmmm_pol=Rx2_qmmm_pol, 
                      receivers_qmmm_pol=indices_qm_pol,
                      md_mode=True, n_channels=self.amp.n_channels)       
        return graph
        
    def prepare_distances_qm(self, coords_qm, boxsize, senders, receivers):        
        R1_qm, Rx1_qm = ForceModuleUtilities.min_image(coords_qm, boxsize, senders, receivers)
        return R1_qm, Rx1_qm, senders, receivers

    def prepare_distances_qmmm(self, coords_qm, coords_mm, boxsize, indices_qm, indices_mm):
        R1_qmmm, Rx1_qmmm = ForceModuleUtilities.min_image_qmmm(coords_qm, coords_mm, boxsize, indices_qm, indices_mm)
        return R1_qmmm, Rx1_qmmm, indices_qm, indices_mm

    def prepare_qm_indices(self, R1_qm, Rx1_qm, senders_qm, receivers_qm):         
        cutoff_indices = torch.where(R1_qm < self.amp.cutoff)[0]
        R1 = torch.index_select(R1_qm, dim=0, index=cutoff_indices)
        Rx1 = torch.index_select(Rx1_qm, dim=0, index=cutoff_indices)
        senders_qm = torch.index_select(senders_qm, dim=0, index=cutoff_indices)
        receivers_qm = torch.index_select(receivers_qm, dim=0, index=cutoff_indices)
        R1 = torch.cat((R1, R1))
        R2 = torch.square(R1)
        Rx1 = torch.cat((Rx1, -Rx1), dim=0) / R1
        Rx2 = build_Rx2(Rx1)
        Rx1, Rx2 = Rx1.unsqueeze(1), Rx2.unsqueeze(1)
        senders = torch.cat((senders_qm, receivers_qm))
        receivers = torch.cat((receivers_qm, senders_qm))
        return R1, R2, Rx1, Rx2, senders, receivers

    def prepare_esp_indices(self, R1, Rx1, senders_qm, receivers_qm):
        cutoff_indices = torch.where(R1 < self.amp.cutoff_esp)[0]
        R1 = torch.index_select(R1, dim=0, index=cutoff_indices)
        Rx1 = torch.index_select(Rx1, dim=0, index=cutoff_indices)
        senders = torch.index_select(senders_qm, dim=0, index=cutoff_indices)
        receivers = torch.index_select(receivers_qm, dim=0, index=cutoff_indices)
        R2 = torch.square(R1)
        Rx2 = build_Rx2(Rx1)
        return R1, R2, Rx1, Rx2, senders, receivers
    
    def prepare_qmmm_indices(self, R1, Rx1, indices_qm, indices_mm):
        cutoff_indices_esp = torch.where(R1 < self.amp.cutoff_qmmm_esp)[0]
        R1_qmmm_esp = torch.index_select(R1, dim=0, index=cutoff_indices_esp)
        Rx1_qmmm_esp = torch.index_select(Rx1, dim=0, index=cutoff_indices_esp)
        Rx2_qmmm_esp = build_Rx2(Rx1_qmmm_esp)
        indices_qm_esp = torch.index_select(indices_qm, dim=0, index=cutoff_indices_esp)
        indices_mm_esp = torch.index_select(indices_mm, dim=0, index=cutoff_indices_esp)
        cutoff_indices_pol = torch.where(R1_qmmm_esp < self.amp.cutoff_qmmm_pol)[0]
        R1_qmmm_pol = torch.index_select(R1_qmmm_esp, dim=0, index=cutoff_indices_pol)
        Rx1_qmmm_pol  = torch.index_select(Rx1_qmmm_esp, dim=0, index=cutoff_indices_pol) / R1_qmmm_pol 
        Rx2_qmmm_pol = build_Rx2(Rx1_qmmm_pol)
        indices_qm_pol = torch.index_select(indices_qm_esp, dim=0, index=cutoff_indices_pol)
        indices_mm_pol = torch.index_select(indices_mm_esp, dim=0, index=cutoff_indices_pol) 
        return R1_qmmm_esp, Rx1_qmmm_esp, Rx2_qmmm_esp, indices_qm_esp, indices_mm_esp,\
                R1_qmmm_pol, Rx1_qmmm_pol, Rx2_qmmm_pol, indices_qm_pol, indices_mm_pol
    
    def build_nlist_qmmm_iteratively(self, positions_a, positions_b, boxsize):
        with torch.no_grad():
            nlist_a, nlist_b = [], []
            a, b = min(self.n_qm, self.index_block_size), min(self.n_charges, self.index_block_size)
            index_matrix = torch.full((a, b), 1, device=self.device, dtype=torch.bool)
            block_indices_qm = torch.arange(0, self.n_qm, self.index_block_size)
            block_indices_mm = torch.arange(0, self.n_charges, self.index_block_size)
            for offset_index_qm in block_indices_qm:
                end_index_qm = self.index_block_size
                if (offset_index_qm + self.index_block_size) > self.n_qm:
                    end_index_qm = self.n_qm % self.index_block_size        
                for offset_index_mm in block_indices_mm:
                    end_index_mm = self.index_block_size
                    if (offset_index_mm + self.index_block_size) > self.n_charges:
                        end_index_mm = self.n_charges % self.index_block_size
                    indices_qm, indices_mm = torch.where(index_matrix[:end_index_qm, :end_index_mm])
                    indices_qm = indices_qm + offset_index_qm
                    indices_mm = indices_mm + offset_index_mm
                    R1, Rx1 = ForceModuleUtilities.min_image_block(positions_a[indices_qm], positions_b[indices_mm], boxsize)
                    cutoff_indices = torch.where(R1.squeeze() < self.cutoff_qmmm_nlist)[0]        
                    indices_qm = torch.index_select(indices_qm, dim=0, index=cutoff_indices)
                    indices_mm = torch.index_select(indices_mm, dim=0, index=cutoff_indices)        
                    nlist_a.append(indices_qm)
                    nlist_b.append(indices_mm)
        return torch.cat(nlist_a, dim=0), torch.cat(nlist_b, dim=0)

    def build_nlist(self, positions_a, positions_b, boxsize, indices_a, indices_b):
        with torch.no_grad():
            chunks_a, chunks_b = ForceModuleUtilities.chunkify(indices_a, self.chunk_size), ForceModuleUtilities.chunkify(indices_b, self.chunk_size)
            nlist_a, nlist_b = [], []
            for chunk_a, chunk_b in zip(chunks_a, chunks_b):
                R1, Rx1 = ForceModuleUtilities.min_image_block(positions_a[chunk_a], positions_b[chunk_b], boxsize)
                cutoff_indices = torch.where(R1.squeeze() < self.cutoff_nlist)[0]
                chunk_a_nlist = torch.index_select(chunk_a, dim=0, index=cutoff_indices)
                chunk_b_nlist = torch.index_select(chunk_b, dim=0, index=cutoff_indices)
                nlist_a.append(chunk_a_nlist)
                nlist_b.append(chunk_b_nlist)
        return torch.cat(nlist_a, dim=0), torch.cat(nlist_b, dim=0)

class ForceModuleUtilities():
    
    def __init__(self):
        pass
    
    #Assuming orthorombic box    
    @staticmethod
    def to_fractional(coords, boxsize):
        return coords / boxsize
    
    @staticmethod
    def from_fractional(coords, boxsize):
        return coords * boxsize
    
    @staticmethod
    def min_image(coords, boxsize, senders, receivers):
        coords_a, coords_b = coords[senders], coords[receivers] 
        return ForceModuleUtilities.min_image_block(coords_a, coords_b, boxsize)

    @staticmethod
    def min_image_qmmm(coords_qm, coords_mm, boxsize, indices_qm, indices_mm):    
        coords_a, coords_b = coords_qm[indices_qm], coords_mm[indices_mm]
        return ForceModuleUtilities.min_image_block(coords_a, coords_b, boxsize)

    @staticmethod
    def min_image_block(coords_a, coords_b, boxsize):
        Rx1 = coords_b - coords_a
        Rx1 = Rx1 - ForceModuleUtilities.from_fractional(torch.round(ForceModuleUtilities.to_fractional(Rx1, boxsize)), boxsize)
        R1 = torch.linalg.norm(Rx1, dim=-1, keepdim=True)
        return R1, Rx1

    @staticmethod
    def chunkify(indices, chunk_size):
        return torch.split(indices, chunk_size)