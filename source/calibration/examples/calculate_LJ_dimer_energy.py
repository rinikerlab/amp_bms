# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import os
import json
import logging
from typing import Dict, Iterable
from openmm.app import ForceField, PDBFile, Modeller, Simulation, CutoffNonPeriodic
import openmm as mm
from openmm import unit as u
import tqdm
import numpy as np

def build_custom_nonbonded_mmmm(system, nb_force, nbforce_idf, cutoff_nb, qm_zone, mm_zone, lr_corr):
    
    eps_rf = 78.4

    krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff_nb**3)
    ONE_4PI_EPS0 = 138.935456  # * u.kilojoules_per_mole*u.nanometer/(u.elementary_charge_base_unit*u.elementary_charge_base_unit)
    mrf = 4
    nrf = 6
    arfm = (3 * cutoff_nb ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * (
        (2 * eps_rf + nrf - 1) / (1 + 2 * eps_rf)
    )
    arfn = (3 * cutoff_nb ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * (
        (2 * eps_rf + mrf - 1) / (1 + 2 * eps_rf)
    )
    crf = (
        ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff_nb)
        + arfm * cutoff_nb**mrf
        + arfn * cutoff_nb**nrf
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
    force_crf = mm.CustomNonbondedForce(crf_exp)
    force_crf.addPerParticleParameter("charge")
    force_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force_crf.setCutoffDistance(cutoff_nb)
    
    lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
    lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
    lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
    lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
    lj_exp += "sigma_over_r = sigma/r;"
    lj_exp += "epsilon = sqrt(epsilon1*epsilon2);"
    lj_exp += "sigma = 0.5*(sigma1+sigma2);"        
      
    force_lj = mm.CustomNonbondedForce(lj_exp)
    force_lj.addPerParticleParameter("sigma")
    force_lj.addPerParticleParameter("epsilon")
    force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force_lj.setCutoffDistance(cutoff_nb)
    force_lj.setUseLongRangeCorrection(lr_corr)
    
    for index in range(nb_force.getNumParticles()):
        
        charge, sigma, epsilon = nb_force.getParticleParameters(index)
        force_lj.addParticle([sigma, epsilon])
        force_crf.addParticle([charge])

    
    force_lj.addInteractionGroup(set(qm_zone), set(mm_zone))
    force_crf.addInteractionGroup(set(qm_zone), set(mm_zone))

    force_lj.setName("LJ")
    force_crf.setName("Coulomb")

    system.removeForce(nbforce_idf)

    system.addForce(force_lj) 
    system.addForce(force_crf)

    return system

def get_nbforce(system):
    
    for idf, force in enumerate(system.getForces()):
        
        if isinstance(force, mm.openmm.NonbondedForce):
            
            return force, idf
        
def get_qm_mm_indices(topology):

    qm_zone = []
    mm_zone = []

    for atom in topology.atoms():

        if (atom.residue.name =="HOH"):
             
            mm_zone.append(atom.index)
        
        else:
            qm_zone.append(atom.index)

    return sorted(qm_zone), sorted(mm_zone)

def calculate_energy_terms(pdb_path, water_ff, cutoff_nb, lr_corr):

    pdbfile = PDBFile(pdb_path)

    modeller = Modeller(pdbfile.topology, pdbfile.positions)

    forcefield = ForceField("amber14-all.xml", water_ff)

    modeller.addExtraParticles(forcefield)

    qm_zone, mm_zone = get_qm_mm_indices(modeller.topology)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, constraints=None)

    nb_force, nbforce_idf = get_nbforce(system)

    system = build_custom_nonbonded_mmmm(system=system, nb_force=nb_force, nbforce_idf=nbforce_idf,
                                        cutoff_nb=cutoff_nb*u.nanometer, qm_zone=qm_zone,
                                        mm_zone=mm_zone, lr_corr=lr_corr)

    for idx, force in enumerate(system.getForces()):

        force.setForceGroup(idx)

    integrator = mm.openmm.LangevinMiddleIntegrator(298 * u.kelvin, 1 / u.picosecond, 0.001 * u.picoseconds)

    simulation = Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)

    energies = {}

    for i,f in enumerate(system.getForces()):
        energies[f.getName()] = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()._value

    return energies

ff_water = "tip4pfb.xml"
base_path = "/path/to/base/folder" # adjust the path as necessary
cutoff_nb = 0.9
lr_corr = True
pdb_path = os.path.join(base_path, "/relative/path/to/dimer/pdb") # adjust the path as necessary (path to PDB file of water-amino acid dimer)
res_path = os.path.join(base_path, "/relative/path/to/dimer/results.npy") # adjust the path as necessary
res = calculate_energy_terms(pdb_path, water_ff=ff_water, cutoff_nb=cutoff_nb, lr_corr=lr_corr)
np.save(res_path, res, allow_pickle=True)