# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

from typing import Iterable, Dict, List, Any
import json
import yaml

def get_atom_ids(residue_names:Iterable[str], pdbfile)->Dict[str, list[int]]:
    #starts counting from 1 (not from 0 as are indices in openmm)

    resnames_in_topology = set(res.name for res in pdbfile.topology.residues())

    if (not set(residue_names).issubset(resnames_in_topology)):

        raise ValueError("Some of the passsed residue names are not present in pdbfile!")

    res_names = set(residue_names)

    output = dict()

    for name in res_names:
         
         output[name] = []

    for atom in pdbfile.topology.atoms():

            if (atom.residue.name in res_names):
                 
                output[atom.residue.name].append(int(atom.id))

    return output

def get_atom_indices(residue_names:Iterable[str], topology)->Dict[str, list[int]]:

    resnames_in_topology = set(res.name for res in topology.residues())

    if (not set(residue_names).issubset(resnames_in_topology)):

        raise ValueError("Some of the passed residue names are not present in pdbfile!")

    res_names = set(residue_names)

    output = dict()

    for name in res_names:
         
         output[name] = []

    for atom in topology.atoms():

            if (atom.residue.name in res_names):
                 
                output[atom.residue.name].append(int(atom.index))

    return output

def dictoflists2set(dictionary:Dict[str, List[int]])->set:
    result_set = set()
    for sublist in dictionary.values():
        result_set.update(sublist)
    return result_set

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

def write_config_yaml(
    yaml_path: str,
    molecules_file: str,
    pdb_path: str,
    dcd_path: str,
    npy_path: str,
    box_dimension: float,
    cutoff_nb: float,
    nonbondedMethod: str,
    forcefield: str,
    cache_path: str,
    ff_name: str,
    AMP_parameters_path: str,
    weights_path: str,
    device_ml: str,
    mol_charge: float,
    qm_zone_resnames: List[str],
    mm_zone_resnames: List[str],
    scaling_charges: float,
    tip4p: bool,
    lambdas_lj: List[float],
    lambdas_coulomb: List[float],
    platform_name: str
):
    config = {
        "molecules": read_molecules_file(molecules_file),
        "pdb_path": pdb_path,
        "dcd_path": dcd_path,
        "npy_path": npy_path,
        "box_dimension": box_dimension,
        "cutoff_nb": cutoff_nb,
        "nonbondedMethod": nonbondedMethod,
        "forcefield": jsonfile2dict(forcefield),
        "cache_path": cache_path,
        "ff_name": ff_name,
        "AMP_parameters_path": AMP_parameters_path,
        "weights_path": weights_path,
        "device_ml": device_ml,
        "mol_charge": mol_charge,
        "qm_zone_resnames": qm_zone_resnames,
        "mm_zone_resnames": mm_zone_resnames,
        "scaling_charges": scaling_charges,
        "tip4p": tip4p,
        "lambdas_lj": lambdas_lj,
        "lambdas_coulomb": lambdas_coulomb,
        "platform_name": platform_name
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)