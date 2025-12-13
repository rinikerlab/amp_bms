# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import numpy as np
import os
import psi4
import argparse
import json

def read_xyz_file(file_path:str):
    """
    Reads an XYZ file, skips the first two lines, and returns the rest.

    Parameters:
        file_path (str): Path to the XYZ file.

    Returns:
        list of str: Lines from the file, excluding the first two.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return "".join(lines[2:])  # Skip the first two lines
    
def read_charge_multiplicity(json_path:str):
    """
    Reads a JSON file and extracts charge and multiplicity.

    Parameters:
        json_path (str): Path to the JSON file.

    Returns:
        tuple: (charge, multiplicity)
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
        charge = data.get('charge')
        multiplicity = data.get('multiplicity')
        return charge, multiplicity

def run_sapt(
    xyz_a:str, xyz_b:str,
    charge_a:int, multiplicity_a:int,
    charge_b:int, multiplicity_b:int,
    log_file:str, results_file:str,
    memory:int=40, num_threads:int=8,
    basis:str= "aug-cc-pvtz",
):
    
    mol_string = f"""
{charge_a} {multiplicity_a}
{xyz_a.strip()}
--
{charge_b} {multiplicity_b}
{xyz_b.strip()}
units angstrom
no_com
no_reorient
"""
    mol = psi4.geometry(mol_string)
    
    mem = f"{memory} GB"
    # Set Psi4 options
    psi4.set_memory(mem)
    psi4.core.set_num_threads(num_threads)
    psi4.set_options({
        'basis': basis,
        'scf_type': 'df'
    })

    # Run SAPT calculation
    psi4.set_output_file(log_file, False)
    energy = psi4.energy("sapt2+", molecule=mol)


    # Define the terms you're interested in
    terms = [
        'SAPT ELST ENERGY',
        'SAPT EXCH ENERGY',
        'SAPT IND ENERGY',
        'SAPT DISP ENERGY',
        'SAPT TOTAL ENERGY',
        'SAPT2+ TOTAL ENERGY',
    ]

    # Convert to kJ/mol
    hartree_to_kjmol = 2625.49962
    sapt_terms_kjmol={}
    
    for term in terms:

        sapt_terms_kjmol[term] = psi4.variable(term)*hartree_to_kjmol

    # Save results
    np.save(results_file, sapt_terms_kjmol, allow_pickle=True)

def main():
    
    arg_parser = argparse.ArgumentParser(description="Run SAPT calculations using Psi4.")
    arg_parser.add_argument('xyz_1', type=str,  help='Path to the first XYZ file.')
    arg_parser.add_argument('xyz_2', type=str,  help='Path to the second XYZ file.')
    arg_parser.add_argument('json_1', type=str, help='Path to the first JSON file with charge and multiplicity.')
    arg_parser.add_argument('json_2', type=str,  help='Path to the second JSON file with charge and multiplicity.')
    arg_parser.add_argument('log_file', type=str, help='Path to the .log file.')
    arg_parser.add_argument('results_file', type=str, help='Path to the .npy file to save results.')
    arg_parser.add_argument('--memory_per_cpu', type=float, default=5, help='Memory per cpu in GB.')
    arg_parser.add_argument('--basis', type=str, default="aug-cc-pvdz", help='Basis set to use for Psi4.')
    arg_parser.add_argument('--num_threads', type=int, default=8, help='Number of threads to use for Psi4.')
    
    
    args = arg_parser.parse_args()
    
    # Read the XYZ files
    xyz_1 = read_xyz_file(args.xyz_1)
    xyz_2 = read_xyz_file(args.xyz_2)
    
    # Read the charge and multiplicity from the JSON files
    charge_1, multiplicity_1 = read_charge_multiplicity(args.json_1)
    charge_2, multiplicity_2 = read_charge_multiplicity(args.json_2)
    
    mem = int(args.memory_per_cpu * args.num_threads)

    # Run the SAPT calculation
    run_sapt(xyz_a=xyz_1, xyz_b=xyz_2,charge_a=charge_1, multiplicity_a=multiplicity_1,
             charge_b=charge_2, multiplicity_b=multiplicity_2,
             log_file=args.log_file, results_file=args.results_file,
             memory=mem, num_threads=args.num_threads,
             basis=args.basis)
    
if __name__ == "__main__":
    main()