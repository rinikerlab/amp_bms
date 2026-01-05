# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import os
import json
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR, timeseries
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Constants
TEMPERATURE = 298.15  # Kelvin
BOLTZMANN_CONSTANT = 8.314  # J/(mol·K)
BETA = 1 / (TEMPERATURE * BOLTZMANN_CONSTANT * 1e-3)  # 1/(kJ/mol)
KCAL_TO_KJ = 4.184
EQUILIBRATION_FRAMES = 200

N_WINDOWS = 9
LAMBDAS = [1.0, 0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.2, 0.0]

def calculate_partial_dG(DeltaF:np.ndarray, dG_eleLJ:np.ndarray, lambda_values:list[float], exp_val:float, mobley_val:float):
    results = []
    for lam in [1.0, 0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.2, 0.0]:
        if lam not in lambda_values:
            continue
        index = lambda_values.index(lam)
        pol_dG = DeltaF[0, index] / BETA
        total_dG = dG_eleLJ + pol_dG
        deviation_exp = total_dG - exp_val
        deviation_mobley = total_dG - mobley_val
        results.append({
            'lambda': lam,
            'pol_dG': pol_dG,
            'total_dG': total_dG,
            'deviation_exp': deviation_exp,
            'deviation_mobley': deviation_mobley
        })
    return results

def load_json_mapping(json_path:str):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_csv_data(csv_path:str):
    df = pd.read_csv(csv_path, comment='#', header=None, sep=";")
    df.columns = [
        'compound_id', 'SMILES', 'name',
        'experimental_value_kcal', 'experimental_uncertainty_kcal',
        'mobley_calculated_kcal', 'mobley_uncertainty_kcal',
        'experimental_ref', 'calculated_ref', 'notes'
    ]

    # Drop rows with missing required values
    df = df.dropna(subset=['compound_id', 'experimental_value_kcal', 'mobley_calculated_kcal'])

    # Convert from kcal/mol to kJ/mol
    df['experimental_value_kJ'] = df['experimental_value_kcal'] * KCAL_TO_KJ
    df['mobley_value_kJ'] = df['mobley_calculated_kcal'] * KCAL_TO_KJ

    # Create a lookup dictionary
    data = df.set_index('compound_id')[['experimental_value_kJ', 'mobley_value_kJ']].to_dict(orient='index')
    # Rename keys for consistency
    data = {k: {'exp': v['experimental_value_kJ'], 'mobley': v['mobley_value_kJ']} for k, v in data.items()}
    return data

def load_energies(path:str):
    dirs = [os.path.join(path, f"lambda_{i:03d}") for i in range(N_WINDOWS)]
    Uis = []
    for d in dirs:
        energy_file = os.path.join(d, "merged_energies.npy")
        if not os.path.isfile(energy_file):
            raise FileNotFoundError(f"Energy file not found: {energy_file}")
        Uis.append(np.load(energy_file, allow_pickle=True))
    return Uis

def compute_dG(Uij:np.ndarray, n_windows:int, label:str):
    print(f"\nComputing ΔG for {label}...")
    Uij = np.array(Uij)
    n_samples = Uij.shape[1]
    N_k = np.zeros([n_windows], np.int32)
    U_kln = np.swapaxes(Uij, 1, 2)
    for k in range(n_windows):
        _, g, _ = timeseries.detectEquilibration(U_kln[k, k, :])
        indices = timeseries.subsampleCorrelatedData(U_kln[k, k, :], g=g)
        print(f"Window {k} has {len(indices)} uncorrelated samples")
        N_k[k] = len(indices)
        U_kln[k, :, :N_k[k]] = U_kln[k, :, indices].T * BETA
        U_kln[k, :, N_k[k]:] = np.nan
    mbar = MBAR(U_kln, N_k)
    DeltaF_ij, dDeltaF_ij = mbar.getFreeEnergyDifferences()
    print(f"Predicted ΔG: {DeltaF_ij[n_windows - 1, 0]/BETA:.3f} ± {dDeltaF_ij[n_windows - 1, 0]/BETA:.3f} kJ/mol")
    return DeltaF_ij, dDeltaF_ij, U_kln, N_k, n_samples

def main(PATH:str, NPY_PATH:str):
    # Load mappings

    HERE = Path(__file__).resolve()
    EXAMPLES_DIR = HERE.parent

    json_path = EXAMPLES_DIR / "data" / "HFE_mols_FreeSolv_conversion.json"
    csv_path = "/path/to/FreeSolv.csv" #  ADJUST THIS PATH TO YOUR FreeSolv.csv LOCATION
    json_mapping = load_json_mapping(json_path)
    csv_data = load_csv_data(csv_path)

    # Identify molecule ID
    folder_name = os.path.basename(os.path.normpath(PATH))
    if folder_name not in json_mapping:
        raise ValueError(f"Folder name {folder_name} not found in JSON mapping.")
    molecule_id = json_mapping[folder_name]
    if molecule_id not in csv_data:
        raise ValueError(f"Molecule ID {molecule_id} not found in CSV data.")
    exp_val = csv_data[molecule_id]['exp']
    mobley_val = csv_data[molecule_id]['mobley']
    print(f"\nMolecule ID: {molecule_id}")
    print(f"Experimental ΔG: {exp_val:.3f} kJ/mol")
    print(f"Mobley ΔG: {mobley_val:.3f} kJ/mol")

    # Load energies
    Uis = load_energies(PATH)

    # Electrostatics
    Uij_AMP = [Uis[i][EQUILIBRATION_FRAMES:] for i in range(N_WINDOWS)]
    DeltaF_pol, dDeltaF_pol, _, _, _ = compute_dG(Uij_AMP, N_WINDOWS, "polarization")

    # Save ΔG matrices
    dG_data = {
        'DeltaF_pol': DeltaF_pol,
        'dDeltaF_pol': dDeltaF_pol,
    }
    
    npy_path = os.path.join(PATH, "dG_matrices.npy")
    
    np.save(npy_path, dG_data, allow_pickle=True)
    
    print(f"\nΔG matrices saved to {npy_path}")
    
    tmp = np.load(NPY_PATH, allow_pickle=True)
    dG_eleLJ = tmp[0]["total_dG"]

    # Partial ΔG and deviations
    results = calculate_partial_dG(DeltaF_pol, dG_eleLJ, LAMBDAS, exp_val, mobley_val)

    print("\nSummary for various λ values:")
    print(f"{'λ':>6} {'ΔG_pol':>10} {'ΔG_total':>12} {'Δ_exp':>10} {'Δ_mobley':>12}")
    for res in results:
        print(f"{res['lambda']:>6.2f} {res['pol_dG']:>10.2f} "
              f"{res['total_dG']:>12.2f} {res['deviation_exp']:>10.2f} {res['deviation_mobley']:>12.2f}")
        
    FE_npy_path = os.path.join(PATH, "FE_results.npy")
    
    np.save(FE_npy_path, results, allow_pickle=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python HFE_analysis_POL.py <path_to_data_folder> <path_to_dG_npy_file>")
        sys.exit(1)
    PATH = sys.argv[1]
    PATH_dGeleLJ = sys.argv[2]
    main(PATH, PATH_dGeleLJ)