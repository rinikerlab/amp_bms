
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

# Constants
TEMPERATURE = 298.15  # Kelvin
BOLTZMANN_CONSTANT = 8.314  # J/(mol·K)
BETA = 1 / (TEMPERATURE * BOLTZMANN_CONSTANT * 1e-3)  # 1/(kJ/mol)
KCAL_TO_KJ = 4.184
EQUILIBRATION_FRAMES = 200

# Lambda values
N_WINDOWS_ELE = 11
N_WINDOWS_LJ = 11
N_WINDOWS = 21
LAMBDA_ELE = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.67, 0.5, 0.33, 0.167, 0.0] + [0.0]*(N_WINDOWS_LJ - 1)
LAMBDA_LJ = [1.0]*(N_WINDOWS_ELE-1) + [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
assert len(LAMBDA_ELE) == len(LAMBDA_LJ) == N_WINDOWS, "Lambda length mismatch"

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
        energy_file = os.path.join(d, "energies.npy")
        if not os.path.isfile(energy_file):
            raise FileNotFoundError(f"Energy file not found: {energy_file}")
        Uis.append(np.load(energy_file, allow_pickle=True).item())
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

def plot_FE_profile(DeltaF0:np.ndarray, exp_val:float, mobley_val:float, path:str):
    plt.plot(DeltaF0/BETA, label=r"$\Delta G$ per window", marker="o")
    labels = [
        (
            ("Full" if i == 0 else "")
            + (f"{(LAMBDA_ELE if i < N_WINDOWS_ELE else LAMBDA_LJ)[i]:.2f}"
            f"{'Q' if i < N_WINDOWS_ELE else 'LJ'}"
            if 0 < i < N_WINDOWS-1 else "")
            + ("Dummy" if i == N_WINDOWS-1 else "")
        )
        for i in range(N_WINDOWS)
    ]
    plt.xticks(range(N_WINDOWS), labels, rotation=45, ha="right")
    plt.xlabel("Window")
    plt.ylabel(r"$\Delta$G (kJ/mol)")
    plt.axhline(exp_val, color="red", label="Experimental value", ls="--")
    plt.axhline(mobley_val, color="green", label="FreeSolv MBAR value", ls="--")
    plt.legend()
    plt.tight_layout()
    plot_filename = os.path.join(path, "FE_profile.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Free energy profile plot saved as {plot_filename}")

def calculate_partial_dG(DeltaF_ele:np.ndarray, DeltaF_LJ:np.ndarray, lambda_values:list[float], exp_val:float, mobley_val:float):
    results = []
    for lam in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7]:
        if lam not in lambda_values:
            continue
        index = lambda_values.index(lam)
        ele_index = N_WINDOWS_ELE - 1
        ele_dG = DeltaF_ele[ele_index, index] / BETA
        lj_dG = DeltaF_LJ[N_WINDOWS_LJ - 1, 0] / BETA
        total_dG = ele_dG + lj_dG
        deviation_exp = total_dG - exp_val
        deviation_mobley = total_dG - mobley_val
        results.append({
            'lambda': lam,
            'ele_dG': ele_dG,
            'lj_dG': lj_dG,
            'total_dG': total_dG,
            'deviation_exp': deviation_exp,
            'deviation_mobley': deviation_mobley
        })
    return results

def main(PATH:str):
    # Load mappings
    json_path = "../data/SFE_mols_FreeSolv_conversion.json" # if needded adjust the path
    csv_path = "../data/FreeSolv.csv" # if needded adjust the path
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
    Uij_AMP = [Uis[i]['energies_AMP'][EQUILIBRATION_FRAMES:] for i in range(N_WINDOWS)]
    Uij_AMP = [u[:, :N_WINDOWS_ELE] for u in Uij_AMP[:N_WINDOWS_ELE]]
    DeltaF_ele, dDeltaF_ele, _, _, _ = compute_dG(Uij_AMP, N_WINDOWS_ELE, "electrostatics")

    # LJ
    Uij_LJ = [Uis[i]['energies_LJ'][EQUILIBRATION_FRAMES:] for i in range(N_WINDOWS)]
    Uij_LJ = [u[:, N_WINDOWS_ELE - 1:] for u in Uij_LJ[N_WINDOWS_ELE - 1:]]
    DeltaF_LJ, dDeltaF_LJ, _, _, _ = compute_dG(Uij_LJ, N_WINDOWS_LJ, "LJ")

    # Plot full FE profile
    DeltaF0 = np.concatenate([DeltaF_ele[:,0], DeltaF_LJ[1:,0]+ DeltaF_ele[-1,0]])
    plot_FE_profile(DeltaF0, exp_val, mobley_val, PATH)

    # Save ΔG matrices
    dG_data = {
        'DeltaF_ele': DeltaF_ele,
        'dDeltaF_ele': dDeltaF_ele,
        'DeltaF_LJ': DeltaF_LJ,
        'dDeltaF_LJ': dDeltaF_LJ,
    }
    npy_path = os.path.join(PATH, "dG_matrices.npy")
    
    np.save(npy_path, dG_data, allow_pickle=True)
    
    print(f"\nΔG matrices saved to {npy_path}")

    # Partial ΔG and deviations
    results = calculate_partial_dG(DeltaF_ele, DeltaF_LJ, LAMBDA_ELE, exp_val, mobley_val)

    print("\nSummary for various λ values:")
    print(f"{'λ':>6} {'ΔG_ele':>10} {'ΔG_LJ':>10} {'ΔG_total':>12} {'Δ_exp':>10} {'Δ_mobley':>12}")
    for res in results:
        print(f"{res['lambda']:>6.2f} {res['ele_dG']:>10.2f} {res['lj_dG']:>10.2f} "
              f"{res['total_dG']:>12.2f} {res['deviation_exp']:>10.2f} {res['deviation_mobley']:>12.2f}")
        
    FE_npy_path = os.path.join(PATH, "FE_results.npy")
    
    np.save(FE_npy_path, results, allow_pickle=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python HFE_analysis_ELE_LJ.py <path_to_data_folder>")
        sys.exit(1)
    PATH = sys.argv[1]
    main(PATH)