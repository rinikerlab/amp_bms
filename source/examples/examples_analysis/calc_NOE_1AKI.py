import mdtraj as md
import numpy as np
import os
import sys
import re
from pathlib import Path

HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent.parent
SOURCE_DIR = EXAMPLES_DIR.parent
file_path = SOURCE_DIR / "data/1e8l.mr"

def find_missing_atoms(traj, noe_restraints):
    """
    Given an mdtraj.Trajectory and NOE restraints, find atoms not present in the PDB.
    Supports wildcards:
      *  matches any string of characters (including none)
      #  matches any string of digits (including none)
    Special cases:
      - 'HN'  in restraints -> 'H' in PDB
      - 'HB1' in restraints -> 'HB2' in PDB
      - 'HB2' in restraints -> 'HB3' in PDB
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The loaded trajectory or PDB structure.
    noe_restraints : list of tuples
        Each tuple is:
        (resid1, atomname1, resid2, atomname2, dist, lower, upper)
    
    Returns
    -------
    missing_atoms : set of (resid, atomname_pattern)
        Set of missing atoms (residue id, original atom name pattern from restraints).
    """
    topology = traj.topology

    # Build lookup: resid -> list of atom names
    pdb_atoms_by_resid = {}
    for atom in topology.atoms:
        pdb_atoms_by_resid.setdefault(atom.residue.resSeq, []).append(atom.name)

    # Special-case direct name mappings
    name_equiv = {
        "HN": "H",     # backbone amide proton
        "HB1": "HB2",  # beta proton 1
        "HB2": "HB3",  # beta proton 2
    }

    # Helper: compile wildcard pattern to regex
    def name_to_regex(pattern):
        pat = re.escape(pattern)
        # '*' → any string of characters (including none)
        pat = pat.replace(r'\*', r'.*')
        # '#' → any string of digits (including none)
        pat = pat.replace(r'\#', r'\d*')
        return re.compile(f"^{pat}$")

    missing_atoms = set()

    # Loop over restraints
    for resid1, name1, resid2, name2, *_ in noe_restraints:
        for resid, atomname in [(resid1, name1), (resid2, name2)]:
            # Apply special-case mapping
            pdb_equiv_name = name_equiv.get(atomname, atomname)

            pdb_names = pdb_atoms_by_resid.get(resid, [])
            atom_regex = name_to_regex(pdb_equiv_name)

            if not any(atom_regex.match(a) for a in pdb_names):
                missing_atoms.add((resid, atomname))  # keep original NOE name

    return missing_atoms

def parse_noe_mr(filename):
    """
    Parse an XPLOR-NIH .mr file for NOE restraints, keeping only those
    between protons in different residues.

    Returns
    -------
    restraints : list of tuples
        (resid1, name1, resid2, name2, dist, lower, upper)
    """
    restraints = []
    pattern = re.compile(
        r"ASSIGN\s+\(RESID\s+(\d+)\s+AND\s+NAME\s+(\S+)\s+\)"
        r"\s+\(RESID\s+(\d+)\s+AND\s+NAME\s+(\S+)\s+\)"
        r"\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    )
    
    parsing_noe = True  # Stop parsing when hydrogen bond restraints start

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("! Hydrogen bond restraints"):
                parsing_noe = False
                continue
            if not parsing_noe:
                continue
            if not line or line.startswith("!"):
                continue
            match = pattern.match(line)
            if match:
                resid1, name1, resid2, name2, val1, val2, val3 = match.groups()
                resid1, resid2 = int(resid1), int(resid2)

                # Keep only if both atoms are protons and in different residues
                if name1.upper().startswith("H") and name2.upper().startswith("H") and resid1 != resid2:
                    restraints.append((
                        resid1, name1,
                        resid2, name2,
                        float(val1), float(val2), float(val3)
                    ))
    return restraints

def expand_wildcard_atoms(traj, resid, atom_name_pattern):
    """
    Return atom indices matching residue and atom_name_pattern with wildcard.
    - resid: int, zero-based residue index in MDTraj
    - atom_name_pattern: string, may include '*' or '#' as wildcard at the end
      e.g. 'HG2#' matches HG21, HG22, HG23 etc.
           'HB*' matches HB1, HB2, HB3 etc.
    """
    if atom_name_pattern.endswith(('*', '#')):
        prefix = atom_name_pattern[:-1]
        atom_indices = [atom.index for atom in traj.topology.atoms
                        if atom.residue.index == resid and atom.name.startswith(prefix)]
    else:
        atom_indices = [atom.index for atom in traj.topology.atoms
                        if atom.residue.index == resid and atom.name == atom_name_pattern]
    if len(atom_indices) == 0:
        raise ValueError(f"No atoms found for resid {resid}, pattern '{atom_name_pattern}'")
    return atom_indices

def get_atom_pairs_and_symmetry(traj, restraints):
    """
    Convert restraints to atom pairs and symmetry groups, skipping those with missing atoms.
    Uses find_missing_atoms() for missing-atom detection and name conversions.
    Wildcard matches put all matching atoms in the same symmetry category.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Loaded structure.
    restraints : list of tuples
        (resid1, name1, resid2, name2, dist, lower, upper)
        residue ids are 1-based (from .mr file)

    Returns
    -------
    atom_pairs : np.ndarray, shape (N,2)
    symmetry_groups : np.ndarray, shape (N,)
    kept_restraints : list of tuples
        The subset of restraints actually used, in the same order as symmetry_groups indices.
    """
    # Use your existing function to detect missing atoms
    missing_atoms = find_missing_atoms(traj, restraints)

    # Build resid -> (atom_index, atom_name) list
    topology = traj.topology
    pdb_atoms_by_resid = {}
    for atom in topology.atoms:
        pdb_atoms_by_resid.setdefault(atom.residue.resSeq, []).append((atom.index, atom.name))

    # Same mapping dictionary from find_missing_atoms
    name_equiv = {
        "HN": "H",
        "HB1": "HB2",
        "HB2": "HB3",
    }

    # Helper: wildcard to regex
    def name_to_regex(pattern):
        pat = re.escape(pattern)
        pat = pat.replace(r'\*', r'.*')
        pat = pat.replace(r'\#', r'\d*')
        return re.compile(f"^{pat}$")

    atom_pairs = []
    symmetry_groups = []
    kept_restraints = []

    for kept_idx, (resid1, name1, resid2, name2, *_rest) in enumerate(
        r for r in restraints
        if (r[0], r[1]) not in missing_atoms and (r[2], r[3]) not in missing_atoms
    ):
        # Apply mapping rules
        name1_eq = name_equiv.get(name1, name1)
        name2_eq = name_equiv.get(name2, name2)

        # Match atoms in PDB
        atoms1 = [idx for idx, aname in pdb_atoms_by_resid.get(resid1, [])
                  if name_to_regex(name1_eq).match(aname)]
        atoms2 = [idx for idx, aname in pdb_atoms_by_resid.get(resid2, [])
                  if name_to_regex(name2_eq).match(aname)]

        if not atoms1 or not atoms2:
            continue  # safeguard, skip if no matches

        # Record the restraint in kept list
        kept_restraints.append((resid1, name1, resid2, name2, *_rest))

        # Group all matches of a restraint into same symmetry category
        for a1 in atoms1:
            for a2 in atoms2:
                atom_pairs.append([a1, a2])
                symmetry_groups.append(len(kept_restraints) - 1)  # index in kept_restraints

    return (
        np.array(atom_pairs, dtype=int),
        np.array(symmetry_groups, dtype=int),
        kept_restraints
    )

def get_atom_pairs_for_kept_restraint(atom_pairs, symmetry_groups, restraint_index):
    """
    Return all atom pairs corresponding to a restraint in kept_restraints.
    
    Parameters
    ----------
    atom_pairs : np.ndarray, shape (N,2)
        Array of atom index pairs returned by get_atom_pairs_and_symmetry.
    symmetry_groups : np.ndarray, shape (N,)
        Array mapping each atom pair to restraint index in kept_restraints.
    restraint_index : int
        Index in kept_restraints.
        
    Returns
    -------
    pairs : np.ndarray, shape (M,2)
        All atom pairs corresponding to this restraint.
    """
    mask = symmetry_groups == restraint_index
    return atom_pairs[mask]

def analyze_noe_violations(traj, atom_pairs, symmetry_groups, kept_restraints, violation_threshold=0.25):
    """
    Compute NOE distances along trajectory, average them with r^-6 averaging, 
    and check for violations against upper bounds.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory or PDB structure.
    atom_pairs : np.ndarray, shape (N,2)
        Atom pairs from get_atom_pairs_and_symmetry.
    symmetry_groups : np.ndarray, shape (N,)
        Symmetry group mapping for each atom pair.
    kept_restraints : list of tuples
        Restraints used to generate atom_pairs, each tuple: (resid1, name1, resid2, name2, dist, lower, upper)
    violation_threshold : float
        Threshold in angstroms for upper bound violation.
    
    Returns
    -------
    mean_distances : np.ndarray, shape (num_restraints,)
        r^-6 averaged distances (in nm) for each restraint.
    violations : np.ndarray, shape (num_restraints,)
        Boolean array indicating if restraint is violated.
    num_violated : int
        Number of restraints that exceed upper bound + threshold.
    """
    # Compute distances for all pairs along trajectory (in nm)
    distances_nm = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)

    num_restraints = len(kept_restraints)
    mean_distances = np.zeros(num_restraints)
    violations = np.zeros(num_restraints, dtype=bool)

    # Loop over symmetry groups (i.e., restraints)
    for i in range(num_restraints):
        mask = symmetry_groups == i
        # select distances of all pairs belonging to this restraint
        dist_subset = distances_nm[:, mask]  # shape (frames, num_pairs)
        
        # r^-6 averaging over all pairs and frames
        inv_r6 = 1.0 / dist_subset**6
        mean_r = (np.mean(inv_r6))**(-1/6)
        mean_distances[i] = mean_r
        
        # check upper bound violation
        upper_bound_angstrom = kept_restraints[i][4] + kept_restraints[i][6]  # upper bound in angstroms
        upper_bound_nm = upper_bound_angstrom / 10.0  # convert to nm
        if mean_r > upper_bound_nm + violation_threshold / 10.0:
            violations[i] = True

    num_violated = np.sum(violations)
    return mean_distances, violations, num_violated, distances_nm

def run_job(topology:str, dcd_files:list[str], save_path:str):
    trim_perc = 0.2
    # Load trajectory and reference
    traj = md.load(dcd_files, top=topology)
    # Strip solvent
    traj = traj.remove_solvent()
    n_frames = traj.n_frames
    cut_index = int(trim_perc * n_frames)

    # Slice the trajectory to remove equilibration
    trimmed_traj = traj[cut_index:]

    noe_restraints = parse_noe_mr(file_path)
    atom_pairs, symmetry_groups, kept_restraints = get_atom_pairs_and_symmetry(trimmed_traj, noe_restraints)
    mean_distances, violations, _, _ = analyze_noe_violations(trimmed_traj, atom_pairs, symmetry_groups, kept_restraints)

    results = {
        "mean_distances":mean_distances,
        "kept_restraints":kept_restraints,
        "violations":violations
    }

    # Save results
    np.save(save_path, results, allow_pickle=True)

    print(f"Saved NOE to {save_path}")


if __name__ == "__main__":
    topology = sys.argv[1]
    dcd_files = sys.argv[2:-1]
    save_path = sys.argv[-1]
    run_job(topology, dcd_files, save_path)
