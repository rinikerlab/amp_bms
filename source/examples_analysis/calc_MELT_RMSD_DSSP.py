import mdtraj as md
import numpy as np
import sys

def get_rmsd_dssp(traj, ref_state, n_excl=0):
    """Compute RMSD (CA only) and DSSP."""
    # Restrict to CA atoms, excluding n_excl residues on each side
    atom_indices = traj.topology.select(
        f"name CA and (resid >= {n_excl} and resid <= {traj.n_residues - n_excl - 1})"
    )

    rmsd = md.rmsd(traj, ref_state, atom_indices=atom_indices) * 10  # convert nm → Å
    dssp = md.compute_dssp(traj)  # shape: (n_frames, n_residues)
    return rmsd, dssp

def compute_alpha_fraction(dssp, offset_start:int=2, offset_end:int=2):
    mask = np.where(dssp[:, offset_start:-offset_end] == 'H', 1, 0)
    return mask.sum(axis=-1) / mask.shape[-1]

def run_job(topology:str, dcd_files:list[str], save_path:str):
    
    if '1uao' in topology:
        cutoff = 2.0
    elif 'gb1' in topology:
        cutoff = 2.5
    elif '1vii' in topology:
        cutoff = 3.0
    else:
        cutoff = 2.0

    # Load trajectory and reference
    traj = md.load(dcd_files, top=topology)
    ref = md.load(topology)

    # Strip solvent
    traj = traj.remove_solvent()
    ref = ref.remove_solvent()

    # Superpose trajectory onto reference
    traj = traj.superpose(ref)

    # Compute RMSD & DSSP
    rmsd, dssp = get_rmsd_dssp(traj, ref)

    result = {}
    
    folded = rmsd < cutoff
    
    fraction_alpha = compute_alpha_fraction(dssp)
    
    result["rmsd"] = rmsd
    result["dssp"] = dssp
    result["folded"] = folded
    result["fraction_alpha"] = fraction_alpha
    
    np.save(save_path, result, allow_pickle=True)

    print(f"Saved results to {save_path}")


if __name__ == "__main__":

    topology = sys.argv[1]
    dcd_files = sys.argv[2:-1]
    save_path = sys.argv[-1]

    run_job(topology, dcd_files, save_path)
