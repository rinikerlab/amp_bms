import mdtraj as md
import numpy as np
import sys

def get_rmsd_dssp(traj, ref_state, n_excl=5):
    """Compute RMSD (CA only) and DSSP."""
    # Restrict to CA atoms, excluding n_excl residues on each side
    atom_indices = traj.topology.select(
        f"name CA and (resid >= {n_excl} and resid <= {traj.n_residues - n_excl - 1})"
    )

    rmsd = md.rmsd(traj, ref_state, atom_indices=atom_indices) * 10  # convert nm → Å
    dssp = md.compute_dssp(traj, simplified=False)  # shape: (n_frames, n_residues)
    return rmsd, dssp

def run_job(topology:str, dcd_files:list[str], save_path_rmsd:str, save_path_dssp:str):

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

    # Save results
    np.save(save_path_rmsd, rmsd)
    np.save(save_path_dssp, dssp, allow_pickle=True)

    print(f"Saved RMSD to {save_path_rmsd} and DSSP to {save_path_dssp}")

if __name__ == "__main__":

    topology = sys.argv[1]
    dcd_files = sys.argv[2:-2]
    save_path_rmsd = sys.argv[-2]
    save_path_dssp = sys.argv[-1]

    run_job(topology, dcd_files, save_path_rmsd, save_path_dssp)
