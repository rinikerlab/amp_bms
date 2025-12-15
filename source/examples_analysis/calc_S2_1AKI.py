import mdtraj as md
import numpy as np
import sys

order_parameters = [
    (2, 0.83),
    (3, 0.83),
    (4, 0.83),
    (5, 0.85),
    (6, 0.86),
    (7, 0.88),
    (8, 0.89),
    (9, 0.93),
    (10, 0.89),
    (11, 0.89),
    (12, 0.91),
    (13, 0.92),
    (14, 0.82),
    (15, 0.84),
    (17, 0.89),
    (18, 0.86),
    (19, 0.84),
    (20, 0.85),
    (21, 0.89),
    (22, 0.99),
    (23, 0.88),
    (24, 0.89),
    (25, 0.87),
    (26, 0.91),
    (27, 0.94),
    (28, 0.87),
    (29, 0.90),
    (31, 0.93),
    (32, 0.94),
    (33, 0.91),
    (34, 0.92),
    (35, 0.88),
    (36, 0.86),
    (37, 0.96),
    (38, 0.90),
    (39, 0.89),
    (40, 0.91),
    (41, 0.86),
    (42, 0.87),
    (43, 0.83),
    (44, 0.83),
    (45, 0.78),
    (46, 0.83),
    (47, 0.78),
    (48, 0.77),
    (49, 0.82),
    (51, 0.89),
    (52, 0.89),
    (53, 0.87),
    (54, 0.91),
    (55, 0.94),
    (56, 0.92),
    (57, 0.94),
    (58, 0.90),
    (59, 0.91),
    (60, 0.93),
    (61, 0.95),
    (62, 0.85),
    (63, 0.90),
    (64, 0.91),
    (65, 0.86),
    (66, 0.89),
    (67, 0.85),
    (68, 0.78),
    (69, 0.76),
    (71, 0.72),
    (72, 0.76),
    (73, 0.88),
    (74, 0.87),
    (75, 0.94),
    (76, 0.92),
    (77, 0.90),
    (78, 0.91),
    (80, 0.91),
    (81, 0.86),
    (82, 0.88),
    (83, 0.83),
    (84, 0.83),
    (85, 0.55),
    (86, 0.80),
    (87, 0.80),
    (88, 0.80),
    (89, 0.92),
    (90, 0.91),
    (91, 0.85),
    (92, 0.93),
    (93, 0.93),
    (94, 0.92),
    (95, 0.92),
    (96, 0.92),
    (97, 0.94),
    (98, 0.92),
    (100, 0.89),
    (101, 0.85),
    (102, 0.72),
    (103, 0.82),
    (104, 0.81),
    (105, 0.88),
    (106, 0.96),
    (107, 0.91),
    (108, 0.84),
    (109, 0.85),
    (111, 0.84),
    (112, 0.89),
    (113, 0.89),
    (114, 0.87),
    (115, 0.79),
    (116, 0.84),
    (117, 0.81),
    (118, 0.72),
    (119, 0.80),
    (120, 0.80),
    (121, 0.91),
    (122, 0.92),
    (123, 0.90),
    (124, 0.90),
    (125, 0.87),
    (126, 0.82),
    (127, 0.77),
    (128, 0.76),
    (129, 0.60),
 ]

# Protein names and replicas
res_list = [(r[0], r[1]) for r in order_parameters]

def nhoparam_s2_backbone_filtered_mdtraj(traj, res_list:list[tuple[int, float]], ref_frame:int=0, trim_perc:float=0.1):
    """
    Compute S² for backbone amide N-H bonds (main chain only, no side chains),
    GROMOS++ nhoparam style (global fit), aligning only on the N atoms
    of residues for which S² is computed.

    Parameters
    ----------
    traj : md.Trajectory
        MDTraj trajectory object.
    res_list : list of tuples
        Residues to process. Each tuple is (resSeq,) with 1-based indexing,
        or (resSeq, resName). Only resSeq is matched.
    ref_frame : int
        Reference frame index for alignment (default 0).

    Returns
    -------
    s2_values : dict
        Mapping {(resSeq, resName): S² value}.
    """
    top = traj.topology

    # Only residue IDs
    allowed_resids = set(r[0] for r in res_list)

    # Fit atoms = backbone N atoms of allowed residues
    atomsfit_idx = [a.index for a in top.atoms
                    if a.name == 'N' and a.is_backbone and a.residue.resSeq in allowed_resids]

    # Align only based on those N atoms
    aligned_traj = traj.superpose(traj[ref_frame], atom_indices=atomsfit_idx, ref_atom_indices=atomsfit_idx)

    n_frames = aligned_traj.n_frames
    cut_index = int(trim_perc * n_frames)  # first 10%

    # Slice the trajectory to remove first 10%
    trimmed_traj = aligned_traj[cut_index:]

    s2_results = []

    for atom in top.atoms:
        if atom.name == 'N' and atom.is_backbone and atom.residue.resSeq in allowed_resids:
            # Find bonded hydrogen atoms via topology bonds
            bonded_H = [a.index for bond in top.bonds
                        if atom in bond
                        for a in bond
                        if a != atom and a.element.symbol == 'H']

            if not bonded_H:
                continue  # skip Prolines or missing H

            N_idx = atom.index
            H_idx = bonded_H[0]

            # Compute normalized N-H vectors
            nh_vecs = trimmed_traj.xyz[:, H_idx, :] - trimmed_traj.xyz[:, N_idx, :]
            nh_vecs /= np.linalg.norm(nh_vecs, axis=1)[:, None]

            # Calculate S²
            Q = np.einsum('ti,tj->ij', nh_vecs, nh_vecs) / trimmed_traj.n_frames
            S2 = 0.5 * (3.0 * np.sum(Q**2) - 1.0)

            s2_results.append((atom.residue.resSeq, round(S2, 3)))

    return s2_results

def run_job(topology:str, dcd_files:list[str], save_path:str, res_list:list[tuple[int, float]]):

    # Load trajectory and reference
    traj = md.load(dcd_files, top=topology)
    # Strip solvent
    traj = traj.remove_solvent()

    S2 = nhoparam_s2_backbone_filtered_mdtraj(traj, res_list, ref_frame=0, trim_perc=0.2)

    S2_sorted = sorted(S2, key=lambda x: x[0])

    # Save results
    np.save(save_path, np.array(S2_sorted), allow_pickle=True)

    print(f"Saved S2 order parameters to {save_path}")

if __name__ == "__main__":
    
    topology = sys.argv[1]
    dcd_files = sys.argv[2:-1]
    save_path = sys.argv[-1]
    run_job(topology, dcd_files, save_path, res_list)
