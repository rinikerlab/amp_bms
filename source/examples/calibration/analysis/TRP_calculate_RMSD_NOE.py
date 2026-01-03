# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import sys
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
import mdtraj as md


def compute_ca_rmsd(pdb_file:str, dcd_file:str, output_file:str):
    """Compute RMSD for Cα atoms excluding first 2 and last 2 residues."""
    u = mda.Universe(pdb_file, dcd_file)

    # Select Cα atoms excluding first 2 and last 2 residues
    ca_sel = "name CA"
    rmsd_calc = RMSD(u, u, select=ca_sel, ref_frame=0)
    rmsd_calc.run()

    np.savetxt(output_file, rmsd_calc.rmsd, delimiter=',', header='Time(ps),Frame,RMSD(Å)', comments='')
    print(f"RMSD saved to {output_file}")


def compute_noe_distances(pdb_file:str, dcd_file:str, output_file:str):
    """Compute NOE distances using MDTraj based on predefined pairs."""

    data = [
    ("resid 3 and name HE1", "resid 18 and name HB2", 3.50, 0.60, 0.90, {"group": 0}), # HE* Y3 HB2 P18
    ("resid 3 and name HE2", "resid 18 and name HB2", 3.50, 0.60, 0.90, {"group": 0}), # HE* Y3 HB2 P18
    ("resid 3 and name HA", "resid 19 and name HB2", 4.00, 0.70,  1.00, {"group": 1}), # HA Y3 HB2 P19
    ("resid 3 and name HA", "resid 19 and name HD2", 3.50, 0.60,  0.50, {"group": 2}), # HA Y3 HD2 P19
    ("resid 3 and name HA", "resid 19 and name HG2", 3.50, 0.60,  0.50, {"group": 3}), # HA Y3 and HG* P19
    ("resid 3 and name HA", "resid 19 and name HG3", 3.50, 0.60,  0.50, {"group": 3}), # HA Y3 and HG* P19
    ("resid 3 and name HB2", "resid 19 and name HG2", 3.50,  0.60,  0.70, {"group": 4}), # HB2 Y3 and HG* P19
    ("resid 3 and name HB2", "resid 19 and name HG3", 3.50,  0.60,  0.70, {"group": 4}), # HB2 Y3 and HG* P19
    ("resid 6 and name HZ2", "resid 12 and name HA", 2.50,0.50,0.50, {"group": 5}), # HZ2 W6 and HA P12
    ("resid 6 and name HH2", "resid 12 and name HG2", 3.50,  0.60,  0.70, {"group": 6}), # HH2 W6 and HG* P12
    ("resid 6 and name HH2", "resid 12 and name HG3", 3.50,  0.60,  0.70, {"group": 6}), # HH2 W6 and HG* P12
    ("resid 6 and name HH2", "resid 12 and name HA", 4.00, 0.70, 1.00, {"group": 7}), # HH2 W6 and HA P12
    ("resid 6 and name HD1", "resid 16 and name HB2", 3.50,  0.60,  0.70, {"group": 8}), # HD1 W6 and HB* R16
    ("resid 6 and name HD1", "resid 16 and name HB3", 3.50,  0.60,  0.70, {"group": 8}), # HD1 W6 and HB* R16
    ("resid 6 and name HE1", "resid 16 and name HB2", 3.50,  0.60,  0.70, {"group": 9}), # HE1 W6 and HB* R16
    ("resid 6 and name HE1", "resid 16 and name HB3", 3.50,  0.60,  0.70, {"group": 9}), # HE1 W6 and HB* R16
    ("resid 6 and name HE1", "resid 17 and name HA", 3.50,  0.60,  0.50, {"group": 10}), # HE1 W6 and HA P17
    ("resid 6 and name HZ2", "resid 17 and name HA", 4.00,  0.70,  1.00, {"group": 11}), # HZ2 W6 and HA P17
    ("resid 6 and name HD1", "resid 16 and name HD2", 3.50,  0.60,  0.70, {"group": 12}), # HD1 W6 and HD* R16
    ("resid 6 and name HD1", "resid 16 and name HD3", 3.50,  0.60,  0.70, {"group": 12}), # HD1 W6 and HD* R16
    ("resid 6 and name HD1", "resid 16 and name HG2", 3.50,  0.60,  0.70, {"group": 13}), # HD1 W6 and HG* R16
    ("resid 6 and name HD1", "resid 16 and name HG3", 3.50,  0.60,  0.70, {"group": 13}), # HD1 W6 and HG* R16
    ("resid 6 and name HZ2", "resid 18 and name HD2", 4.00,  0.70,  1.00, {"group": 14}), # HZ2 W6 and HD2 P18
    ("resid 6 and name HE1", "resid 18 and name HA", 3.50,  0.60,  0.50, {"group": 15}), # HE1 W6 and HA P18
    ("resid 6 and name HD1", "resid 18 and name HA",   4.00,  0.70,  1.00, {"group": 16}), # HD1 W6 and HA P18
    ("resid 6 and name HD1", "resid 19 and name HD2",   4.00,  0.70,  1.00, {"group": 17}), # HD1 W6 and HD2 P19
    ("resid 9 and name HB2", "resid 14 and name HB2",  3.50,  0.60,  0.50, {"group": 18}), # HB1 D9 and HB2 S14
    ("resid 9 and name HB3", "resid 14 and name HB2",  3.50,  0.60,  0.50, {"group": 18}), # HB1 D9 and HB2 S14
    ("resid 6 and name HH2", "resid 12 and name HD2", 3.00,  0.50,  0.50, {"group": 19}), # HH2 W6 and HD1 P12
    ("resid 6 and name HH2", "resid 12 and name HD3", 3.00,  0.50,  0.50, {"group": 19}), # HH2 W6 and HD1 P12
    ("resid 6 and name HE1", "resid 16 and name NH2", 4.00,  0.70,  1.00, {"group": 20}), # HE1 W6 and HN R16
    ("resid 6 and name HZ2", "resid 18 and name HD2", 3.50,  0.60,  0.50, {"group": 21}), # HZ2 W6 and HD1 P18
    ("resid 6 and name HZ2", "resid 18 and name HD3", 3.50,  0.60,  0.50, {"group": 21}), # HZ2 W6 and HD1 P18
    ("resid 6 and name HZ2", "resid 18 and name HB2", 4.00,  0.70,  1.00, {"group": 22}), # HZ2 W6 and HB1 P18
    ("resid 6 and name HZ2", "resid 18 and name HB3", 4.00,  0.70,  1.00, {"group": 22}), # HZ2 W6 and HB1 P18
    ("resid 6 and name HZ2", "resid 18 and name HG2", 4.00,  0.70,  1.00, {"group": 23}), # HZ2 W6 and HG1 P18
    ("resid 6 and name HZ2", "resid 18 and name HG3", 4.00,  0.70,  1.00, {"group": 23}), # HZ2 W6 and HG1 P18
    ("resid 6 and name HH2", "resid 18 and name HB2", 4.00,  0.70,  1.00, {"group": 24}), # HH2 W6 and HB1 P18
    ("resid 6 and name HH2", "resid 18 and name HB3", 4.00,  0.70,  1.00, {"group": 24}), # HH2 W6 and HB1 P18
    ("resid 6 and name HH2", "resid 18 and name HB2", 4.00,  0.70,  1.00, {"group": 25}), # HH2 W6 and HB1 P18
    ("resid 6 and name HH2", "resid 18 and name HB3", 4.00,  0.70,  1.00, {"group": 25}), # HH2 W6 and HB1 P18
    ("resid 6 and name HH2", "resid 18 and name HG2", 4.00,  0.70,  1.00, {"group": 26}), # HH2 W6 and HG1 P18
    ("resid 6 and name HH2", "resid 18 and name HG3", 4.00,  0.70,  1.00, {"group": 26}), # HH2 W6 and HG1 P18
    ("resid 7 and name HD21", "resid 12 and name HD2",  3.50,  0.60,  0.70, {"group": 27}), # HD2* L7 and HD1 P12
    ("resid 7 and name HD21", "resid 12 and name HD3",  3.50,  0.60,  0.70, {"group": 27}), # HD2* L7 and HD1 P12
    ("resid 7 and name HD22", "resid 12 and name HD2",  3.50,  0.60,  0.70, {"group": 27}), # HD2* L7 and HD1 P12
    ("resid 7 and name HD22", "resid 12 and name HD3",  3.50,  0.60,  0.70, {"group": 27}), # HD2* L7 and HD1 P12
    ("resid 7 and name HD23", "resid 12 and name HD2",  3.50,  0.60,  0.70, {"group": 27}), # HD2* L7 and HD1 P12
    ("resid 7 and name HD23", "resid 12 and name HD3",  3.50,  0.60,  0.70, {"group": 27}), # HD2* L7 and HD1 P12
    ]

    traj = md.load_dcd(dcd_file, top=pdb_file)

    # Get atom pairs as indices
    atom_pairs = []
    for sel1, sel2, *_ in data:
        atom1 = traj.topology.select(sel1)
        atom2 = traj.topology.select(sel2)
        if len(atom1) != 1 or len(atom2) != 1:
            raise ValueError(f"Selections must return exactly one atom each, got {len(atom1)} and {len(atom2)}")
        atom_pairs.append([atom1[0], atom2[0]])

    atom_pairs = np.array(atom_pairs)

    # Compute distances (in nm), convert to angstroms (1 nm = 10 Å)
    distances_nm = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)
    distances_angstrom = distances_nm * 10.0

    # Save the distances
    print(f"Saving distances to {output_file}...")
    np.save(output_file, distances_angstrom)


def main(pdb_file:str, dcd_file:str):
    output_dir = os.path.dirname(os.path.abspath(dcd_file))
    rmsd_file = os.path.join(output_dir, "rmsd.csv")
    noe_file = os.path.join(output_dir, "noe_distances.npy")

    compute_ca_rmsd(pdb_file, dcd_file, rmsd_file)
    compute_noe_distances(pdb_file, dcd_file, noe_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python TRP_calculate_RMSD_NOE.py <input.pdb> <input.dcd>")
        sys.exit(1)

    pdb_file = sys.argv[1]
    dcd_file = sys.argv[2]
    main(pdb_file, dcd_file)
