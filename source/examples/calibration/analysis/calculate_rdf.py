# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import numpy as np
from MDAnalysis.analysis.rdf import InterRDF
import MDAnalysis as mda
import os
import argparse
import sys

def compute_rdf_xxx_hoh(universe, out_path:str=None, start:int=1000):
    """
    Compute RDF between O atoms in residue 'XXX' and O atoms in water ('HOH')
    from a given MDAnalysis Universe and save the RDF to a file.

    Parameters:
    - universe (MDAnalysis.Universe): Pre-loaded universe containing trajectory.
    - out_path (str, optional): Full path for the output file. If not provided,
      defaults to 'rdf_output.txt' in the current directory.

    Output:
    - Saves RDF to the specified output file.
    """
    # Select O atoms from "XXX" and "HOH"
    xxx_O = universe.select_atoms("resname UNL and name O1")
    hoh_O = universe.select_atoms("resname HOH and name O")

    if len(xxx_O) == 0 or len(hoh_O) == 0:
        raise ValueError("No O atoms found in either UNL or HOH residues.")

    # Compute RDF
    rdf = InterRDF(xxx_O, hoh_O, nbins=250, range=(0.0, 11.0))
    rdf.run(start=start)

    # Define output filename
    if out_path is None:
        out_path = "rdf_output.txt"

    # Save RDF data
    np.savetxt(out_path, np.column_stack((rdf.bins, rdf.rdf)),
               header="Distance(A) RDF", comments='')

    print(f"Saved RDF to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RDF between UNL (O1) and HOH (O) using MDAnalysis."
    )

    parser.add_argument("--pdb", required=True,
                        help="Path to the topology .pdb file")
    parser.add_argument("--dcd", required=True,
                        help="Path to the trajectory .dcd file")
    parser.add_argument("--out", default="rdf_output.txt",
                        help="Output .txt file for RDF data")
    parser.add_argument("--start", type=int, default=1000,
                        help="Frame number at which to start RDF calculation")

    args = parser.parse_args()

    # Validate input files
    if not os.path.isfile(args.pdb):
        print(f"ERROR: pdb file does not exist: {args.pdb}")
        sys.exit(1)

    if not os.path.isfile(args.dcd):
        print(f"ERROR: dcd file does not exist: {args.dcd}")
        sys.exit(1)

    # Load universe
    print("Loading trajectory...")
    u = mda.Universe(args.pdb, args.dcd)

    # Compute RDF
    compute_rdf_xxx_hoh(
        universe=u,
        out_path=args.out,
        start=args.start
    )


if __name__ == "__main__":
    main()
            
            
    
    
