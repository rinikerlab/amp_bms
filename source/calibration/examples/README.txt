The scripts cannot be used without adjusting the 'base_path' which should point to
the user-specified directory, where all output is generated.

Scripts 'HFE_postprocessing_*' assume that the scripts "HFE_ELE_LJ_mol_012.py" and "HFE_POL_mol_012.py"
are first completed (they read-in generated trajectories).

The jupyter notebook 'filter_FreeSolv_HFE_molecules.ipynb' assumes that the user has
downloaded full FreeSolv database (adjust the path that points to SDF files of FreeSolv database).

All other scripts can be run directy from 'calibration' folder by 'python ./examples/script_name.py'

'calculate_LJ_dimer_energy.py' - example script to calculate the LJ term of dimer interaction energy
 in water(TIP4P-FB)-amino acid gas-phase dimer.

'calculate_sapt.py' - example script used to calculate the full QM (SAPT) dimer interaction energy

'charge_scaling_RDF.py' - example script to perform the ML-water-in-MM-water MD simulations with AMP using charge scaling scheme

'example_input.yaml' - example of config.yaml that is read in by "main_calibration.py" for AMP configuration of AMP MD

'filter_FreeSolv_HFE_molecules.ipynb' - example script to filter FreeSolv database to find closest neighbors to
 amino acid side chains (the filetered molecules were used for HFE calculations)

'HFE_ELE_LJ_mol_0.py' - example script to perform HFE calculations, run MD for given lambda (electrostatics and LJ decoupling,
transformation from state 3 to state 1, as defined in Supporting Information).

'HFE_POL_mol_0.py' - example script to perform HFE calculations, run MD for a given lambda (polarization decoupling,
transformation from state 1 to state 4, as defined in Supporting Information).

'HFE_postprocessing_ELE_LJ_mol_0.py' - example script to perform HFE calculations, get electrostatic AMP and LJ energies
for a set of lambdas for a given trajectory (transformation from state 3 to state 1, as defined in Supporting Information).

'HFE_postprocessing_POL_mol_0.py' - example script to perform HFE calculations, get electrostatic AMP and LJ energies
for a set of lambdas for a given trajectory (transformation from state 1 to state 4, as defined in Supporting Information).

'pol_scaling_RDF.py' - example script to perform the ML-water-in-MM-water MD simulations with
 AMP using polarization scaling scheme

'pol_scaling_TRP.py' - example script to perform the MD simuilations with AMP for Trp-cage using polarization scaling scheme


 
