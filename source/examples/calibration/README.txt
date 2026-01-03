This part of the code was used for all simulations/calculations that were necessary for calibration
of ML/MM interactions (Trp-cage ML/MM simulations, water-in-water ML/MM simuilations, SAPT dimer interaction energy calculations, etc)
as well as scripts used to process and analyze the associated trajectories

'analysis' - folder containing example scripts used to analyze and postprocess the ML/MM calculations used for calibration
'data' - folder containing files used for ML/MM calibration runs (topologies, configuration files, force field parameters, etc)
'examples' - folder containing example python scripts used to submit the ML/MM calculations necessary for calibration.

The folders contain their own README files with detailed descriptions of their contents.

The scripts cannot be used without adjusting all the necessary paths as indicated in the code.
Some of the scripts assume that other data than provided by the repository is available (restart .xml files for OpenMM simulations,
equilibrated simulation boxes, SDF files for all FreeSolv database molecules, etc).
Thus, the scripts provided are not meant to be run by external users, rather they showcase the logic of how
different simulations/calculations were set up and performed.

'AMP_calibration.py' - copy of the AMP model with adjusted classes for ML/MM calibration

'AMPHelpers_calibration.py' - copy of the AMPHelpers file with adjusted classes for ML/MM calibration

'datastructures_calibration' - copy of the datastructures Graph class with adjustments for ML/MM calibration

'helper_functions.py' - various useful functions

'HFE_get_lambda_energies_ELE_LJ.py' - script used to get the potential energies of AMP and LJ
 at different lambda values for electrostatic and LJ decoupling along the given trajectory (used for HFE)

'HFE_get_lambda_energies_POL.py' - script used to get the potential energies of AMP and LJ
 at different lambda values for polarization decoupling along the given trajectory (used for HFE)

'main_calibration.py' - the main script to perform the MD simulations with AMP for Trp-cage

'main_water_in_water_calibration.py' - the main script to perform the MD simulations with AMP
 for water-in-water AMP simulations

'modules_calibration.py' - various AMP modules adapted for calibration

'Simulator_calibration.py' - file with all classes used for AMP MD simulation setups

'write_yaml.py' - the CLI script to generate the config.yaml files for AMP MD configuration (read in by main_calibration.py later)

 
