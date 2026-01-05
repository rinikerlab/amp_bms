import os
from md.Simulator import Simulator
import openmm as mm
from openmm.app import PDBFile
from pathlib import Path

HERE = Path(__file__).resolve()
EXAMPLES_DIR = HERE.parent.parent
SOURCE_DIR = EXAMPLES_DIR.parent

OUTPUT_FOLDER = Path("/path/to/output/folder") # ADJUST
SIMULATION_LENGTH = 100000 # 100'000 x 0.5fs = 50 picoseconds
REPORT_FREQUENCY = 1000 # in steps
SYSTEM_PDB = str(SOURCE_DIR / "data/melting_curves_topologies/gb1_capped.pdb")
TEMPERATURE = 298.15 # in Kelvin
STEP_SIZE = 0.5 # in femtoseconds
system_xml_path = OUTPUT_FOLDER / "system.xml"
init_topology_path = OUTPUT_FOLDER / "init_topology.pdb"
final_state_xml_path = OUTPUT_FOLDER / "state.xml"

simulator = Simulator(T=TEMPERATURE,
                      output_folder=str(OUTPUT_FOLDER)+"/", 
                      report_frequency=REPORT_FREQUENCY, 
                      dT=STEP_SIZE*mm.unit.femtosecond
                      )

simulation = simulator(SYSTEM_PDB, minimize=True, fix=False)

simulation.minimizeEnergy()

positions = simulation.context.getState(getPositions=True).getPositions()

PDBFile.writeFile(simulation.topology, positions, open(init_topology_path, "w"))

with open(system_xml_path, 'w') as output:

    output.write(mm.XmlSerializer.serialize(simulation.system))

simulation.step(SIMULATION_LENGTH)

simulation.saveState(final_state_xml_path)