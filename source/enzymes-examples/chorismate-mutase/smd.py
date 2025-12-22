from md.Simulator import Simulator
import openmm as mm
from openmm import unit as u
from openff.toolkit import Molecule
import argparse
import os
import numpy as np

def current_cv(simulator, pullingForce):
    cv1_value, cv2_value = pullingForce.getCollectiveVariableValues(simulator.simulation.context)
    current_cv_value = cv2_value - cv1_value
    return current_cv_value * 10, cv1_value * 10, cv2_value * 10

parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Name of the system.")
parser.add_argument("c1_idx", type=int, help="PDB index of C1")
parser.add_argument("output_folder", type=str, help="Output folder base path for simulation.")
parser.add_argument("--protein", type=str, help="Name of the protein with pdb of the same name.")
parser.add_argument("--ligand", type=str, help="Name of the ligand with sdf of the same name.")
parser.add_argument("--steps", type=int, default=100000, help="Number of steps for production run.")
parser.add_argument("--minimization", type=int, default=10, help="Minimization steps for initial simulation.")
parser.add_argument("--equilibration", type=int, default=20000, help="Number of steps for equilibration run.")
parser.add_argument("--force_constant", type=float, default=200, help="Force constant in kJ / (mol A **2).")
parser.add_argument("--reporter", type=int, default=2000, help="Reporter and checkpoint frequency.")
parser.add_argument("--cv_reporter", type=int, default=200, help="Reporter of collective variable.")
parser.add_argument("--temperature", type=int, default=298, help="Temperature of the simulation in K.")
parser.add_argument("--fix", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Fix the simulation or not.")
parser.add_argument("--seed", type=int, default=130185, help="Seed of the simulation.")
args = parser.parse_args()

# output_folder: basepath + simulation-specific path
output_folder = os.path.join(args.output_folder, f"{args.system}-{args.seed}-smd")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ligand if required
if args.ligand:
    sdf = f"{args.ligand}.sdf"
    molecules = [Molecule.from_file(sdf)]
else:
    molecules = None

# define pdb depending on restart condition
if args.protein: 
    pdb_input = f"{args.protein}.pdb"
else: 
    pdb_input = None

# initialize simulation object
simulator = Simulator(output_folder=f"{output_folder}/",
                      report_frequency=args.reporter,
                      dT = 0.5 * u.femtosecond,
                      T = args.temperature * u.kelvin,
                      constraints=None,
                      padding=30 * u.angstrom)

minimize = True if args.minimization > 0 else False

simulator(pdb_input=pdb_input,
          molecules=molecules,
          minimize=minimize,
          max_iterations=args.minimization,
          fix=args.fix)

# parameters
fc_pull = args.force_constant * u.kilojoules_per_mole / u.angstrom ** 2 
fc_pull_eq = 1000 * u.kilojoules_per_mole / u.angstrom ** 2
r0 = -2.7 * u.angstrom
v_pulling = 0.2 * u.angstrom / u.picosecond 
dt = simulator.simulation.integrator.getStepSize()
increment_steps = 1
tolerance = 0.4 # allow small deviations from reaction coordinate

# restrain system
c1_idx = args.c1_idx
c6_idx = c1_idx + 12
o3_idx = c1_idx + 7
c4_idx = c1_idx + 8
print("restraining indices:", c1_idx, c6_idx, o3_idx, c4_idx)

# define reaction coordinate (subtract offset to account for TER in PDB file and 0 indexing in C++)
if args.protein:
    c1_idx -= 4
    c6_idx -= 4
    o3_idx -= 4
    c4_idx -= 4
else:
    c1_idx -= 1
    c6_idx -= 1
    o3_idx -= 1
    c4_idx -= 1
print("(corrected) restraining indices:", c1_idx, c6_idx, o3_idx, c4_idx)

# define collective variables and add potential
cv1 = mm.CustomBondForce("r")
cv2 = mm.CustomBondForce("r")
cv1.addBond(c6_idx, c1_idx)
cv2.addBond(c4_idx, o3_idx)
pullingForce = mm.CustomCVForce("0.5 * fc_pull * (cv2-cv1-r0)^2")
pullingForce.addGlobalParameter("fc_pull", fc_pull_eq)
pullingForce.addGlobalParameter("r0", r0 * u.angstrom)
pullingForce.addCollectiveVariable("cv1", cv1)
pullingForce.addCollectiveVariable("cv2", cv2)
simulator.simulation.system.addForce(pullingForce)

if not args.protein:
    # restrain dihedrals to give reasonable TS structures
    c4_idx = c1_idx + 8 
    o3_idx = c1_idx + 7
    c2_idx = c1_idx + 3
    c3_idx = c1_idx + 4

    o4_idx = c1_idx + 19
    c9_idx = c1_idx + 17
    c4_idx = c1_idx + 8 
    o3_idx = c1_idx + 7

    restraint = mm.PeriodicTorsionForce()
    simulator.simulation.system.addForce(restraint)
    restraint.addTorsion(c4_idx, o3_idx, c2_idx, c3_idx, 1, 100*u.degrees, 1*u.kilojoules_per_mole)
    restraint.addTorsion(o4_idx, c9_idx, c4_idx, o3_idx, 1, -154*u.degrees, 1*u.kilojoules_per_mole)


simulator.simulation.context.reinitialize(preserveState=True)

# equilibration
print(f"Runing equilibration for {args.equilibration} steps with {fc_pull_eq}")
simulator.simulation.context.setVelocitiesToTemperature(args.temperature, args.seed)
simulator.simulation.step(args.equilibration)

# production

epsilon = 1e-3
small_step = 0.1
big_step = 0.2
first_half = np.arange(-2.6, -1.6 + epsilon, big_step) # 180 
second_half = np.arange(1.6, 2.6 + epsilon, big_step) # 180
transition_state = np.arange(-1.5, 1.5  + epsilon, small_step) # 1800
windows = np.concatenate((first_half, transition_state, second_half))
print(windows)
window_coords = []
window_index = 0

print(f"changing fc_pull to: {fc_pull}")
simulator.simulation.context.setParameter('fc_pull',fc_pull)

# SMD pulling loop
for i in range(args.steps//increment_steps):
    simulator.simulation.step(increment_steps)
    cvs = current_cv(simulator, pullingForce)
    current_cv_value = cvs[0]

    if (i*increment_steps)%args.reporter == 0:
        print("r0 = ", r0, "r = ", cvs)

    # increment the location of the CV based on the pulling velocity
    r0 += v_pulling * dt * increment_steps
    simulator.simulation.context.setParameter('r0',r0)

    # check if we should save this config as a window starting structure
    delta_r = np.abs(current_cv_value - windows[window_index])
    if (window_index < len(windows) and delta_r < tolerance):
        print(f"adding: {window_index} with {cvs[0]:.2f}, {cvs[1]:.2f}, {cvs[2]:.2f} for windows={windows[window_index]:.2f} at delta {delta_r:.2f}")
        window_coords.append(simulator.simulation.context.getState(getPositions=True).getPositions())
        simulator.simulation.saveState(os.path.join(output_folder, f'checkpoint_{window_index}.xml'))  
        window_index += 1
    
    if window_index == len(windows): break

# save the window structures
for i, coords in enumerate(window_coords):
    outfile = open(os.path.join(output_folder, f'window_{i}.pdb'), 'w')
    mm.app.PDBFile.writeFile(simulator.simulation.topology,coords, outfile)

print(f"SMD finished.")