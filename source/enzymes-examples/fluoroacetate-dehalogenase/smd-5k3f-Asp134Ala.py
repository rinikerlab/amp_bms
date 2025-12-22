from md.Simulator import Simulator
from openmm.app import *
from openmm import *
from openmm.unit import *
from openff.toolkit import Molecule
import os
import numpy as np
import argparse

def current_cv(simulator, pullingForce):
    cv1_value, cv2_value = pullingForce.getCollectiveVariableValues(simulator.simulation.context)
    current_cv_value =  cv2_value - cv1_value
    return current_cv_value * 10, cv2_value * 10, cv1_value * 10

def get_position(simulator, idx1, idx2):
    positions = simulator.simulation.context.getState(getPositions=True).getPositions()
    dist = norm(positions[idx1]-positions[idx2]).value_in_unit(angstrom)
    return dist

def print_constraints(simulator, atoms):
    for idx in range(simulator.simulation.system.getNumConstraints()):
        constraint = simulator.simulation.system.getConstraintParameters(idx)
        if constraint[0] in atoms or constraint[1] in atoms:
            print(constraint)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100000, help="Number of steps for production run.")
parser.add_argument("--minimization", type=int, default=10, help="Minimization steps for initial simulation.")
parser.add_argument("--equilibration", type=int, default=5000, help="Number of steps for equilibration run.")
parser.add_argument("--force_constant", type=float, default=500, help="Force constant in kJ / (mol A **2).")
parser.add_argument("--reporter", type=int, default=1000, help="Reporter and checkpoint frequency.")
parser.add_argument("--cv_reporter", type=int, default=200, help="Reporter of collective variable.")
parser.add_argument("--temperature", type=int, default=298, help="Temperature of the simulation in K.")
parser.add_argument("--fix", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Fix the simulation or not.")
parser.add_argument("--seed", type=int, default=130185, help="Seed of the simulation.")
args = parser.parse_args()

basename = "5k3f-Asp134Ala"
system = f"{basename}-equilibrated"

# output_folder: basepath + simulation-specific path
output_folder = os.path.join("smd", f"{basename}-smd")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# define pdb depending on restart condition
pdb_input = f"{system}.pdb"
molecules = [Molecule.from_file("hydroxide.sdf")]


# initialize simulation object
simulator = Simulator(output_folder=f"{output_folder}/",
                      report_frequency=args.reporter,
                      dT = 0.5 * femtosecond,
                      T = 298,
                      constraints=None,
                      padding=30 * angstrom,
                      ff_additional="forcefield-asb/protein.xml")


simulator(pdb_input=pdb_input,
          molecules=molecules,
          minimize=False,
          fix=False,
          restart=True)

# water
o1x_idx = 9251
h1_idx = 9252

# his
his_h1_idx = 4316
his_n_idx = 4319
his_h2_idx = 4320

# asb
asb_c_idx = 1653

atoms = [o1x_idx, h1_idx, his_h1_idx, his_n_idx, his_h2_idx, asb_c_idx]

print("%.2f, %.2f, %.2f" % (get_position(simulator, o1x_idx, his_h2_idx), 
                            get_position(simulator, o1x_idx, h1_idx), 
                            get_position(simulator, his_n_idx, his_h2_idx)))
print_constraints(simulator, atoms)
simulator.simulation.minimizeEnergy(maxIterations=5)

# parameters
fc_pull = args.force_constant * kilojoules_per_mole / angstrom ** 2 
fc_pull_eq = 100 * kilojoules_per_mole / angstrom ** 2
r0 = 3.0 * angstrom
v_pulling = 0.2 * angstrom / picosecond 
dt = simulator.simulation.integrator.getStepSize()
increment_steps = 5
tolerance = 0.1 # allow small deviations from reaction coordinate

print("restraining indices:", o1x_idx, asb_c_idx, his_h2_idx, his_n_idx)

for atom in simulator.simulation.topology.atoms():
    if atom.index in [o1x_idx, asb_c_idx, his_h2_idx, his_n_idx]:
        print(atom)

# define collective variables and add potential
cv1 = CustomBondForce("r")
cv2 = CustomBondForce("r")
cv1.addBond(o1x_idx, his_h2_idx) # 1.0
cv2.addBond(o1x_idx, asb_c_idx) # 2.9
pullingForce = CustomCVForce("0.5 * fc_pull * (cv2-cv1-r0)^2")
pullingForce.addGlobalParameter("fc_pull", fc_pull_eq)
pullingForce.addGlobalParameter("r0", r0 * angstrom)
pullingForce.addCollectiveVariable("cv1", cv1)
pullingForce.addCollectiveVariable("cv2", cv2)
simulator.simulation.system.addForce(pullingForce)

simulator.simulation.context.reinitialize(preserveState=True)

cvs = current_cv(simulator, pullingForce)
print(cvs)

# # equilibration
print(f"Runing equilibration for {args.equilibration} steps with {fc_pull_eq}")
simulator.simulation.context.setVelocitiesToTemperature(args.temperature, args.seed)
simulator.simulation.step(args.equilibration)
print("%.2f, %.2f, %.2f" % (get_position(simulator, o1x_idx, his_h2_idx), 
                            get_position(simulator, o1x_idx, h1_idx), 
                            get_position(simulator, his_n_idx, his_h2_idx)))

cvs = current_cv(simulator, pullingForce)
print(cvs)

# production

epsilon = 1e-3
small_step = 0.1
big_step = 0.2
first_half = np.arange(-3.0, -1.6 + epsilon, big_step) # 180 
second_half = np.arange(1.6, 3.0 + epsilon, big_step) # 180
transition_state = np.arange(-1.5, 1.5  + epsilon, small_step) # 1800
windows = np.concatenate((first_half, transition_state, second_half))
windows = windows[::-1]

print("windows")
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
        print("%.2f, %.2f, %.2f" % (get_position(simulator, o1x_idx, his_h2_idx), 
                                    get_position(simulator, o1x_idx, h1_idx), 
                                    get_position(simulator, his_n_idx, his_h2_idx)))

    # increment the location of the CV based on the pulling velocity
    r0 -= v_pulling * dt * increment_steps
    simulator.simulation.context.setParameter('r0',r0)

    # check if we should save this config as a window starting structure
    delta_r = np.abs(current_cv_value - windows[window_index])
    if (window_index < len(windows) and delta_r < tolerance):
        print(f"adding: {window_index} with {cvs[0]:.2f}, {cvs[1]:.2f}, {cvs[2]:.2f} for windows={windows[window_index]:.2f} at delta {delta_r:.2f}")
        positions = simulator.simulation.context.getState(getPositions=True).getPositions()
        outfile = open(os.path.join(output_folder, f'window_{window_index}.pdb'), 'w')
        PDBFile.writeFile(simulator.simulation.topology, positions, outfile)
        simulator.simulation.saveState(os.path.join(output_folder, f'checkpoint_{window_index}.xml'))  
        window_index += 1
    
    if window_index == len(windows): break

print(f"SMD finished.")
