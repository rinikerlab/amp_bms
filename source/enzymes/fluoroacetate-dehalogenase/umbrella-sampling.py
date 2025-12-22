from md.Simulator import Simulator
import openmm as mm
from openmm import unit as u
from openff.toolkit import Molecule
import argparse
import os


def submit_protein(name, checkpoint_folder, output_folder, idx, steps, force_constant, xi_0, start, number, seed):
    base_folder = os.path.abspath(os.path.dirname(__file__))
    simulation_dir = os.path.join(base_folder, "simulations", f"{name}-{seed}")
    gpu_string = "--gpus=rtx_4090:1"
    hours = 4

    output = f"{simulation_dir}/{name}-{seed}-{idx}-{start}.out"
    error  = f"{simulation_dir}/{name}-{seed}-{idx}-{start}.err"
    os.system(f"sbatch -n 1 --cpus-per-task=4 {gpu_string} --time={hours}:00:00  --mem-per-cpu=4096 --output={output}  --error={error} --open-mode=truncate --wrap=\" \
                python {__file__} {name} {checkpoint_folder} {output_folder} --r0 {xi_0} --force_constant {force_constant} --index {idx} --steps {steps} --start {start} --number {number} --seed {seed}\"")

def current_cv(simulator, pullingForce):
    cv1_value, cv2_value = pullingForce.getCollectiveVariableValues(simulator.simulation.context)
    current_cv_value = cv2_value - cv1_value
    return current_cv_value * 10, cv1_value * 10, cv2_value * 10

def get_position(simulator, idx1, idx2):
    positions = simulator.simulation.context.getState(getPositions=True).getPositions()
    dist = mm.unit.unit_math.norm(positions[idx1]-positions[idx2]).value_in_unit(u.angstrom)
    return dist

parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Name of the system.")
parser.add_argument("checkpoint_folder", type=str, help="Folder with checkpoints from SMD simulation.")
parser.add_argument("output_folder", type=str, help="Output folder base path for simulation.")
parser.add_argument("--steps", type=int, default=10000, help="Number of steps for production run.")
parser.add_argument("--index", type=int, default=0, help="Index of the umbrella.")
parser.add_argument("--start", type=int, default=0, help="Restart point.")
parser.add_argument("--number", type=int, default=3, help="Number of repeats to run.")
parser.add_argument("--force_constant", type=float, default=10, help="Force constant in kJ / (mol A **2).")
parser.add_argument("--r0", type=float, default=0.0, help="Target distance in Angstrom.")
parser.add_argument("--reporter", type=int, default=2000, help="Reporter and checkpoint frequency.")
parser.add_argument("--cv_reporter", type=int, default=200, help="Reporter of collective variable.")
parser.add_argument("--temperature", type=int, default=298, help="Temperature of the simulation in K.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
args = parser.parse_args()

assert args.start < args.number 
# output_folder: basepath + simulation-specific path
output_folder = os.path.join(args.output_folder, f"{args.system}-{args.seed}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# parameters
fc_pull = args.force_constant * u.kilojoules_per_mole / u.angstrom ** 2 

molecules = [Molecule.from_file("hydroxide.sdf")]
print(molecules)

# pdb comes from SMD simulation
pdb_input = os.path.join(args.checkpoint_folder, f"window_{args.index}.pdb")
print(f"Loading initial structure from: {pdb_input}")

# initialize simulation object
simulator = Simulator(output_folder=f"{output_folder}/{args.system}-{args.index}-{args.start}/",
                      report_frequency=args.reporter,
                      dT = 0.5 * u.femtosecond,
                      T = args.temperature * u.kelvin,
                      constraints=None,
                      padding=30 * u.angstrom,
                      ff_additional="protein.xml")

simulator(pdb_input=pdb_input,
          molecules=molecules,
          minimize=False,
          restart=True,
          fix=False)

if "Asp" in args.system:
    offset = 2
else:
    offset = 0

# restrain system
o1x_idx = 9255 - 2 * offset
h1_idx = 9256 - 2 * offset

# his
his_h1_idx = 4318 - 1 * offset
his_n_idx = 4321 - 1 * offset
his_h2_idx = 4322 - 1 * offset

# asb
if "Asp" in args.system:
    asb_c_idx = 1653
else:
    asb_c_idx = 1653

atoms = [o1x_idx, h1_idx, his_h1_idx, his_n_idx, his_h2_idx, asb_c_idx]
for atom in simulator.simulation.topology.atoms():
    if atom.index in [o1x_idx, asb_c_idx, his_h2_idx, his_n_idx]:
        print(atom)

print("restraining indices:", o1x_idx, asb_c_idx, his_h2_idx)

# define collective variables and add potential
cv1 = mm.CustomBondForce("r")
cv2 = mm.CustomBondForce("r")
cv1.addBond(o1x_idx, his_h2_idx) # 1.0
cv2.addBond(o1x_idx, asb_c_idx) # 2.9
pullingForce = mm.CustomCVForce("0.5 * fc_pull * (cv2-cv1-r0)^2")
pullingForce.addGlobalParameter("fc_pull", fc_pull)
pullingForce.addGlobalParameter("r0", args.r0 * u.angstrom)
pullingForce.addCollectiveVariable("cv1", cv1)
pullingForce.addCollectiveVariable("cv2", cv2)
simulator.simulation.system.addForce(pullingForce)
simulator.simulation.context.reinitialize(preserveState=True)

print(simulator.simulation.context.getParameter("r0"))
print(simulator.simulation.context.getParameter("fc_pull"))

if args.start > 0:
    print(f"Loading checkpoint from {args.start - 1}...")
    simulator.simulation.loadState(os.path.join(output_folder, f"{args.system}-{args.index}-{args.start - 1}", "checkpoint.xml"))  
else:
    checkpoint_path = os.path.join(args.checkpoint_folder, f"checkpoint_{args.index}.xml")
    print(f"Loading checkpoint from SMD: {checkpoint_path}")
    simulator.simulation.loadState(checkpoint_path)  
    simulator.simulation.context.setVelocitiesToTemperature(args.temperature, args.seed)

print(f"Runing production with fc_pull {fc_pull} and {args.r0} for {args.steps} steps")

# re-initialize parameters in case they were overwritten by SMD
simulator.simulation.context.setParameter("r0", args.r0 * u.angstrom)
simulator.simulation.context.setParameter("fc_pull", fc_pull)

# production
with open(os.path.join(output_folder, f"{args.system}-{args.index}-{args.start}", "report.dat"), "w") as f:
    for i in range(args.steps // args.cv_reporter):
        cv, cv1, cv2 = current_cv(simulator, pullingForce)
        o1x_his_h = get_position(simulator, o1x_idx, his_h2_idx), 
        o1x_h = get_position(simulator, o1x_idx, h1_idx), 
        his_n_his_h = get_position(simulator, his_n_idx, his_h2_idx)
        f.write(f"{i * args.cv_reporter}, {cv}, {cv1}, {cv2}, {o1x_his_h}, {o1x_h}, {his_n_his_h}\n")
        simulator.simulation.step(args.cv_reporter)

if (args.start + 1) < args.number:
        submit_protein(args.system, 
                       args.checkpoint_folder,
                       args.output_folder,
                       args.index, 
                       args.steps, 
                       args.force_constant, 
                       args.r0, 
                       args.start + 1, 
                       args.number, 
                       args.seed)

print(f"Simulation {args.start} finished.")