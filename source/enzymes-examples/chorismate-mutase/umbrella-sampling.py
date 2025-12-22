from md.Simulator import Simulator
import openmm as mm
from openmm import unit as u
from openff.toolkit import Molecule
import argparse
import os

def submit_ligand(name, c1_idx, checkpoint_folder, output_folder, time, cpu, memory, idx, steps, force_constant, xi_0, start, number, ligand, seed):
    simulation_folder = os.path.join(output_folder, f"{args.system}-{args.seed}")
    output = os.path.join(simulation_folder, f"{name}-{idx}-{start}.out")
    error = os.path.join(simulation_folder, f"{name}-{idx}-{start}.err")
    os.system(f"sbatch -n 1 --gpus=1 --gres=gpumem:11g --cpus-per-task={cpu} --time={time}:00:00  --mem-per-cpu={memory} --output={output}  --error={error} --open-mode=truncate --wrap=\" \
                python {__file__} {name} {c1_idx} {checkpoint_folder} {output_folder} --index {idx} --ligand {ligand} --steps {steps} --r0 {xi_0}  --force_constant {force_constant}  --start {start} --number {number} --seed {seed}\"")

def submit_protein(name, c1_idx, checkpoint_folder, output_folder, time, cpu, memory, idx, steps, force_constant, xi_0, start, number, ligand, seed):
    simulation_folder = os.path.join(output_folder, f"{args.system}-{args.seed}")
    output = os.path.join(simulation_folder, f"{name}-{idx}-{start}.out")
    error = os.path.join(simulation_folder, f"{name}-{idx}-{start}.err")
    os.system(f"sbatch -n 1 --gpus=1 --gres=gpumem:11g --cpus-per-task={cpu} --time={time}:00:00  --mem-per-cpu={memory} --output={output}  --error={error} --open-mode=truncate --wrap=\" \
                python {__file__} {name} {c1_idx} {checkpoint_folder} {output_folder} --index {idx}  --protein --ligand {ligand} --steps {steps} --r0 {xi_0}  --force_constant {force_constant}  --start {start} --number {number} --seed {seed}\"")

def current_cv(simulator, pullingForce):
    cv1_value, cv2_value = pullingForce.getCollectiveVariableValues(simulator.simulation.context)
    current_cv_value = cv2_value - cv1_value
    return current_cv_value * 10, cv1_value * 10, cv2_value * 10


parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Name of the system.")
parser.add_argument("c1_idx", type=int, help="PDB index of C1")
parser.add_argument("checkpoint_folder", type=str, help="Folder with checkpoints from SMD simulation.")
parser.add_argument("output_folder", type=str, help="Output folder base path for simulation.")
parser.add_argument("--protein", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Is protein or ligand simulation.")
parser.add_argument("--ligand", type=str, help="Name of the ligand with sdf of the same name (required for topology).")
parser.add_argument("--steps", type=int, default=10000, help="Number of steps for production run.")
parser.add_argument("--index", type=int, default=0, help="Index of the umbrella.")
parser.add_argument("--start", type=int, default=0, help="Restart point.")
parser.add_argument("--number", type=int, default=3, help="Number of repeats to run.")
parser.add_argument("--force_constant", type=float, default=10, help="Force constant in kJ / (mol A **2).")
parser.add_argument("--r0", type=float, default=0.0, help="Target distance in Angstrom.")
parser.add_argument("--reporter", type=int, default=2000, help="Reporter and checkpoint frequency.")
parser.add_argument("--cv_reporter", type=int, default=200, help="Reporter of collective variable.")
parser.add_argument("--temperature", type=int, default=298, help="Temperature of the simulation in K.")
parser.add_argument("--cpus", type=int, default=4, help="CPUs per GPU.")
parser.add_argument("--memory", type=int, default=4096, help="Memory per CPU in MB.")
parser.add_argument("--time", type=int, default=24, help="Time in hours.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "1" 

assert args.start < args.number 
# output_folder: basepath + simulation-specific path
output_folder = os.path.join(args.output_folder, f"{args.system}-{args.seed}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# parameters
restart = True # always restart as equilibration happens during SMD
fix = False # all the fixing happens during SMD
fc_pull = args.force_constant * u.kilojoules_per_mole / u.angstrom ** 2 

# ligand if required
if args.ligand:
    sdf = f"{args.ligand}.sdf"
    molecules = [Molecule.from_file(sdf)]
else:
    molecules = None

device = "cuda"

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
                      device_ml=device,
                      device_mm=device)

simulator(pdb_input=pdb_input,
          molecules=molecules,
          minimize=False,
          restart=restart,
          fix=fix)


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
        f.write(f"{i * args.cv_reporter}, {cv}, {cv1}, {cv2}\n")
        simulator.simulation.step(args.cv_reporter)

if (args.start + 1) < args.number:
    if args.protein:
        submit_protein(args.system, 
                       args.c1_idx, 
                       args.checkpoint_folder,
                       args.output_folder,
                       8, 
                       args.cpus, 
                       args.memory, 
                       args.index, 
                       args.steps, 
                       args.force_constant, 
                       args.r0, 
                       args.start + 1, 
                       args.number, 
                       args.ligand,
                       args.seed)
    else:
        submit_ligand(args.system, 
                      args.c1_idx, 
                      args.checkpoint_folder,
                      args.output_folder,
                      4, 
                      args.cpus, 
                      args.memory, 
                      args.index, 
                      args.steps, 
                      args.force_constant, 
                      args.r0, 
                      args.start + 1, 
                      args.number, 
                      args.ligand,
                      args.seed)

print(f"Simulation {args.start} finished.")