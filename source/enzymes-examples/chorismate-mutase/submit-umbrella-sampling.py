import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("simulation", type=str, help="Name of the simulation")
parser.add_argument("c1", type=int, help="Position of c1 index in PDB")
parser.add_argument("seed", type=str, help="Random seed")
parser.add_argument("checkpoint_folder", type=str, help="Folder with checkpoints from SMD.")
parser.add_argument("number", type=int, help="How many iterations to take")
parser.add_argument("steps", type=int, help="How many steps to take per iteration")
parser.add_argument("--base_folder", type=str, help="Base folder that contains simulation folder", default="/cluster/work/igc/fpultar/projects/bioff/chorismate-mutase")
parser.add_argument("--protein", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Is protein or ligand simulation.")

args = parser.parse_args()

output_folder = os.path.join(args.base_folder, "simulations")
simulation_folder = os.path.join(output_folder, f"{args.simulation}-{args.seed}")

if not os.path.exists(simulation_folder):
    os.makedirs(simulation_folder)

epsilon = 1e-3
small_step = 0.1
big_step = 0.2

first_half = np.arange(-2.6, -1.6 + epsilon, big_step) # 180 
second_half = np.arange(1.6, 2.6 + epsilon, big_step) # 180
transition_state = np.arange(-1.5, 1.5  + epsilon, small_step) # 1800

first_half_fc = [180] * len(first_half)
second_half_fc = [180] * len(second_half)
transition_state_fc = [1800] * len(transition_state)

windows = np.concatenate((first_half, transition_state, second_half))
force_constants = np.concatenate((first_half_fc, transition_state_fc, second_half_fc))

cpus = 2
memory = 3100
gpu_string = "--gpus=1 --gres=gpumem:11g"
protein_string = "--protein" if args.protein else ""
hours = 8 if args.protein else 4

for idx, (force_constant, window) in enumerate(zip(force_constants, windows)):
    output = os.path.join(simulation_folder, f"{args.simulation}-{idx}-0.out")
    error = os.path.join(simulation_folder, f"{args.simulation}-{idx}-0.err")
    submit_string = f"sbatch -n 1 {gpu_string} --cpus-per-task={cpus} --time={hours}:00:00  --mem-per-cpu={memory} --output={output}  --error={error} --open-mode=truncate --wrap=\"\
python {os.path.join(args.base_folder, "umbrella-sampling.py")} {args.simulation} {args.c1} {args.checkpoint_folder} {output_folder} --cpus {cpus} --memory {memory} --index {idx} --ligand {args.base_folder}/chorismate-edited --steps {args.steps} --r0 {window:.1f}  --force_constant {force_constant:.1f}  --start 0 --number {args.number} --seed {args.seed} {protein_string}\""
    print(submit_string)
    os.system(submit_string)
