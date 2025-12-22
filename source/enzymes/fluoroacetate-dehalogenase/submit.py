import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Name of the system")
parser.add_argument("seed", type=str, help="Random seed")
args = parser.parse_args()

epsilon = 1e-3
small_step = 0.1
big_step = 0.2
first_half = np.arange(-3.0, -1.6 + epsilon, big_step) # 150 
second_half = np.arange(1.6, 3.0 + epsilon, big_step) # 150
transition_state = np.arange(-1.5, 1.5  + epsilon, small_step) # 1500
windows = np.concatenate((first_half, transition_state, second_half))
windows = windows[::-1] # adjust for correct reaction coordinate sign

fc_first_half = np.array([150] * len(first_half))
fc_transition_state = np.array([1500] * len(transition_state))
fc_second_half = np.array([150] * len(second_half))
force_constants = np.concatenate((fc_first_half, fc_transition_state, fc_second_half))

steps = 100_000
number = 50
seed = args.seed
system = args.system
gpu_string = "--gpus=rtx_4090:1"

base_folder = os.path.abspath(os.path.dirname(__file__))
simulation_dir = os.path.join(base_folder, "simulations", f"{system}-{seed}")
script = os.path.join(base_folder, "umbrella-sampling.py")
simulation_base = os.path.join(base_folder, "simulations")

for idx, (r, fc) in enumerate(zip(windows, force_constants)):
    output = f"{simulation_dir}/{system}-{seed}-{idx}-0.out"
    error  = f"{simulation_dir}/{system}-{seed}-{idx}-0.err"
    submit_string = f"sbatch -n 1 --cpus-per-task=4 {gpu_string} --time=24:00:00  --mem-per-cpu=4096 --output={output}  --error={error} --open-mode=truncate --wrap=\"python {script}  {system} {system}-smd {simulation_base} --r0 {r:.1f} --force_constant {fc} --index {idx} --steps {steps} --number {number} --seed {seed} --start 0 \""
    print(submit_string)
    os.system(submit_string)