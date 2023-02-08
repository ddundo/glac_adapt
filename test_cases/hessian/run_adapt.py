from mismip_adapt import *
import argparse

rank = COMM_WORLD.rank

parser = argparse.ArgumentParser()
parser.add_argument('--chk-idx', type=int, default=0)
parser.add_argument('--input', type=str, default='/output/')
parser.add_argument('--output', type=str, default='/output/')

args = parser.parse_args()

chk_idx = args.chk_idx
options = Options(end_time=3000, chk_idx=chk_idx)

if chk_idx == 0:
    # initial_mesh = afile.
    with CheckpointFile("../initial_mesh.h5", 'r') as afile:
        initial_mesh = afile.load_mesh("firedrake_default")
else:
    with CheckpointFile(args.input + f"adapted_{chk_idx}.h5", 'r') as afile:
        initial_mesh = afile.load_mesh(f"adapted_mesh_{chk_idx}")

options['initial_mesh'] = initial_mesh
GlacierGOMS = Glacier(options)


