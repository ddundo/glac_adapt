from pyroteus import *
from firedrake import *
import argparse

from icepack.models import IceStream
from icepack.solvers import FlowSolver

from options import *


rank = COMM_WORLD.rank

parser = argparse.ArgumentParser()
parser.add_argument('--chk-idx', type=int, default=0)
parser.add_argument('--input', type=str, default='/output/')
parser.add_argument('--output', type=str, default='/output/')

args = parser.parse_args()

chk_idx = args.chk_idx

if chk_idx == 0:
    with CheckpointFile("../initial_mesh.h5", 'r') as afile:
        input_mesh = afile.load_mesh("firedrake_default")
else:
    with CheckpointFile(input + f"adapted_{chk_idx}.h5", 'r') as afile:
        input_mesh = afile.load_mesh(f"adapted_mesh_{chk_idx}")

opts = SolverParameters()

icepack_model = IceStream(friction=friction_law)
icepack_solver = FlowSolver(icepack_model, **opts)

