from glac_adapt.adapt import *
import argparse
from glac_adapt.meshadapt import adapt
# from firedrake.meshadapt import adapt

rank = COMM_WORLD.rank
print(rank)

parser = argparse.ArgumentParser()
parser.add_argument('--chk-idx', type=int, default=0)
parser.add_argument('--input', type=str, default='output')
parser.add_argument('--output', type=str, default='output')

args = parser.parse_args()
chk_idx = args.chk_idx
print('chk_idx = ', chk_idx)

def adaptor(mesh_seq, sols, inds):
    mesh_seq.options.simulation.chk_idx += 1
    chk_idx = mesh_seq.options.simulation.chk_idx
    sol_u = sols['u']['forward'][0][-1]
    err_ind = inds[-1][-1]

    # Q = FunctionSpace(mesh_seq[0], family='CG', degree=1)
    # ux = Function(Q)
    # ux.interpolate(sol_u[0])
    # uy = Function(Q)
    # uy.interpolate(sol_u[1])
    # hessianx = recover_hessian(ux)
    # hessiany = recover_hessian(uy)
    # hmetricx = hessian_metric(hessianx)
    # hmetricy = hessian_metric(hessiany)

    # int_metric = metric_intersection(hmetricx, hmetricy)
    final_metric = isotropic_metric(err_ind)

    metcom = metric_complexity(final_metric)#hmetricx)
    Nvert = 3000
    d = 2
    alpha = (Nvert / metcom) ** (2/d)

    # hmetricx.assign(alpha*hmetricx)
    final_metric.assign(alpha*final_metric)

    with CheckpointFile(f'{options.simulation.output}/adapted_{chk_idx}.h5', 'a') as afile:
        afile.save_mesh(mesh_seq[0])
        # afile.save_function(hmetricx, name="metric")
        afile.save_function(final_metric, name='metric')
        afile.save_function(sol_u, name="u_forward")
        afile.save_function(err_ind, name="error_indicator")

    if rank == 0:
        with CheckpointFile(f'{options.simulation.output}/adapted_{chk_idx}.h5', 'r', comm=COMM_SELF) as afile:
            old_mesh = afile.load_mesh(f"mesh_{chk_idx-1}")
            load_metric = afile.load_function(old_mesh, "metric")

        adapted_mesh = adapt(old_mesh, load_metric, name=f"mesh_{chk_idx}")

        print('old, new num_cells: ', old_mesh.num_cells(), adapted_mesh.num_cells())

        with CheckpointFile(f'{options.simulation.output}/adapted_{chk_idx}.h5', 'a', comm=COMM_SELF) as afile:
            afile.save_mesh(adapted_mesh)

    COMM_WORLD.barrier()

    with CheckpointFile(f'{options.simulation.output}/adapted_{chk_idx}.h5', 'r') as afile:
        adapted_mesh = afile.load_mesh(f"mesh_{chk_idx}")
        
    mesh_seq.meshes = [adapted_mesh]

opts = {
    "timestep": 10.0, 
    "end_time": 30.0, 
    "chk_idx": args.chk_idx,
    "output": args.output,
    "input": args.input,
}
options = Options(**opts)

output_dir = f"{os.getcwd()}/{options.simulation.output}"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if chk_idx == 0:
    initial_mesh = RectangleMesh(
        100, 
        16, 
        options.domain.Lx, 
        options.domain.Ly, 
        name="mesh_0",
    )
else:
    with CheckpointFile(f"{options.simulation.input}/adapted_{chk_idx}.h5", 'r') as afile:
        initial_mesh = afile.load_mesh(f"mesh_{chk_idx}")

options['initial_mesh'] = initial_mesh
glacier = Glacier(options, 1)

glacier.fixed_point_iteration(adaptor)

