from firedrake import *
from pyroteus_adjoint import *
from icepack import compute_surface
from icepack.models import IceStream
from icepack.solvers import FlowSolver
from icepack.constants import (
    ice_density as rho_I,
    water_density as rho_W,
    gravity as g,
    weertman_sliding_law as m,
)
from glac_adapt.adapt import mismip_bed_topography, friction_law
from glac_adapt.options import Options
from tqdm import trange
import argparse
# from math import ceil
import os

rank = COMM_WORLD.rank
print(rank)

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--input-level', type=int, default=0)
parser.add_argument('--output-level', type=int)
parser.add_argument('--input-idx', type=int)
parser.add_argument('--melt', action='store_true')
parser.add_argument('--simulation-end-time', type=float)
parser.add_argument('--simulation-timestep', type=float)
parser.add_argument('--timesteps-per-export', type=int)
parser.add_argument('--half', action='store_true')

args = parser.parse_args()
if args.output_level < args.input_level:
    raise ValueError('Output level must be >= input level!')


def get_function_spaces(mesh):
    """
    Get the finite element space for a given mesh.
    """
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_solver(mesh_seq):
    options = mesh_seq.options

    def solver(index, ic):
        """
        Solve forward over time window (`t_start`, `t_end`).
        """
        t_start, t_end = mesh_seq.time_partition.subintervals[index]
        fspace = ic.u.function_space()
        msh = fspace.mesh()

        Q = FunctionSpace(msh, "CG", fspace._ufl_element.degree())

        if args.melt:
            accumulation = Function(Q, name='accumulation')
        else:
            accumulation = options.constants.acc_rate
        
        u_ = Function(fspace, name="u_old")
        u = Function(fspace, name="u")
        mesh_seq.h = Function(Q, name="thickness")

        if args.input is not None:
            # if args.input_level < args.output_level:
            #     input_mesh = mesh_hierarchy[args.input_level]
            # else:
            #     input_mesh = mesh_hierarchy[args.output_level]
            
            with CheckpointFile(f"{output_dir}/{args.input}", 'r') as afile:
                input_mesh = afile.load_mesh("firedrake_default")
                _u = afile.load_function(input_mesh, "velocity", idx=args.input_idx)
                _h = afile.load_function(input_mesh, "thickness", idx=args.input_idx)

            if args.input_level < args.output_level:
                prolong(_u, u_)
                prolong(_h, mesh_seq.h)
            else:
                u_ = _u
                mesh_seq.h = _h
        else:
            u_.assign(ic["u"])
            mesh_seq.h = interpolate(Constant(100), Q)

        u.assign(u_)
        mesh_seq.z_b = interpolate(mismip_bed_topography(msh, 80e3), Q) # TODO: Ly
        mesh_seq.s = compute_surface(thickness=mesh_seq.h, bed=mesh_seq.z_b)

        mesh_seq.icepack_model = IceStream(friction=friction_law)
        mesh_seq.icepack_solver = FlowSolver(
            mesh_seq.icepack_model, **options.domain, **options.solvers)

        h_0 = mesh_seq.h.copy(deepcopy=True)
        num_steps = int((t_end - t_start) / mesh_seq.options.simulation.timestep)
        progress_bar = trange(num_steps)

        for i, _ in enumerate(progress_bar):

            if args.melt:
                z_d = mesh_seq.s - mesh_seq.h  # elevation of the ice base
                h_c = z_d - mesh_seq.z_b  # water column thickness
                melt = omega * dan_tanh(h_c / h_c0) * max_value(z_0 - z_d, 0)

                accumulation = options.constants.acc_rate
                accumulation.interpolate(options.constants.acc_rate - melt)

            mesh_seq.h = mesh_seq.icepack_solver.prognostic_solve(
                options.timestep,
                thickness=mesh_seq.h,
                velocity=u,
                accumulation=accumulation,
                thickness_inflow=h_0
            )
            # mesh_seq.h.interpolate(max_value(mesh_seq.h, 1.0))
            mesh_seq.s = compute_surface(thickness=mesh_seq.h, bed=mesh_seq.z_b)

            u = mesh_seq.icepack_solver.diagnostic_solve(
                velocity=u,
                thickness=mesh_seq.h,
                surface=mesh_seq.s,
                fluidity=options.constants.viscosity,
                friction=options.constants.friction
            )

            if (i+1) % options.simulation_export_idx == 0 :
                with CheckpointFile(f"{options.simulation.output}/{args.output}", 'a') as afile:
                    afile.save_function(
                        u, 
                        idx=int((i+1)/options.simulation_export_idx), 
                        name="velocity"
                        )
                    afile.save_function(
                        mesh_seq.h, 
                        idx=int((i+1)/options.simulation_export_idx), 
                        name="thickness"
                    )

            # min_h = mesh_seq.h.dat.data_ro.min()
            # max_h = mesh_seq.h.dat.data_ro.max()
            volume = assemble(mesh_seq.h * dx)
            # avg_h = assemble(mesh_seq.h * dx) / (options.domain.Lx * options.domain.Ly)
            progress_bar.set_description(f"volume: {volume / (1e9 * 917):4.2f} GT we")

            # qoi = mesh_seq.get_qoi(i)

        return {"u": u}
    return solver


def get_initial_condition(mesh_seq):
    V = mesh_seq.function_spaces["u"][0]
    x = SpatialCoordinate(mesh_seq[0])[0]
    u = interpolate(as_vector((90 * x / mesh_seq.options.domain.Lx, 0)), V)

    return {'u': u}


def get_form(mesh_seq):
    def form(index, sols):
        u, u_ = sols["u"]

        action = mesh_seq.icepack_solver._diagnostic_solver._model.action(
            velocity=u,
            thickness=mesh_seq.h,
            surface=mesh_seq.s,
            fluidity=mesh_seq.options.constants.viscosity,
            friction=mesh_seq.options.constants.friction,
            **mesh_seq.options.domain
        )

        F = derivative(action, u)

        return F
    return form


def get_bcs(mesh_seq):
    def bcs(index):
        V = mesh_seq.function_spaces["u"][index]

        if hasattr(V._ufl_element, "_sub_element"):
            bc = DirichletBC(V, Constant((0, 0)), mesh_seq.options.domain.dirichlet_ids)
        else:
            bc = DirichletBC(V, Constant(0), mesh_seq.options.domain.dirichlet_ids)
        if not mesh_seq.options.domain.dirichlet_ids:
            bc = None

        return bc
    return bcs

def dan_tanh(z):
    return (exp(z) - exp(-z)) / (exp(z) + exp(-z))

pwd = os.path.dirname(os.path.realpath(__file__))
output_dir = f"{pwd}/output"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

Ly = 40e3 if args.half else 80e3

opts = {
    "timestep": args.simulation_timestep, 
    "end_time": args.simulation_end_time, 
    "Ly": Ly,
    "output": output_dir, 
    "simulation_export_idx": args.timesteps_per_export,
    # "input": input_dir,
    # "simulation_export_time": args.simulation_export_time,
}
options = Options(**opts)

ny = int(Ly / 2e3)
nx = int(options.domain.Lx/Ly) * ny

print(f'resolution: {int(Ly / ny)}m x {int(options.domain.Lx / nx)}m')

coarse_mesh = firedrake.RectangleMesh(nx, ny, options.domain.Lx, Ly)
mesh_hierarchy = firedrake.MeshHierarchy(coarse_mesh, args.output_level)
mesh = mesh_hierarchy[args.output_level]
options.initial_mesh = mesh

# melt
omega = Constant(0.2 if args.melt else 0.0)
z_0 = Constant(-100)
h_c0 = Constant(75.0)
fields = ["u"]
num_subintervals=1
meshes = [options.initial_mesh]
time_partition = TimePartition(
    options.simulation.end_time,
    num_subintervals,
    options.simulation.timestep,
    fields,
    timesteps_per_export=args.timesteps_per_export,
)

msq = MeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_bcs=get_bcs,
    get_solver=get_solver,
)
msq.options = options

with CheckpointFile(f"{options.output}/{args.output}", 'w') as afile:
    afile.save_mesh(mesh)

msq.solve_forward()
