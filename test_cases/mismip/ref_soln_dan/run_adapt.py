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
from glac_adapt.meshadapt import adapt
from glac_adapt.options import Options
from tqdm import trange
from math import ceil
import os
import numpy as np
from numpy.linalg import eigh
# from pyroteus.options import GoalOrientedParameters

chk_idx = 1

rank = COMM_WORLD.rank
print(rank)


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
        msh = ic.u.function_space().mesh()
        fspace = ic.u.function_space()

        options.simulation_end_time = t_end

        u_ = Function(fspace, name="u_old")
        u_.assign(ic["u"])
        u = Function(fspace, name="u")
        u.assign(u_)

        Q = FunctionSpace(msh, "CG", fspace._ufl_element.degree())

        mesh_seq.icepack_model = IceStream(friction=friction_law)
        mesh_seq.icepack_solver = FlowSolver(
            mesh_seq.icepack_model, **options.domain, **options.solvers)

        mesh_seq.z_b = interpolate(mismip_bed_topography(msh, options.domain.Ly), Q)
        mesh_seq.h = interpolate(Constant(100), Q)
        mesh_seq.s = compute_surface(thickness=mesh_seq.h, bed=mesh_seq.z_b)

        h_0 = mesh_seq.h.copy(deepcopy=True)
        num_steps = int((t_end - t_start) / mesh_seq.options.simulation.timestep)
        progress_bar = trange(num_steps)

        for _ in progress_bar:
            mesh_seq.h = mesh_seq.icepack_solver.prognostic_solve(
                        options.timestep,
                        thickness=mesh_seq.h,
                        velocity=u,
                        accumulation=options.constants.acc_rate,
                        thickness_inflow=h_0
            )
            mesh_seq.h.interpolate(max_value(mesh_seq.h, 1.0))
            mesh_seq.s = compute_surface(thickness=mesh_seq.h, bed=mesh_seq.z_b)

            u = mesh_seq.icepack_solver.diagnostic_solve(
                velocity=u,
                thickness=mesh_seq.h,
                surface=mesh_seq.s,
                fluidity=options.constants.viscosity,
                friction=options.constants.friction
            )

            min_h = mesh_seq.h.dat.data_ro.min()
            max_h = mesh_seq.h.dat.data_ro.max()
            # avg_h = assemble(mesh_seq.h * dx) / (options.domain.Lx * options.domain.Ly)
            progress_bar.set_description(f"avg, min h: {min_h:4.2f}, {max_h:4.2f}")

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


def get_qoi(mesh_seq, sol, index):
    def qoi():
        u = sol["u"]
        msh = mesh_seq[index]

        # metadata = {
        #     "quadrature_degree": mesh_seq.icepack_solver._diagnostic_solver._model.quadrature_degree(velocity=u, thickness=mesh_seq.h),
        # }
        # _ds = ds(domain=msh, metadata=metadata)

        v = FacetNormal(msh)
        j = mesh_seq.h * inner(u, v) * ds(mesh_seq.options.domain.ice_front_ids)

        return j
    return qoi


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


def metric_from_hessian(hessian, tol):  # B is the 2d hessian matrix 2x2
    l_min = 1.
    l_max = 50e3

    B = np.array(hessian)

    # Compute mean diagonal and set values appropriately
    B[0, 1] = 0.5 * (B[0, 1] + B[1, 0])
    B[1, 0] = B[0, 1]

    # Solve eigenvalue problem
    eigen_values, eigen_vectors = eigh(B)

    # Take modulus of eigenvalues
    # eigen_values = np.clip(np.abs(eigen_values), l_min, l_max)
    mod_eigen_values = np.minimum(np.maximum(2/9 / tol * np.abs(eigen_values), 1/l_max**2), 1/l_min**2)

    # Build metric from eigendecomposition
    Q = eigen_vectors
    D = np.diag(mod_eigen_values)
    # A += Q @ D @ Q.T

    M = Q @ D @ Q.T
    return M


# load ref soln
coarse_mesh = RectangleMesh(160, 20, 640e3, 80e3, name="coarse_mesh")
mesh_hierarchy = MeshHierarchy(coarse_mesh, 3)
mesh_ref = mesh_hierarchy[3]

Q_ref = FunctionSpace(mesh_ref, family='CG', degree=1)
V_ref = VectorFunctionSpace(mesh_ref, family='CG', degree=1)

h_ref = Function(Q_ref)
u_ref = Function(V_ref)

with DumbCheckpoint('steady-state', mode=FILE_READ) as chk:
    timesteps, indices = chk.get_timesteps()
    chk.set_timestep(timesteps[-1], idx=indices[-1])

    chk.load(h_ref, name='h')
    chk.load(u_ref, name='u')

def adapt_loop(in_mesh, tol):
    # interpolate variable
    Q_in = FunctionSpace(in_mesh, family='CG', degree=1)
    V_in = VectorFunctionSpace(in_mesh, family='CG', degree=1)
    u_in = project(u_ref, V_in)

    ux = Function(Q_in)
    ux.interpolate(u_in[0])
    ux_hess = recover_hessian(ux)

    ux_hess_data = ux_hess.dat.data
    ux_hess_met_fnA = hessian_metric(ux_hess)
    ux_hess_met_fnA_temp = np.zeros(ux_hess_data.shape)

    for i in range(ux_hess_met_fnA_temp.shape[0]):
        ux_hess_met_fnA_temp[i] = metric_from_hessian(ux_hess_data[i], tol)
    ux_hess_met_fnA.dat.data[:] = ux_hess_met_fnA_temp

    metcom = metric_complexity(ux_hess_met_fnA)
    Nvert = 700
    d = 2
    alpha = (Nvert / metcom) ** (2/d)

    ux_hess_met_fnA.assign(alpha*ux_hess_met_fnA)

    # with CheckpointFile(f'output/out_{chk_idx}.h5', 'w') as afile:
        # afile.save_mesh(in_mesh)
        # afile.save_function(ux_hess_met_fnA, name='metric')

    # if rank == 0:
        # with CheckpointFile(f'output/out_{chk_idx}.h5', 'r', comm=COMM_SELF) as afile:
            # old_mesh = afile.load_mesh(f"coarse_mesh")
            # load_metric = afile.load_function(old_mesh, "metric")

    adapted_mesh = adapt(in_mesh, ux_hess_met_fnA, name=f"adapted_mesh")

    print('old, new num_cells: ', in_mesh.num_cells(), adapted_mesh.num_cells())

        # with CheckpointFile(f'output/out_{chk_idx}.h5', 'a', comm=COMM_SELF) as afile:
            # afile.save_mesh(adapted_mesh)

    # COMM_WORLD.barrier()
    
    return adapted_mesh


opts = {
    "timestep": 5.0, 
    "end_time": 5000.0, 
    "chk_idx": chk_idx, 
# "output": args.output, "input": args.input,
# 'ditichlet_ids': tuple([4]), 'side_wall_ids': tuple([1,3]),
}
options = Options(**opts)

# if chk_idx == 0:
#     # with CheckpointFile("initial_mesh.h5", 'r') as afile:
#         # initial_mesh = afile.load_mesh("firedrake_default")
#     initial_mesh = RectangleMesh(100, 16, options.domain.Lx, options.domain.Ly, name="mesh_0")
# else:
#     with CheckpointFile(f"{options.simulation.input}/adapted_{chk_idx}.h5", 'r') as afile:
#         initial_mesh = afile.load_mesh(f"mesh_{chk_idx}")

initial_mesh_5km = RectangleMesh(128, 16, 640e3, 80e3)
meshes = [initial_mesh_5km]

for i in range(10):
    meshes.append(adapt_loop(meshes[-1], 0.1))
options.initial_mesh = meshes[-1]

fields = ["u"]
num_subintervals=1
meshes = [options.initial_mesh]
time_partition = TimePartition(
    options.simulation.end_time,
    num_subintervals,
    options.simulation.timestep,
    fields,
    timesteps_per_export=ceil(options.simulation.end_time/(options.simulation.timestep*10)),
)


# msq = GoalOrientedMeshSeq(
msq = MeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_bcs=get_bcs,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
    # **params,
)
msq.options = options

solutions = msq.solve_forward()

with CheckpointFile(f'output/out_{chk_idx}.h5', 'a') as afile:
    afile.save_mesh(msq.meshes[-1])
    afile.save_function(solutions.u.forward[-1][-1], name="velocity_adapted")
    afile.save_function(msq.h, name="thickness_adapted")
    afile.save_function(msq.s, name="surface_adapted")
