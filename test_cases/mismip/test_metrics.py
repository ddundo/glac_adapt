from firedrake import *
from pyroteus_adjoint import *
# import matplotlib.pyplot as plt
# import ufl

fields = ["c"]


def get_function_spaces(mesh):
    return {"c": FunctionSpace(mesh, "CG", 1)}

def source(mesh):
    x, y = SpatialCoordinate(mesh)
    x0, y0, r = 2, 5, 0.05606388
    return 100.0 * exp(-((x - x0) ** 2 + (y - y0) ** 2) / r ** 2)

def get_form(mesh_seq):
    def form(index, sols):
        c, c_ = sols["c"]
        function_space = mesh_seq.function_spaces["c"][index]
        D = Constant(0.1)
        u = Constant(as_vector([1, 0]))
        h = CellSize(mesh_seq[index])
        S = source(mesh_seq[index])

        # Stabilisation parameter
        unorm = sqrt(dot(u, u))
        tau = 0.5 * h / unorm
        tau = min_value(tau, unorm * h / (6 * D))

        # Setup variational problem
        psi = TestFunction(function_space)
        psi = psi + tau * dot(u, grad(psi))
        F = (
            dot(u, grad(c)) * psi * dx
            + inner(D * grad(c), grad(psi)) * dx
            - S * psi * dx
        )
        return F

    return form

def get_bcs(mesh_seq):
    def bcs(index):
        function_space = mesh_seq.function_spaces["c"][index]
        return DirichletBC(function_space, 0, 1)

    return bcs

def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["c"][index]

        # Ensure dependence on the initial condition
        c_ = Function(function_space, name="c_old")
        c_.assign(ic["c"])
        c = Function(function_space, name="c")
        c.assign(c_)

        # Setup variational problem
        F = mesh_seq.form(index, {"c": (c, c_)})
        bc = mesh_seq.bcs(index)

        solve(F == 0, c, bcs=bc, ad_block_tag="c")
        return {"c": c}

    return solver

def get_qoi(mesh_seq, sol, index):
    def qoi():
        c = sol["c"]
        x, y = SpatialCoordinate(mesh_seq[index])
        xr, yr, rr = 20, 7.5, 0.5
        kernel = conditional((x - xr) ** 2 + (y - yr) ** 2 < rr ** 2, 1, 0)
        return kernel * c * dx

    return qoi

mesh_joe = RectangleMesh(50, 10, 50, 10)
time_partition = TimeInstant(fields)

mesh_seq = GoalOrientedMeshSeq(
    time_partition,
    mesh_joe,
    get_function_spaces=get_function_spaces,
    get_form=get_form,
    get_bcs=get_bcs,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)

solutions = mesh_seq.solve_forward()

sol_c = solutions.c.forward[-1][-1]

hess = recover_hessian(sol_c)
hmet = hessian_metric(hess)

def enforce_spd(mesh, metric, restrict_sizes=False, restrict_anisotropy=False):
    """
    Enforce that the metric is symmetric positive-definite.
    :param restrict_sizes: should minimum and maximum metric magnitudes
        be enforced?
    :param restrict_anisotropy: should maximum anisotropy be enforced?
    :return: the :class:`RiemannianMetric`.
    """
    kw = {
        "restrictSizes": restrict_sizes,
        "restrictAnisotropy": restrict_anisotropy,
    }
    plex = mesh.topology_dm
    bsize = metric.dat.cdim
    size = [metric.dat.dataset.total_size * bsize] * 2
    v = PETSc.Vec().createWithArray(metric.dat.data_with_halos, size=size, bsize=bsize, comm=PETSc.COMM_SELF)
    det = plex.metricDeterminantCreate()
    plex.metricEnforceSPD(v, v, det, **kw)
    size = np.shape(metric.dat.data_with_halos)
    metric.dat.data_with_halos[:] = np.reshape(v.array, size)
    v.destroy()
    return metric

def adapt(mesh, metric):
    import firedrake.cython.dmcommon as dmcommon
    """
    Adapt the mesh with respect to the provided metric.
    :return: a new :class:`MeshGeometry`.
    """
    metric = enforce_spd(mesh, metric, restrict_sizes=True, restrict_anisotropy=True)
    # metric = dmcommon.to_petsc_local_numbering(metric.vec, metric.function_space())
    bsize = metric.dat.cdim
    size = [metric.dat.dataset.total_size * bsize] * 2
    v = PETSc.Vec().createWithArray(metric.dat.data_with_halos, size=size, bsize=bsize, comm=PETSc.COMM_SELF)
    newplex = mesh.topology_dm.adaptMetric(v, "Face Sets", "Cell Sets")
    v.destroy()
    return Mesh(newplex, distribution_parameters={"partition": False})

adapted_mesh = adapt(mesh_joe, hmet)