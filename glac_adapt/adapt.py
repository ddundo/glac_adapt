from glac_adapt import *
# from firedrake.petsc import PETSc
# from firedrake import max_value
from icepack.models import IceStream
from icepack.solvers import FlowSolver
from tqdm import trange


class Glacier(MeshSeq):
    """
    Class to facilitate goal-oriented metric-based mesh
    adaptive simulations of the MISMIP benchmark problem.
    """

    # @PETSc.Log.EventDecorator()
    def __init__(self, options, root_dir, num_subintervals):
        self.options = options

        fields = ["u"]
        dt = options.simulation.timestep
        dt_per_export = [int(options.simulation.export_time / dt)] * num_subintervals
        time_partition = TimePartition(
            options.simulation.end_time,
            num_subintervals,
            dt,
            fields,
            timesteps_per_export=dt_per_export,
        )

        initial_meshes = [options.initial_mesh]

        # Create GoalOrientedMeshSeq
        super(Glacier, self).__init__(
            time_partition,
            initial_meshes,
            qoi_type="steady",
        )

    # @PETSc.Log.EventDecorator()
    def get_function_spaces(self, mesh):
        """
        Get the finite element space for a given mesh.
        """
        return {"u": VectorFunctionSpace(mesh, "CG", 2)}

    # @PETSc.Log.EventDecorator()
    def get_solver(self):
        options = self.options

        def solver(index, ic):
            """
            Solve forward over time window (`t_start`, `t_end`).
            """
            t_start, t_end = self.time_partition.subintervals[index]
            msh = ic.u.function_space().mesh()
            fspace = ic.u.function_space()

            options.simulation_end_time = t_end
            i_export = int(np.round(t_start / options.simulation_export_time))

            u_ = Function(fspace, name="u_old")
            u_.assign(ic["u"])
            u = Function(fspace, name="u")
            u.assign(u_)

            Q = FunctionSpace(msh, "CG", fspace._ufl_element.degree())

            self.icepack_model = IceStream(friction=friction_law)
            self.icepack_solver = FlowSolver(icepack_model, **opts)

            self.z_b = interpolate(mismip_bed_topography(msh, options.domain.Ly), Q)
            self.h = interpolate(Constant(100), Q)
            self.s = compute_surface(thickness=h, bed=z_b)

            h_0 = self.h.copy(deepcopy=True)
            num_steps = int((t_end - t_start) / self.options.simulation.dt)
            progress_bar = trange(num_steps)

            for step in progress_bar:
                self.h = icepack_solver.prognostic_solve(
                    dt,
                    thickness=h,
                    velocity=u,
                    accumulation=options.const.acc_rate,
                    thickness_inflow=h_0
                )
                self.h.interpolate(max_value(h, 1.0))
                self.s = compute_surface(thickness=self.h, bed=self.z_b)

                u = icepack_solver.diagnostic_solve(
                    velocity=u,
                    thickness=self.h,
                    surface=self.s,
                    fluidity=options.const.fluidity,
                    friction=options.const.friction
                )

                min_h = self.h.dat.data_ro.min()
                max_h = self.h.dat.data_ro.max()
                # avg_h = assemble(self.h * dx) / (options.domain.Lx * options.domain.Ly)
                progress_bar.set_description(f"avg, min h: {min_h:4.2f}, {max_h:4.2f}")

                # qoi = self.get_qoi(i)

            return {"u": u}
        return solver

    def get_initial_condition(self):
        V = self.function_spaces.u[0]
        x = SpatialCoordinate(self[0])[0]
        u = interpolate(as_vector((90 * x / self.options.domain.Lx, 0)), V)

        return {'u': u}

    def get_form(self):
        def form(index, sols):
            u, u_ = sols["u"]
            action = self.icepack_solver._diagnostic_solver._model.action(
                velocity=u,
                thickness=self.h,
                surface=self.s,
                fluidity=self.options.const.fluidity,
                friction=self.options.const.friction,
                **self.options.domain
            )
            F = derivative(action, u)

            return F
        return form


    def get_qoi(self, sol, index):
        def qoi():
            u = sol["u"]
            msh = self[index]

            # metadata = {
            #     "quadrature_degree": self.icepack_solver._diagnostic_solver._model.quadrature_degree(velocity=u, thickness=self.h),
            # }
            # _ds = ds(domain=msh, metadata=metadata)

            v = FacetNormal(msh)
            j = self.h * inner(u, v) * ds(self.options.domain.ice_front_ids)

            return j
        return qoi

    
    def adaptor(self, sols, inds):
        chk_idx = self.options.simulation.chk_idx
        sol_u = sols['u']['forward'][0][-1]

        Q = FunctionSpace(self[0], family='cg', degree=1)
        ux = Function(Q)
        ux.interpolate(sol_u[0])
        xHess = recover_hessian(sol_u[0])
        metricxHess = hessian_metric(xHess)

        metcom = metric_complexity(xHess)
        Nvert = 3000
        d = 2
        alpha = (Nvert / metcom) ** (2/d)

        metricxHess.assign(alpha*metricxHess)

        with CheckpointFile('temp_chk.h5', 'w') as afile:
            afile.save_mesh(self[0])
            afile.save_function(metricxHess, name="metric")

        if COMM_WORLD.rank == 0:
            with CheckpointFile('temp_chk.h5', 'r', comm=COMM_SELF) as afile:
                old_mesh = afile.load_mesh(f"adapted_mesh_{chk_idx}")  # TODO: is this needed? self?
        # old_mesh = self[0]
            metric = reorderec_vec(metricxHess, old_mesh)
            newplex = old_mesh.topology_dm.adaptMetric(metric, "Face Sets", "Cell Sets")
            adapted_mesh = mesh(newplex, 
                                distribution_parameters={"partition": False}, 
                                name=f"adapted_mesh_{chk_idx}")


def mismip_bed_topography(msh, Ly):
    x, y = SpatialCoordinate(msh)

    x_c = Constant(300e3)
    X = x / x_c

    B_0 = Constant(-150)
    B_2 = Constant(-728.8)
    B_4 = Constant(343.91)
    B_6 = Constant(-50.57)
    B_x = B_0 + B_2 * X**2 + B_4 * X**4 + B_6 * X**6

    f_c = Constant(4e3)
    d_c = Constant(500)
    w_c = Constant(24e3)

    B_y = d_c * (
        1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c)) +
        1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c))
    )

    z_deep = Constant(-720)
    return max_value(B_x + B_y, z_deep)


def friction_law(**kwargs):
    from icepack.constants import (
        ice_density as rho_I,
        water_density as rho_W,
        gravity as g,
        weertman_sliding_law as m,
    )

    variables = ("velocity", "thickness", "surface", "friction")
    u, h, s, C = map(kwargs.get, variables)

    p_W = rho_W * g * max_value(0, -(s - h))
    p_I = rho_I * g * h
    N = max_value(0, p_I - p_W)
    tau_c = N / 2

    u_c = (tau_c / C) ** m
    u_b = sqrt(inner(u, u))

    return tau_c * (
        (u_c**(1 / m + 1) + u_b**(1 / m + 1))**(m / (m + 1)) - u_c
    )
