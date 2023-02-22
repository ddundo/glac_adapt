from glac_adapt import *
# from firedrake.petsc import PETSc
# from firedrake import max_value
from icepack.models import IceStream
from icepack.solvers import FlowSolver
from icepack import compute_surface
from tqdm import trange
from math import ceil


class Glacier(GoalOrientedMeshSeq):  # TODO: switch
    """
    Class to facilitate goal-oriented metric-based mesh
    adaptive simulations of the MISMIP benchmark problem.
    """

    # @PETSc.Log.EventDecorator()
    def __init__(self, options, num_subintervals):
        self.options = options

        fields = ["u"]
        dt_per_export = [ceil(options.simulation.end_time/(options.simulation.timestep*10))] * num_subintervals
        time_partition = TimePartition(
            options.simulation.end_time,
            num_subintervals,
            options.simulation.timestep,
            fields,
            timesteps_per_export=dt_per_export,
        )

        meshes = [options.initial_mesh] * num_subintervals

        # Create GoalOrientedMeshSeq
        super().__init__(
            time_partition,
            meshes,
            get_function_spaces=self.get_function_spaces,
            get_form=self.get_form,
            get_bcs=self.get_bcs,
            get_solver=self.get_solver,
            get_qoi=self.get_qoi,
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

            # options.simulation_end_time = t_end
            # i_export = int(np.round(t_start / options.simulation_export_time))

            u_ = Function(fspace, name="u_old")
            u_.assign(ic["u"])
            u = Function(fspace, name="u")
            u.assign(u_)

            Q = FunctionSpace(msh, "CG", fspace._ufl_element.degree())

            self.icepack_model = IceStream(friction=friction_law)
            self.icepack_solver = FlowSolver(
                self.icepack_model, **options.domain, **options.solvers)

            self.z_b = interpolate(mismip_bed_topography(msh, options.domain.Ly), Q)
            self.h = interpolate(Constant(100), Q)
            self.s = compute_surface(thickness=self.h, bed=self.z_b)

            h_0 = self.h.copy(deepcopy=True)
            num_steps = int((t_end - t_start) / options.simulation.timestep)
            progress_bar = trange(num_steps)

            for _ in progress_bar:
                self.h = self.icepack_solver.prognostic_solve(
                    options.timestep,
                    thickness=self.h,
                    velocity=u,
                    accumulation=options.constants.acc_rate,
                    thickness_inflow=h_0
                )
                self.h.interpolate(max_value(self.h, 1.0))
                self.s = compute_surface(thickness=self.h, bed=self.z_b)

                u = self.icepack_solver.diagnostic_solve(
                    velocity=u,
                    thickness=self.h,
                    surface=self.s,
                    fluidity=options.constants.viscosity,
                    friction=options.constants.friction
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


    def get_bcs(self):
        def bcs(index):
            V = self.function_spaces["u"][index]

            if hasattr(V._ufl_element, "_sub_element"):
                bc = DirichletBC(V, Constant((0, 0)), self.options.domain.dirichlet_ids)
            else:
                bc = DirichletBC(V, Constant(0), self.options.domain.dirichlet_ids)
            if not self.options.domain.dirichlet_ids:
                bc = None

            return bc
        return bcs


    def get_form(self):
        def form(index, sols):
            u, u_ = sols["u"]

            action = self.icepack_solver._diagnostic_solver._model.action(
                velocity=u,
                thickness=self.h,
                surface=self.s,
                fluidity=self.options.constants.viscosity,
                friction=self.options.constants.friction,
                **self.options.domain
            )

            F = derivative(action, u)

            return F
        return form


    def get_qoi(self, sol, index):
        def qoi():
            u = sol["u"]
            msh = self[index]

            v = FacetNormal(msh)
            j = self.h * inner(u, v) * ds(self.options.domain.ice_front_ids)

            return assemble(j)
        return qoi

    
    # def fpi(
    #     self,
    #     adaptor: Callable,
    #     enrichment_kwargs: dict = {},
    #     adj_kwargs: dict = {},
    #     indicator_fn: Callable = get_dwr_indicator,
    #     **kwargs,
    # ):
    #     update_params = kwargs.get("update_params")
    #     P = self.params
    #     self.element_counts = [self.count_elements()]
    #     self.qoi_values = []
    #     self.estimator_values = []
    #     self.converged = False
    #     msg = "Terminated due to {:s} convergence after {:d} iterations"
    #     for fp_iteration in range(P.maxiter):
    #         if update_params is not None:
    #             update_params(P, fp_iteration)

    #         # Indicate errors over all meshes
    #         sols, indicators = self.indicate_errors(
    #             enrichment_kwargs=enrichment_kwargs,
    #             adj_kwargs=adj_kwargs,
    #             indicator_fn=indicator_fn,
    #         )

    #         # Check for QoI convergence
    #         # TODO: Put this check inside the adjoint solve as
    #         #       an optional return condition so that we
    #         #       can avoid unnecessary extra solves
    #         self.qoi_values.append(self.J)
    #         self.check_qoi_convergence()
    #         if self.converged:
    #             pyrint(msg.format("QoI", fp_iteration + 1))
    #             break

    #         # Check for error estimator convergence
    #         ee = indicators2estimator(indicators, self.time_partition)
    #         self.estimator_values.append(ee)
    #         self.check_estimator_convergence()
    #         if self.converged:
    #             pyrint(msg.format("error estimator", fp_iteration + 1))
    #             break

    #         # Adapt meshes and log element counts
    #         adaptor(self, sols, indicators)
    #         self.element_counts.append(self.count_elements())

    #         # Check for element count convergence
    #         self.check_element_count_convergence()
    #         if self.converged:
    #             pyrint(msg.format("element count", fp_iteration + 1))
    #             break
    #     if not self.converged:
    #         pyrint(f"Failed to converge in {P.maxiter} iterations")


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
