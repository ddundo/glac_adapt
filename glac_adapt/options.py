from firedrake import exp, max_value, Constant, SpatialCoordinate, sqrt, inner
from icepack.constants import (
    ice_density as rho_I,
    water_density as rho_W,
    gravity as g,
    weertman_sliding_law as m,
)

__all__ = [
    "AttrDict",
    "DomainParameters",
    "PhysicsParameters"
]

class AttrDict(dict):
    """
    Dictionary that provides both ``self[key]``
    and ``self.key`` access to members.
    **Disclaimer**: Copied from `stackoverflow
    <http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python>`__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class Options(AttrDict):
    def __init__(self, initial_mesh, **kwargs):
        self["initial_mesh"] = initial_mesh

        self["domain"] = {
            kwargs.get('Lx', 640e3),
            kwargs.get('Ly', 80e3),
            kwargs.get('dirichlet_ids', tuple([4])),
            kwargs.get('side_wall_ids', tuple([1, 3])),
            kwargs.get('ice_front_ids', tuple([2])),
        }

        self["constants"] = {
            kwargs.get('viscosity', 20),
            kwargs.get('friction', 1e-2),
            kwargs.get('acc_rate', 0.3),  # accumulation rate [m/yr]
        }

        dsp_dict = {
            kwargs.get('ksp_type', 'preonly'),
            kwargs.get('pc_type', 'lu'),
            kwargs.get('snes_linesearch_type', 'bt'),
            kwargs.get('pc_factor_shift_type', 'inblocks'),
            kwargs.get('snes_maxiter', '300'),
            # kwargs.get('snes_', None),
        }

        self["solvers"] = {
            kwargs.get('diagnostic_solver_type', 'petsc'),
            kwargs.get('diagnostic_solver_parameters', dsp_dict),
        }

        self["simulation"] = {
            kwargs.get('timestep', 5),
            kwargs.get('end_time'),
            kwargs.get('export_time', 1000),
            kwargs.get('chk_idx')
        }

# class Options(AttrDict):
#     """
#     A class for .
#     """
#     def __init__(self, domainParams, physicsParams, solverParams, simParams):
#         self["domain"] = domainParams
#         self["constants"] = physicsParams
#         self["solver"] = solverParams
#         self["simulation"] = simParams


class DomainParameters(AttrDict):
    """
    A class for holding parameters associated with mismip domain.
    """
    def __init__(self, parameters: dict = {}):
        self["Lx"] = 640e3  # length of domain
        self["Ly"] = 80e3  # width of domain
        self["dirichlet_ids"] = tuple([4])
        self["side_wall_ids"] = tuple([1, 3])
        self["ice_front_ids"] = tuple([2])

        super().__init__(parameters)


class PhysicsParameters(AttrDict):
    """
    A class for holding parameters associated with the icepack physics models.
    """
    def __init__(self, parameters: dict = {}):
        self["A"] = Constant(20)  # viscosity
        self["C"] = Constant(1e-2)  # friction
        self["a"] = Constant(0.3)  # accumulation rate [m/yr]

        super().__init__(parameters)


class SolverParameters(AttrDict):
    """
    A class for holding parameters associated with the solvers.
    """
    def __init__(self, parameters: dict = {}):
        self["diagnostic_solver_type"] = "petsc"
        self["diagnostic_solver_parameters"] = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "snes_linesearch_type": "bt",
            "pc_factor_shift_type": "inblocks",
            "snes_maxiter": 300,
            # "snes_": None,  # TODO
        }

        super().__init__(parameters)


class SimulationParameters(AttrDict):
    def __init__(self, parameters: dict = {}):
        self["timestep"] = 5.0
        self["simulation_export_time"] = 1000
        self["num_subintervals"] = 1

        super().__init__(parameters)
