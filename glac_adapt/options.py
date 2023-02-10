from firedrake import Constant

__all__ = [
    "AttrDict",
    "Options",
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
    def __init__(self, **kwargs):
        self["domain"] = AttrDict({
            "Lx": kwargs.get('Lx', 640e3),
            "Ly": kwargs.get('Ly', 80e3),
            "dirichlet_ids": kwargs.get('dirichlet_ids', tuple([1])),  # 4 in triangle
            "side_wall_ids": kwargs.get('side_wall_ids', tuple([3, 4])),  # 1, 3 in triangle
            "ice_front_ids": kwargs.get('ice_front_ids', tuple([2])),
        })

        self["constants"] = AttrDict({
            "viscosity": kwargs.get('viscosity', Constant(20)),  # sometimes called fluidity in icepack?
            "friction": kwargs.get('friction', Constant(1e-2)),
            "acc_rate": kwargs.get('acc_rate', Constant(0.3)),  # accumulation rate [m/yr]
        })

        dsp_dict = AttrDict({
            "ksp_type": kwargs.get('ksp_type', 'preonly'),
            "pc_type": kwargs.get('pc_type', 'lu'),
            "snes_linesearch_type": kwargs.get('snes_linesearch_type', 'bt'),
            "pc_factor_shift_type": kwargs.get('pc_factor_shift_type', 'inblocks'),
            "snes_max_it": kwargs.get('snes_max_it', '300'),
            # "snes_monitor": kwargs.get('snes_monitor', None),
        })

        self["solvers"] = AttrDict({
            "diagnostic_solver_type": kwargs.get('diagnostic_solver_type', 'petsc'),
            "diagnostic_solver_parameters": kwargs.get('diagnostic_solver_parameters', dsp_dict),
        })

        self["simulation"] = AttrDict({
            "initial_mesh": kwargs.get('initial_mesh'),
            "input": kwargs.get('input', 'output'),
            "output": kwargs.get('output', 'output'),
            "timestep": kwargs.get('timestep', 5),
            "end_time": kwargs.get('end_time'),
            "export_time": kwargs.get('export_time'),
            "chk_idx": kwargs.get('chk_idx')
        })

        self["adaptation"] = AttrDict({
            "n_vert": kwargs.get('n_vert'),
        })

        super().__init__(**kwargs)
