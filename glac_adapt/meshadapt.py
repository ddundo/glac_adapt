import firedrake.cython.dmcommon as dmcommon
from firedrake.petsc import PETSc
from firedrake import Mesh
from pyroteus.metric import metric_intersection


def adapt(mesh, *metrics, name='firedrake_default'):
    r"""
    Adapt a mesh with respect to a metric and some adaptor parameters.
    If multiple metrics are provided, then they are intersected.
    :param mesh: :class:`MeshGeometry` to be adapted.
    :param metrics: Riemannian metric :class:`Function`\s.
    :param adaptor_parameters: parameters used to drive
        the metric-based mesh adaptation
    :return: a new :class:`MeshGeometry`.
    """
    num_metrics = len(metrics)
    if len(metrics) == 1:
        metric = metrics[0]
    else:
        metric = metric_intersection(*metrics)

    size = metric.dat.dataset.layout_vec.getSizes()
    data = metric.dat._data[:size[0]]
    # get the local vec.
    v = PETSc.Vec().createWithArray(data, size=size, bsize=metric.dat.cdim, comm=mesh.comm)
    metric = dmcommon.to_petsc_local_numbering(v, metric.function_space())
    # v.destroy()
    newplex = mesh.topology_dm.adaptMetric(metric, "Face Sets", "Cell Sets")
    adapted_mesh = Mesh(
        newplex, 
        distribution_parameters={"partition": False},
        comm=mesh.comm, 
        name=name,
    )

    return adapted_mesh

# def vec(f, msh):
#     """
#     Get the local vec.
#     """
#     size = f.dat.dataset.layout_vec.getSizes()
#     data = f.dat._data[:size[0]]
#     return PETSc.Vec().createWithArray(
#         data, 
#         size=size, 
#         bsize=f.dat.cdim, 
#         comm=msh.comm)

# def reordered_vec(f, msh):
#     return dmcommon.to_petsc_local_numbering(vec(f, msh), f.function_space())


