import firedrake.cython.dmcommon as dmcommon
from firedrake.petsc import PETSc

def vec(f, msh):
    """
    Get the local vec.
    """
    size = f.dat.dataset.layout_vec.getSizes()
    data = f.dat._data[:size[0]]
    return PETSc.Vec().createWithArray(data, size=size, bsize=f.dat.cdim, comm=msh.comm)

def reordered_vec(f):
    return dmcommon.to_petsc_local_numbering(vec(f), f.function_space())
