import firedrake.cython.dmcommon as dmcommon
from firedrake.petsc import PETSc
from firedrake import *

def vec(f, msh):
    """
    Get the local vec.
    """
    
    size = f.dat.dataset.layout_vec.getSizes()
    data = f.dat._data[:size[0]]
    return PETSc.Vec().createWithArray(
        data, 
        size=size, 
        bsize=f.dat.cdim, 
        comm=msh.comm)

def reordered_vec(f, msh):
    return dmcommon.to_petsc_local_numbering(vec(f, msh), f.function_space())
