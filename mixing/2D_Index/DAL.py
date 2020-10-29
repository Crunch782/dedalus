from grid import grid
from direct_Problem import direct_Problem
from adjoint_Problem import adjoint_Problem
from direct_NS import direct_NS
from adjoint_NS import adjoint_NS
import os
import psutil
import numpy as np
import gc
from numpy import linalg as LA
from dedalus import public as de
from param import param


def DAL(solver_Direct, solver_Adjoint, dom, Q, p, nx, ny):

    u0 = []
    v0 = []

    #Convert Q to the field
    arr0 = [u0, v0]
    uv = np.array_split(Q, 2)
    for i in range(0, 2):
        arr0[i] = np.reshape(uv[i], (nx, int(ny/p)))

    #arr0 contains IC for DAL
    [J, RT] = direct_NS(solver_Direct, dom, arr0)
    solver_Direct.sim_time = 0.
    solver_Direct.iteration = 0

    [udag, vdag] = adjoint_NS(solver_Adjoint, dom, RT)
    solver_Adjoint.sim_time = 0.
    solver_Adjoint.iteration = 0

    a = udag.ravel()
    b = vdag.ravel()
    X = np.concatenate((a, b), axis=None)

    gc.collect()

    return [J, X] #return the functional and the direction vector (1D)
