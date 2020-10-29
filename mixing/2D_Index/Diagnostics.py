from DAL import DAL
from grid import grid
from direct_Problem import direct_Problem
from adjoint_Problem import adjoint_Problem
from diag_Problem import diag_Problem
from direct_NS import direct_NS
from adjoint_NS import adjoint_NS
from diag_NS import diag_NS
from terminal_Problem import terminal_Problem
from terminal_Lp import terminal_Lp
import os
import psutil
import numpy as np
import gc
from numpy import linalg as LA
from dedalus import public as de
from param import param
import array as ar
import checkPoints
import terminal
import operator as op
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import h5py
import sys

[Re, Pe, nx, ny, T, Td, L, da, e0, Vol, mag, p] = parameters()

"""

Code for generating results from the optimal solution u0 for mixing problem. Code works as follows:


"""

uOpt = []
vOpt = []

if rank == 0:
    print("Beginning Diagnositcs ... \n")

# Convert Q to the field
arrOpt = [uOpt, vOpt]
uvOpt = np.array_split(Xopt, [nx*int(ny/p)])
for i in range(0, 2):
    arrOpt[i] = np.reshape(uvOpt[i], (nx, int(ny/p)))

diag_NS(solver_Diag, solver_Terminal, dom, arrOpt, rank)

if rank == 0:
    print("Diagnostics Complete!")
