from DAL import DAL
from grid import grid
from diag_Problem import diag_Problem
from diag_Solver import diag_Solver
import os
import psutil
import numpy as np
import gc
from numpy import linalg as LA
from dedalus import public as de
from parameters import parameters
import array as ar
import checkpoints
import terminal
import operator as op
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import h5py
import sys

[Re, Pe, nx, ny, T, Td, L, da, e0, Vol, mag, p] = parameters(float(sys.argv[1]), float(sys.argv[2]))

s = float(sys.argv[3])

# Define grid and MPI stuff
dom = grid(nx, ny, L, da)
reducer = GlobalArrayReducer(dom.distributor.comm)
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

solver_Diag = diag_Problem(dom, Re, Pe, Td)

"""

Code for generating results from the optimal solution u0 for mixing problem.

"""

# Load the IC
sdir = './Results/Re='+str(Re)
sTdir = sdir+'/s='+str(sys.argv[3])
sTRdir = sTdir+'/T='+str(T)
u0dir = sTRdir+'/u0'
plotdir = sTRdir+'/Plots'

Xn = []

with h5py.File(u0dir+'/u'+str(rank)+'.h5', 'r') as hf:
    Xn = hf[u0dir+'/u'+str(rank)][:]

uOpt = []
vOpt = []

if rank == 0:
    print("\nBeginning Diagnositcs for ( "+str(s)+", "+str(T)+", "+str(Re)+" ) ... \n")

# Convert Q to the field
arrOpt = [uOpt, vOpt]
uvOpt = np.array_split(Xn, [nx*int(ny/p)])
for i in range(0, 2):
    arrOpt[i] = np.reshape(uvOpt[i], (nx, int(ny/p)))

diag_Solver(solver_Diag, dom, arrOpt, s, p, nx, ny, plotdir, rank)

if rank == 0:
    print("\nDiagnostics Complete!\n")
