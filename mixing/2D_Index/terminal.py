import numpy as np
from mpi4py import MPI
import checkpoints
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

def terminal(x, y, nx, ny, r, s, domain):

    kx = domain.elements(0)
    ky = domain.elements(1)
    J = 0.
    kxmax = int(nx/2)
    kymax = ny - 1
    RT = np.cos(x+y) # To get array in right dimensions - np.cos is irrelevant
    RT = 0 * RT

    #Compute Terminal Adjoint Concentration
    for i in range(0, kxmax):
        for j in range(0, kymax):
            if i != 0 or j != 0 or :
                a = np.real(r[i, j])
                b = np.imag(r[i, j])
                k2 = kx[i, 0]*kx[i, 0] + ky[0, j]*ky[0, j]
                RT = RT + (1./(k2)**s) * (np.cos(i*x+j*y+k*z)*a - np.sin(i*x+j*y+k*z)*b)

    # Compute Objective function J(T)
    for i in range(0, kxmax):
        for j in range(0, kymax):
            if i != 0 or j != 0 :
                a = np.real(r[i, j])
                b = np.imag(r[i, j])
                k2 = kx[i, 0]*kx[i, 0] + ky[0, j]*ky[0, j] # k*2
                    J = J + (1./(k2)**s) * (a*a + b*b)
    J = 0.5*J

    return [J, RT]
