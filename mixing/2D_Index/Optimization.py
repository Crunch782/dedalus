from DAL import DAL
from grid import grid
from direct_Problem import direct_Problem
from adjoint_Problem import adjoint_Problem
#from diag_Problem import diag_Problem
from direct_Solver import direct_Solver
from adjoint_Solver import adjoint_Solver
import os
import numpy as np
from numpy import linalg as LA
from dedalus import public as de
from parameters import parameters
import array as ar
import checkpoints
import operator as op
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import h5py
import sys
from datetime import datetime

[Re, Pe, nx, ny, T, Td, L, da, e0, Vol, mag, p] = parameters(float(sys.argv[1]), float(sys.argv[2]))

"""

Code for performing Direct-Adjoint Looping (DAL) Calculations for mixing problem. Code works as follows:

1. At t=0: (u0, v0) = (scaled) noise

2. t in [0, T]: NS integrates to t=T, saving direct variables (u,v) (velocity) and s (concentration)

3. At t=T: 'Initialize' the adjoint variables (f,g) (velocity) and r (concentration)

4. t' = -t in [0, T]: Integrate adjoint variables backwards in time to obtain (f0, g0)

5. Use (f0, g0) to update (u0, v0)

6. Repeat 1. - 5. until convergence criteria is satisfied

The optimization algorithm in this script is based on code from https://www.repository.cam.ac.uk/handle/1810/278450
 which is based on the method described in Appendix A of Foures, Caulfield & Schmid 2012 (Journal of Fluid Mechanics).

The Direct-Adjoint Looping problem is based on an unpublished chapter of a PhD thesis.

"""

# Define grid and MPI stuff
dom = grid(nx, ny, L, da)
reducer = GlobalArrayReducer(dom.distributor.comm)
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank


# Methods needed for running the update algorithm
def L2Norm(X):
    sum_square = reducer.reduce_scalar(np.square(LA.norm(X)), MPI.SUM)
    R = np.sqrt(sum_square)
    return R

def proj_grad(Xold, dJn, alpha, C, method):
    if(method == 'rot'):
        Ghold = np.multiply(np.abs(Xold), np.sign(Xold))
        upper = reducer.reduce_scalar(np.inner(Ghold, dJn), MPI.SUM)
        lower = reducer.reduce_scalar(np.inner(Ghold, Ghold), MPI.SUM)
        Ghold = Ghold * (upper/lower)
        dJn = dJn - Ghold
        return dJn
    if(method == 'lag'):
        dJn = dJn / L2Norm(dJn)
        lambdam = 1j
        while np.isreal(lambdam) == 0:
            cosA = reducer.reduce_scalar(np.inner(Xold, dJn), MPI.SUM)
            cosA = cosA / (L2Norm(dJn)*L2Norm(Xold))
            lambdam = (1./alpha) + (L2Norm(dJn)/C)*cosA - (np.square(1. /
                                                                     alpha) - np.square(L2Norm(dJn)/C)*(1. - cosA*cosA))
            alpha = alpha / 10.
        dJn = dJn - lambdam*Xold
        return dJn

def update_pos(Xold, L, e, C, method):
    L = L / L2Norm(L)
    L = proj_grad(Xold, L, e, C, method)
    if (method == 'rot'):
        Xn = np.cos(e)*Xold
        Xn = Xn + np.sin(e)*C*L
        Xn = Xn / L2Norm(Xn)
        Xn = Xn * C
        return Xn
    if (method == 'lag'):
        Xn = Xold + e*L
        Xn = Xn / L2Norm(Xn)
        Xn = Xn * C
        return Xn

# Functions for convergence

def write_history(J, dJ, dJp, e):
    nJ = J
    ndJ = L2Norm(dJ)
    ndJp = L2Norm(dJp)
    alpha = e
    nr = ndJp / ndJ
    return [nJ, ndJ, ndJp, alpha, nr]


def display(JVEC, n):
    residualNorm = np.square(JVEC[n][4])
    residual = np.square(JVEC[n][2])

    if rank == 0:
        print("After ", n, " loops: Residual = ", residual,
              "| Normalized Residual = ", residualNorm, " ... \n")
    return [residual, residualNorm]


# Initialize the checkpoint storage for u,v,s

checkpoints.checkpoints()


# Define problems/solvers for Direct/Adjoint variables

solver_Direct = direct_Problem(dom, Re, Pe, T)
solver_Adjoint = adjoint_Problem(dom, Re, Pe, T)


# Random perturbations, initialized globally for same results in parallel
gshape = dom.dist.grid_layout.global_shape(scales=1)
slices = dom.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

# Define IC (guess) for first run
a = noise.ravel()
b = noise.ravel()

# Scale for Energy Constraint
R1 = reducer.reduce_scalar(np.square(LA.norm(a)), MPI.SUM)
R2 = reducer.reduce_scalar(np.square(LA.norm(b)), MPI.SUM)
R = np.sqrt(R1 + R2)
dA = (L/nx)*(L/ny)
E = mag / (R * np.sqrt(dA))
a = a * E
b = b * E

"""

############## ALGORITHM PARAMETERS ##############

"""

epsilon = 2.2204e-16                # Same as eps in MATLAB
write = 0                           # Write to file 0 for no (default), 1 for yes
resMin = 1.                         # The minimum residual (from current and previous)
K = float(sys.argv[3])              # Gradient Search Line Multiplication Factor
r = float(sys.argv[4])              # Factor when step too large
eps = 0.002                          # Normalized Residual Tolerance (aim for O(0.001))
tol = epsilon**2                    # Machine precision for residual
e0init = float(sys.argv[5])         # Initial Step or Angle or Rotation
LS = int(sys.argv[6])               # Line Search or Not
LSI = int(sys.argv[7])              # Line Search Interpolation or Not
proj = int(sys.argv[8])	            # Projection or not
dir = -1.                           # Type of Opt
method = str(sys.argv[9])	        # Rotational Update ('rot') or Lagrange Multiplier ('lag')
dmethod = str(sys.argv[10])          # Direction update, conj graident ('conj') or steepest descent ('grad')
powit = int(sys.argv[11])            # Power iteration method
N = int(sys.argv[12])               # Max number of DALs performed
start = str(sys.argv[13])           # Starting from noise or continuing from a previous run
sstr = str(sys.argv[14])            # s index as string
s = float(sys.argv[14])             # s as real
if start == 'cont':
    resMin = float(sys.argv[15])    # Previous min residual

if rank == 0:
    print("\n\n=====Optimization Algorithm Parameters=====\n")
    print("\n(s, T, Re) = (",s, ", ", T, ", ", Re, ")\n")
    print("K       = ", K)
    print("r       = ", r)
    print("e0init  = ", e0init)
    print("LS      = ", LS)
    print("LSI     = ", LSI)
    print("proj    = ", proj)
    print("method  = ", method)
    print("dmethod = ", dmethod)
    print("powit   = ", powit)
    print("N       = ", N)
    print("start   = ", start)
    print("sstr    = ", sstr)
    if start == 'cont':
        print("resMin  = ", resMin)
    print("\n\n===========================================\n")


# Set up directory for results and u0

# Set up the s directory, now the results are stored in a folder marked T=... for the target time used which lies inside a folder marked s=... for the index used
sdir = './Results/Re='+str(Re)
sTdir = sdir+'/s='+sstr
sTRdir = sTdir+'/T='+str(T)
u0dir = sTRdir+'/u0'
plotdir = sTRdir+'/Plots'
if rank == 0:
    
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    
    if not os.path.exists(sTdir):
        os.makedirs(sTdir)

    
    if not os.path.exists(sTRdir):
        os.makedirs(sTRdir)

    # Set up folder to hold the IC and the plots
    
    if not os.path.exists(u0dir):
        os.makedirs(u0dir)
    
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

Xn = []

# Create the IC/Reload previous IC
if start == 'rand':
    if rank == 0:
        print("Starting from Random IC ... \n")
    Xn = np.concatenate((a, b), axis=None)
    Csq = reducer.reduce_scalar(np.square(LA.norm(Xn)), MPI.SUM)
    C = np.sqrt(Csq)
elif start == 'cont':
    if rank == 0:
        print("Continuing from previous solution ... \n")
    with h5py.File(u0dir+'/u'+str(rank)+'.h5', 'r') as hf:
        Xn = hf[u0dir+'/u'+str(rank)][:]
    Csq = reducer.reduce_scalar(np.square(LA.norm(Xn)), MPI.SUM)
    C = np.sqrt(Csq)



"""

##################################################


        Optimization algorithm begins here.

        DAL is run using [J, dJ] = DAL(solver_Direct, solver_Adjoint, dom, Xn)

"""

JDJ = []
n = 0

[J0, dJ0] = DAL(solver_Direct, solver_Adjoint, dom, Xn, p, nx, ny, s, reducer)
dJ0p = proj_grad(Xn, dJ0, e0init, C, method)
JDJ.append(write_history(J0, dJ0, dJ0p, e0init))
n = n + 1

dJold = dJ0
dJoldp = dJ0p
Jold = J0
Xold = Xn
L = dir*dJold

if powit == 0 :

    e0 = e0init
    res = 1.
    dres = 1.
    while res > eps and n < N and dres > tol:

        if rank == 0:
            print("Starting Loop ", n, " ... \n")
        e = e0

        # Current Value
        Jc = Jold
        dJc = dJold
        dJcp = dJoldp
        Xc = Xold

        # Update
        Xn = update_pos(Xc, L, e, C, method)

        # Evaluate
        [Jn, dJn] = DAL(solver_Direct, solver_Adjoint, dom, Xn, p, nx, ny, s, reducer)
        dJnp = proj_grad(Xn, dJn, e, C, method)
        JDJ.append(write_history(Jn, dJn, dJnp, e))
        n = n + 1

        nl = 0
        sa = [e]
        Js = [Jn]

        # Line Search
        while dir*(Jn - Jc) > 0.:
            if rank == 0:
                print("Starting Line Search ... \n")
            nl = nl + 1

            Xc = Xn
            Jc = Jn
            dJc = dJn

            e = K * e
            Xn = update_pos(Xc, L, e, C, method)
            [Jn, dJn] = DAL(solver_Direct, solver_Adjoint,
                            dom, Xn, p, nx, ny, s, reducer)
            dJnp = proj_grad(Xn, dJn, e, C, method)
            JDJ.append(write_history(Jn, dJn, dJnp, e))
            n = n + 1
            sa.append(sa[nl-1] + e)
            Js.append(Jn)

        if nl == 0:
            if rank == 0:
                print("First line search failed ... reducing step size from ", e0, " to ", r*e0, " ... \n")
            e0 = r * e0
            Xc = Xold
            dJc = dJold
            Jc = Jold


        if nl == 1:
            Xn = Xc

            if LS == 1:
                [Jc, dJc] = DAL(solver_Direct, solver_Adjoint,
                                dom, Xc, p, nx, ny, s, reducer)
                dJcp = proj_grad(Xc, dJc, e, C, method)
                JDJ.append(write_history(Jc, dJc, dJcp, e))
                n = n + 1

            dJcp = proj_grad(Xc, dJc, e, C, method)
            JDJ.append(write_history(Jc, dJc, dJcp, e))
            n = n + 1

        if nl > 1:
            if LSI == 1:
                if rank == 0:
                    print("Line search from ", min(sa), " to ", max(sa), " ... \n")
                si = np.linspace(sa[0], sa[nl], num=100)
                coefs = np.polyfit(sa, Js, np.size(sa) - 1)
                Jsi = 0*si
                for i in range(0, np.size(s)):
                    Jsi = Jsi + coefs[i] * np.power(si, np.size(coefs) - i)
                [ie, Jext] = min(enumerate(Jsi), key=op.itemgetter(1))
                e = si[ie] - sa[np.size(sa) - 1]

                Xn = update_pos(Xc, L, e, C, method)
                [Jn, dJn] = DAL(solver_Direct, solver_Adjoint,
                                dom, Xn, p, nx, ny, s, reducer)
                dJnp = proj_grad(Xn, dJn, e, C, method)
                JDJ.append(write_history(Jn, dJn, dJnp, e))
                n = n + 1

                if(-1.*dir*(Jn - Jc) > 0):
                    if rank == 0:
                        print("Line search interpolation unsuccessful ... \n")
                    Xn = Xc
                    [Jc, dJc] = DAL(solver_Direct, solver_Adjoint,
                                    dom, Xc, p, nx, ny, s, reducer)
                    dJcp = proj_grad(Xc, dJc, e, C, method)
                    JDJ.append(write_history(Jc, dJc, dJcp, e))
                    n = n + 1
                else:
                    if rank == 0:
                        print("Line search interpolation successful ... \n")
                    Xc = Xn
                    Jc = Jn
                    dJc = dJn
            elif LSI == 0:
                Xn = Xc
                [Jc, dJc] = DAL(solver_Direct, solver_Adjoint,
                                dom, Xc, p, nx, ny, s, reducer)
                dJcp = proj_grad(Xc, dJc, e, C, method)
                JDJ.append(write_history(Jc, dJc, dJcp, e))
                n = n + 1
                res = np.square(L2Norm(dJcp)/L2Norm(dJc))

        dJcp = proj_grad(Xc, dJc, e, C, method)

        upper = reducer.reduce_scalar(np.inner(dJcp, dJcp - dJoldp), MPI.SUM)
        lower = reducer.reduce_scalar(np.inner(dJoldp, dJoldp), MPI.SUM)

        if dmethod == 'conj':
            b = upper/lower
        elif dmethod == 'grad':
            b = 0.0

        b = max(0.0, b)

        if proj == 1:
            L = b*L
            L = L + dir*dJcp
        elif proj == 0:
            L = b*L
            L = L + dir*dJc

        e0 = min(e0init, (L2Norm(dJoldp)/L2Norm(dJcp))*e0)
        if LS == 0:
            e0 = e0 * K
            if rank == 0:
                print("New step size = ", e0, " ... \n")


        dJcp = proj_grad(Xc, dJc, e, C, method)
        RES = display(JDJ, n-1)
        dres = RES[0]
        res = RES[1]

        if res < resMin:
            resMin = res
            write = 1

        dJoldp = dJcp
        dJold = dJcp
        Jold = Jc
        Xold = Xn

        # Write to u0 file if solution has a lower residual than the running residual
        if write == 1:
            with h5py.File(u0dir+'/u'+str(rank)+'.h5', 'w') as hf:
                 hf.create_dataset(u0dir+'/u'+str(rank),  data=Xold)
            write = 0

if powit == 1:

    n = 0
    res = 1.
    dres = 1.

    while res > tol and dres > epsilon :

        Xn = L * (C / (L2Norm(L)))
        [Jn, dJn] = DAL(solver_Direct, solver_Adjoint, dom, Xn, p, nx, ny, s, reducer)
        dJnp = proj_grad(Xn, dJn, 1., C, 'rot')
        JDJ.append(write_history(Jn, dJn, dJnp, 1.))
        n = n + 1

        res_old = res
        RES = display(JDJ, n-1)
        res = RES[0]

        dres = np.sqrt((res_old - res)**2)

        L = dir*dJn
        Jold = Jn

Xopt = Xold
neval = n

if rank == 0:
    if res < eps:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Convergence Reached with r = ", res, " after ", neval, " loops ... Optimization Complete at ", current_time)
    elif n == N:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Hit Loop Limit N = ", n, " ... Program Complete at ", current_time)

