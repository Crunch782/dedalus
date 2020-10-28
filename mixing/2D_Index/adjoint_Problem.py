import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
import checkpoints


def adjoint_Problem(domain, Re, Pe, T):

  # Define Dedalus Functions to use direct variables in the adjoint equations
  def direct_U(solver):
    return checkpoints.ulist[solver.iteration]

  U = de.operators.GeneralFunction(domain,'g',direct_U,args=[])

  def direct_V(solver):
    return checkpoints.vlist[solver.iteration]

  V = de.operators.GeneralFunction(domain,'g',direct_V,args=[])

  def direct_S(solver):
    return checkpoints.slist[solver.iteration]

  S = de.operators.GeneralFunction(domain,'g',direct_S,args=[])

  # Define Problem
  problem = de.IVP(domain, variables=['q','f','g','fx','gx','fy','gy','c','cx','cy'])

  # Define parameters
  problem.parameters['U'] = U
  problem.parameters['V'] = V
  problem.parameters['S'] = S
  problem.parameters['nu'] = 1./Re
  problem.parameters['vu'] = 1./Pe

  # Define equations
  problem.add_equation("dt(f) - dx(q) - nu*(dx(fx) + dy(fy)) = -c*dx(S) + U*fx + V*fy ")
  problem.add_equation("dt(g) - dy(q) - nu*(dx(gx) + dy(gy)) = -c*dy(S) + U*gx + V*gy")
  problem.add_equation("dt(c)         - vu*(dx(cx) + dy(cy)) =            U*cx + V*cy")

  # Define Gauge Condition
  problem.add_equation("dx(f) + dy(g) = 0", condition="(nx != 0) or (ny != 0)")
  problem.add_equation("     integ(q) = 0", condition="(nx == 0) and (ny == 0)")

  # Define First Order
  problem.add_equation("dx(f) - fx = 0")
  problem.add_equation("dx(g) - gx = 0")
  problem.add_equation("dx(c) - cx = 0")
  problem.add_equation("dy(f) - fy = 0")
  problem.add_equation("dy(g) - gy = 0")
  problem.add_equation("dy(c) - cy = 0")

  # Create Solver
  solver = problem.build_solver(de.timesteppers.RK443)
  solver.stop_sim_time = T
  solver.stop_iterations = np.inf

  # Correct Arguement for General Functions
  U.args = [solver]
  U.original_args = [solver]
  V.args = [solver]
  V.original_args = [solver]
  W.args = [solver]
  W.original_args = [solver]
  R.args = [solver]
  R.original_args = [solver]

  return solver
