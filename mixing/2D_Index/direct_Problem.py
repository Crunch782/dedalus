import numpy as np
from dedalus import public as de


def direct_Problem(domain, Re, Pe, T):

  # Create Problem
  problem = de.IVP(domain, variables=['p','u','v','ux','vx','uy','vy','s','sx','sy'])

  # Parameters
  problem.parameters['nu'] = 1./Re
  problem.parameters['vu'] = 1./Pe

  # Equations
  problem.add_equation("dt(u) + dx(p) - nu*(dx(ux) + dy(uy)) = -u*ux -v*uy")
  problem.add_equation("dt(v) + dy(p) - nu*(dx(vx) + dy(vy)) = -u*vx -v*vy")
  problem.add_equation("dt(s)         - vu*(dx(sx) + dy(sy)) = -u*sx -v*sy")

  # Gauge Condition
  problem.add_equation("dx(u) + dy(v) = 0", condition="(nx != 0) or (ny != 0)")
  problem.add_equation("     integ(p) = 0", condition="(nx == 0) and (ny == 0)")

  # First Order
  problem.add_equation("dx(u) - ux = 0")
  problem.add_equation("dx(v) - vx = 0")
  problem.add_equation("dx(s) - sx = 0")
  problem.add_equation("dy(u) - uy = 0")
  problem.add_equation("dy(v) - vy = 0")
  problem.add_equation("dy(s) - sy = 0")

  # Create Solver
  solver = problem.build_solver(de.timesteppers.RK443)
  solver.stop_sim_time = T
  solver.stop_iterations = np.inf

  return solver
