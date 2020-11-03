import numpy as np
from mpi4py import MPI
import checkpoints
from dedalus import public as de
from dedalus.extras import flow_tools
from terminal import terminal
import time
import logging
logger = logging.getLogger(__name__)

def direct_Solver(solver, domain, ls, s, pn, nx, ny):

    # Set domain and variables
    x = domain.grid(0)
    y = domain.grid(1)
    p = solver.state['p']
    u = solver.state['u']
    v = solver.state['v']
    r = solver.state['r']
    ux = solver.state['ux']
    vx = solver.state['vx']
    rx = solver.state['rx']
    uy = solver.state['uy']
    vy = solver.state['vy']
    ry = solver.state['ry']

    # Set up IC u0,s0
    u['g'] = ls[0]
    v['g'] = ls[1]
    r['g'] = np.tanh(6.*(x - np.pi/2.)) - np.tanh(6.*(x - (3./2.)*np.pi) - 1.)

    # Set derivatives and scale
    u.differentiate('x', out=ux)
    u.differentiate('y', out=uy)
    v.differentiate('x', out=vx)
    v.differentiate('y', out=vy)
    r.differentiate('x', out=rx)
    r.differentiate('y', out=ry)
    u.set_scales(1)
    v.set_scales(1)
    r.set_scales(1)

    # Set list
    u_list = [np.copy(u['g'])]
    v_list = [np.copy(v['g'])]
    r_list = [np.copy(r['g'])]
    t_list = [solver.sim_time]

    # mMain Loop
    dt = 2e-3
    try:
        logger.info('\nStarting Direct loop ... \n')
        start_run_time = time.time()
        while solver.ok:
            solver.step(dt)
            u.set_scales(1)
            v.set_scales(1)
            r.set_scales(1)
            u_list.append(np.copy(u['g']))
            v_list.append(np.copy(v['g']))
            r_list.append(np.copy(r['g']))
            t_list.append(solver.sim_time)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    # Reverse for backward integration
    u_list.reverse()
    v_list.reverse()
    r_list.reverse()
    checkpoints.ulist = u_list
    checkpoints.vlist = v_list
    checkpoints.rlist = r_list

    # Compute The Terminal Condition
    r.set_scales(1)
    [J, Rt] = terminal(x, y, nx, ny, r['c'], s, pn, domain)

    return [J, Rt]
