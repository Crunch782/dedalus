import numpy as np
from mpi4py import MPI
import checkpoints
from dedalus import public as de
from dedalus.extras import flow_tools
from terminal import terminal
import pandas as pd
import time
import logging
logger = logging.getLogger(__name__)


def diag_Solver(solver, domain, ls, s, pn, nx, ny, dir, rank):

    # Set domain and variables
    x = domain.grid(0)
    y = domain.grid(1)
    p = solver.state['p']
    u = solver.state['u']
    v = solver.state['v']
    r = solver.state['r']
    rd = solver.state['rd']
    ux = solver.state['ux']
    vx = solver.state['vx']
    rx = solver.state['rx']
    rdx = solver.state['rdx']
    uy = solver.state['uy']
    vy = solver.state['vy']
    ry = solver.state['ry']
    rdy = solver.state['rdy']

    # Set up IC u0,s0
    u['g'] = ls[0]
    v['g'] = ls[1]
    r['g'] = np.tanh(6.*(x - np.pi/2.)) - np.tanh(6.*(x - (3./2.)*np.pi) - 1.)
    rd['g'] = np.tanh(6.*(x - np.pi/2.)) - np.tanh(6.*(x - (3./2.)*np.pi) - 1.)

    # Set derivatives and scale
    u.differentiate('x', out=ux)
    u.differentiate('y', out=uy)
    v.differentiate('x', out=vx)
    v.differentiate('y', out=vy)
    r.differentiate('x', out=rx)
    r.differentiate('y', out=ry)
    rd.differentiate('x', out=rdx)
    rd.differentiate('y', out=rdy)
    u.set_scales(1)
    v.set_scales(1)
    r.set_scales(1)
    rd.set_scales(1)

    # Main Loop
    dt = 2e-3
    m_list = [1.]
    t_list = [0.]
    try:
        logger.info('\nStarting Direct loop ... \n')
        start_run_time = time.time()
        while solver.ok:
            solver.step(dt)
            r.set_scales(1)
            rd.set_scales(1)
            if solver.iteration % 500 == 0:
                [Ma, dJa] = terminal(x, y, nx, ny, r['c'], s, pn, domain)
                [Md, dJd] = terminal(x, y, nx, ny, rd['c'], s, pn, domain)
                if rank == 0:
                    print("t = ", solver.sim_time)
                M = np.sqrt(Ma/Md)
                m_list.append(M)
                t_list.append(solver.sim_time)

    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    if rank == 0:
        mf = pd.DataFrame({"t": t_list, "M(t)": m_list})
    if rank == 0:
        mf.to_csv(dir+"/M(t).csv", index=False)
