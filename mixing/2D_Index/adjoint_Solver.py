import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

def adjoint_Solver(solver, domain, RT):

    x = domain.grid(0)
    y = domain.grid(1)
    q = solver.state['q']
    f = solver.state['f']
    g = solver.state['g']
    c = solver.state['c']
    fx = solver.state['fx']
    gx = solver.state['gx']
    cx = solver.state['cx']
    fy = solver.state['fy']
    gy = solver.state['gy']
    cy = solver.state['cy']

    # Set up the u+ = 0 terminal condition
    f['g'] = x # convenient...
    g['g'] = y
    c['g'] = RT
    f['g'] = 0 * f['g']
    g['g'] = 0 * g['g']

    f.differentiate('x', out=fx)
    f.differentiate('y', out=fy)
    f.differentiate('z', out=fz)
    g.differentiate('x', out=gx)
    g.differentiate('y', out=gy)
    g.differentiate('z', out=gz)
    h.differentiate('x', out=hx)
    h.differentiate('y', out=hy)
    h.differentiate('z', out=hz)
    c.differentiate('x', out=cx)
    c.differentiate('y', out=cy)
    c.differentiate('z', out=cz)


    # Main loop
    dt = 2e-3
    try:
        logger.info('\nStarting Adjoint loop\n')
        while solver.ok:
            solver.step(dt)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    f.set_scales(1)
    g.set_scales(1)
    return [f['g'], g['g']]
