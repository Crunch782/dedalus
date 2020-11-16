import numpy as np

def parameters(T, Re):
    L = 2.*np.pi
    Pe = Re
    Sc = Pe / Re
    n = 128
    nx = n
    ny = n
    da = 1
    e0 = 0.03
    Vol = L**2
    mag = np.sqrt(2.*e0*Vol)
    Td = 10.
    p = 1
    return [Re, Pe, nx, ny, T, Td, L, da, e0, Vol, mag, p]
