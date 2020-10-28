import numpy as np

def parameters():
    L = 2.*np.pi
    Pe = 50.0
    Re = 50.0
    Sc = Pe / Re
    n = 64
    nx = n
    ny = n
    da = 1
    e0 = 0.03
    Vol = L**2
    mag = np.sqrt(2.*e0)
    T = 2.
    return [Re, Pe, nx, ny, T, L, da, e0, Vol, mag]
