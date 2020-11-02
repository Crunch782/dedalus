import numpy as np
import time
import math
import matplotlib.pyplot as plt
from dedalus import public as de

def grid(nx, ny, L, da):

  # Create bases and domain
  x_basis = de.Fourier('x', nx, interval=(0, L), dealias=da)
  y_basis = de.Fourier('y', ny, interval=(0, L), dealias=da)

  domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

  return domain
