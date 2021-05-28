import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

R = np.load("sin_rot.npy")

names = ["sid", "sid_single", "max", "max_single"]
for s in names:
  M, S = np.load(f"results/sin_rot/{s}_mean.npy"), np.load(f"results/sin_rot/{s}_variance.npy")
  ax = plt.subplot(111, aspect = "equal")
  E = [Ellipse(xy = M[i,:], width = math.sqrt(S[i,0]), height = math.sqrt(S[i,1])) for i in range(M.shape[0])]
  for e in E:
    ax.add_artist(e)
    e.set_facecolor([0.8, 0.1, 0.1, 0.1])
  plt.scatter(R[:,0], R[:,1])
  plt.savefig(f"plots/sin_rot_{s}.pdf")
  plt.clf()

  print("Plotting density...")
  # x_bounds = list(np.arange(-1.5, 1.5, 0.05))
  # x_bounds.append(1.5)
  # y_bounds = list(np.arange(-2.0, 2.0, 0.05))
  # y_bounds.append(2.0)
  # B = np.array([[x, y] for y in y_bounds for x in x_bounds])
  x_bounds = (-1.5, 1.5, int(3/0.05)+1)
  y_bounds = (-2.0, 2.0, int(4/0.05)+1)
  density = np.exp(np.load(f"results/sin_rot/{s}_density.npy").reshape(y_bounds[2], x_bounds[2]))
  plt.imshow(density, cmap = 'plasma', origin = 'lower',
             extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             norm = matplotlib.colors.Normalize())
  plt.savefig(f"plots/sin_rot_density_{s}.pdf")
  plt.clf()
