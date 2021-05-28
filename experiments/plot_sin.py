import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

R = np.load("sin_noise.npy")

names = ["sid", "sid_single", "max", "max_single"]
for s in names:
  M, S = np.load(f"results/sin/{s}_mean.npy"), np.load(f"results/sin/{s}_variance.npy")
  ax = plt.subplot(111, aspect = "equal")
  E = [Ellipse(xy = M[i,:], width = math.sqrt(S[i,0]), height = math.sqrt(S[i,1])) for i in range(M.shape[0])]
  for e in E:
    ax.add_artist(e)
    e.set_facecolor([0.8, 0.1, 0.1, 0.1])
  plt.scatter(R[:,0], R[:,1])
  plt.savefig(f"plots/sin_{s}.pdf")
  plt.clf()

  print("Plotting density...")
  # x_bounds = list(np.arange(-7.5, 7.5, 0.1))
  # x_bounds.append(7.5)
  # y_bounds = list(np.arange(-3.2, 3.2, 0.1))
  # y_bounds.append(3.2)
  # B = np.array([[x, y] for y in y_bounds for x in x_bounds])
  x_bounds = (-7.5, 7.5, int(15/0.1)+1)
  y_bounds = (-3.2, 3.2, int(6.4/0.1)+1)
  density = np.exp(np.load(f"results/sin/{s}_density.npy").reshape(y_bounds[2], x_bounds[2]))
  plt.imshow(density, cmap = 'plasma', origin = 'lower',
             extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             norm = matplotlib.colors.Normalize())
  plt.savefig(f"plots/sin_density_{s}.pdf")
  plt.clf()
