import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

R = np.load("moons.train.npy")

names = ["sid", "sid_single", "max", "max_single"]
for s in names:
  print("Plotting Gaussians...")
  M, S = np.load(f"results/moons/{s}_mean.npy"), np.load(f"results/moons/{s}_variance.npy")
  ax = plt.subplot(111, aspect = "equal")
  E = [Ellipse(xy = M[i,:], width = math.sqrt(S[i,0]), height = math.sqrt(S[i,1])) for i in range(M.shape[0])]
  for e in E:
    ax.add_artist(e)
    e.set_facecolor([0.8, 0.1, 0.1, 0.1])
  plt.scatter(R[:,0], R[:,1])
  plt.savefig(f"plots/moons_{s}.pdf")
  plt.clf()

  print("Plotting density...")
  x_bounds = list(np.arange(-1.5, 2.5, 0.025))
  x_bounds.append(2.5)
  y_bounds = list(np.arange(-0.75, 1.25, 0.01))
  y_bounds.append(1.25)
  density = np.load(f"results/moons/{s}_density.npy")
  plt.hist2d(x_bounds, y_bounds, weights = density)
  plt.savefig(f"plots/moons_density_{s}.pdf")
  plt.clf()
