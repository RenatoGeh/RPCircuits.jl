import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

R = np.load("moons.train.npy")

names = ["sid", "sid_single", "max", "max_single"]
configs = ["nogauss", "gauss01", "gauss005", "gauss001", "online_nogauss", "online_gauss01",
           "online_gauss005", "online_gauss001"]
for c in configs:
  for s in names:
    print("Plotting Gaussians...")
    M, S = np.load(f"results/moons/{c}_{s}_mean.npy"), np.load(f"results/moons/{c}_{s}_variance.npy")
    ax = plt.subplot(111, aspect = "equal")
    E = [Ellipse(xy = M[i,:], width = math.sqrt(S[i,0]), height = math.sqrt(S[i,1])) for i in range(M.shape[0])]
    for e in E:
      ax.add_artist(e)
      e.set_facecolor([0.8, 0.1, 0.1, 0.1])
    plt.scatter(R[:,0], R[:,1])
    plt.savefig(f"plots/{c}_moons_{s}.pdf", bbox_inches = "tight")
    plt.clf()

    print("Plotting density...")
    # x_bounds = list(np.arange(-1.5, 2.5, 0.05))
    # x_bounds.append(2.5)
    # y_bounds = list(np.arange(-0.75, 1.25, 0.025))
    # y_bounds.append(1.25)
    # B = np.array([[x, y] for y in y_bounds for x in x_bounds])
    # density = np.load(f"results/moons/{s}_density.npy")
    # plt.hist2d(B[:,0], B[:,1], weights = np.exp(density), bins = [len(x_bounds), len(y_bounds)])
    x_bounds = (-1.5, 2.5, int(4/0.05)+1)
    y_bounds = (-0.75, 1.25, int(2/0.025)+1)
    density = np.exp(np.load(f"results/moons/{c}_{s}_density.npy").reshape(y_bounds[2], x_bounds[2]))
    plt.imshow(density, cmap = 'plasma', origin = 'lower',
               extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
               norm = matplotlib.colors.Normalize())
    plt.savefig(f"plots/{c}_moons_density_{s}.pdf", bbox_inches = "tight")
    plt.clf()
