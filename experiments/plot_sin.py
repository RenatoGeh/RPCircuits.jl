import math
import numpy as np
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
