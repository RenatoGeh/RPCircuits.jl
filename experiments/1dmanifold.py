import sys
import random
import math
import time
import numpy as np

sys.path.insert(0, "/home/renatogeh/python/pyspn")

from spn.learn import gens
from spn.structs import Variable
from spn.utils.evidence import Evidence
import spn.io.file

def get_data(filename):
  D = np.load(filename).astype(float)
  R, T = D[0:-100], D[-100:]
  n = R.shape[1]
  Sc = [Variable(i, -1) for i in range(n)]
  return R, T, Sc

def learn(R, Sc):
  S = gens(Sc, R, 2, 0.1, True)
  return S

def evaluate(S, T):
  E = [Evidence.from_data(x, Sc) for x in T]
  return np.mean(np.array([S.log_value(e) for e in E]))

R, T, Sc = get_data("synt10d.npy")
S = learn(R, Sc)
ll = evaluate(S, T)
print(ll)
print(S.children)
spn.io.file.to_file(S, "saved/synt3d/learnspn_10.spn")
