using Pkg; Pkg.activate("..")
using RPCircuits
using NPZ

D = npzread("sin_noise.npy")

println("Learning...")
C_sid = learn_projections(D; n_projs = 3, min_examples = 5, max_height = 30, t_proj = :sid, binarize = false, no_dist = true, c = 10.0)
C_sid_single = learn_projections(D; min_examples = 5, max_height = 500, n_projs = 100, t_proj = :sid, binarize = false, no_dist = true, single_mix = true, c = 10.0)

C_max = learn_projections(D; n_projs = 3, min_examples = 5, max_height = 30, t_proj = :max, binarize = false, no_dist = true, r = 1.0, c = 10.0)
C_max_single = learn_projections(D; min_examples = 5, max_height = 500, n_projs = 100, t_proj = :max, binarize = false, no_dist = true, r = 1.0, single_mix = true, c = 10.0)

println("Exporting...")
names = ["sid", "sid_single", "max", "max_single"]
for (C, name) ∈ zip([C_sid, C_sid_single, C_max, C_max_single], names)
  L = leaves(C)
  n = length(L)÷2
  S = Matrix{Float64}(undef, n, 2)
  M = Matrix{Float64}(undef, n, 2)
  for i ∈ 1:n
    j = 2*(i-1)+1
    S[i,1], S[i,2] = L[j].variance, L[j+1].variance
    M[i,1], M[i,2] = L[j].mean, L[j+1].mean
  end
  npzwrite("results/sin/$(name)_mean.npy", M)
  npzwrite("results/sin/$(name)_variance.npy", S)
end
