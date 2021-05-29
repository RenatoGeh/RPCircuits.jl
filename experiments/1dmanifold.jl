using Pkg; Pkg.activate("..")
using RPCircuits
using NPZ
#using ThreadPools
using Random

function learn_parameters!(C::Circuit, L, T)
  learner = SEM(C)
  while learner.steps < 100
    η = 1.0 #0.975^learner.steps #max(0.95^learner.steps, 0.3)
    update(learner, L, η, 1e-4, true, 0.1)
    testnll = NLL(C, T)
    batchnll = NLL(C, L)
    # running average NLL
    println("It: $(learner.steps) \t train NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
  end
  return learner
end

D = npzread("synt1d.npy")
L, T = D[1:end-101,:], D[end-100:end,:]

println("Learning with $(Threads.nthreads()) threads...")
L_f = [
       () -> learn_projections(L; n_projs = 3, min_examples = 5, max_height = 30, t_proj = :sid, binarize = false, no_dist = true, c = 10.0),
       () -> learn_projections(L; min_examples = 5, max_height = 500, n_projs = 100, t_proj = :sid, binarize = false, no_dist = true, single_mix = true, c = 10.0),
       () -> learn_projections(L; n_projs = 3, min_examples = 5, max_height = 30, t_proj = :max, binarize = false, no_dist = true, r = 1.0, c = 10.0),
       () -> learn_projections(L; min_examples = 5, max_height = 500, n_projs = 100, t_proj = :max, binarize = false, no_dist = true, r = 1.0, single_mix = true, c = 10.0),
      ]

load_from_file = false

names = [
         "sid", 
         "sid_single", 
         "max", 
         "max_single"
         ]
if load_from_file C_all = Circuit.("saved/synt1d/" .* names .* ".pc")
else
  C_all = Vector{Circuit}(undef, length(names))
  for i ∈ 1:length(C_all)
    println(names[i])
    println("Learning structure...")
    C = L_f[i]()
    println("Learning parameters...")
    learn_parameters!(C, L, T)
    ll = NLL(C, T)
    open("results/synt1d/$(names[i]).txt", "w") do out write(out, string(ll)) end
    C_all[i] = C
    println("Saving to file...")
    save(C, "saved/synt1d/" * names[i] * ".pc")
  end
end
x_bounds, y_bounds = -2:0.1:2, -2:0.1:2
bounds = vec([[i, j] for i in x_bounds, j in y_bounds])
for (C, name) ∈ zip(C_all, names)
  G = leaves(C)
  n = length(G)÷2
  S = Matrix{Float64}(undef, n, 2)
  M = Matrix{Float64}(undef, n, 2)
  println("Exporting Gaussians...")
  for i ∈ 1:n
    j = 2*(i-1)+1
    S[i,1], S[i,2] = G[j].variance, G[j+1].variance
    M[i,1], M[i,2] = G[j].mean, G[j+1].mean
  end
  npzwrite("results/synt1d/$(name)_mean.npy", M)
  npzwrite("results/synt1d/$(name)_variance.npy", S)
  println("Exporting densities...")
  density = Vector{Float64}(undef, length(bounds))
  println("Computing densities...")
  Threads.@threads for i ∈ 1:length(bounds)
    density[i] = logpdf(C, bounds[i])
  end
  npzwrite("results/synt1d/$(name)_density.npy", density)
  println(name, "... OK.")
end
