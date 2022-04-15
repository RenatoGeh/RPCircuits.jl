using Pkg; Pkg.activate("..")
using RPCircuits
using NPZ
#susing ThreadPools
using Random

Random.seed!(3)

function learn_parameters!(C::Circuit, R, V; batchsize = 30)
  learner = SEM(C)
  avgnll = 0.0
  runnll = 0.0
  indices = shuffle!(collect(1:size(R,1)))
  while !converged(learner) && learner.steps < 20
    sid = rand(1:(length(indices)-batchsize))
    batch = R #view(R, indices[sid:(sid+batchsize-1)], :)
    η = 1.0 #^learner.steps #max(0.95^learner.steps, 0.3)
    update(learner, batch; learningrate=η)
    testnll = NLL(C, V)
    batchnll = NLL(C, batch)
    # running average NLL
    avgnll *= (learner.steps-1)/learner.steps # discards initial NLL
    avgnll += batchnll/learner.steps
    runnll = (1-η)*runnll + η*batchnll
    println("It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $runnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
  end
  return learner
end

D = npzread("sin_rot.npy")
indices = shuffle!(collect(1:size(D,1)))
R, V, T = D[indices[1:(end-151)],:], D[indices[end-150:end-101],:], D[indices[end-100:end],:]

println("Learning...")
L_f = [
       () -> learn_projections(R; n_projs = 3, min_examples = 5, max_height = 15, t_proj = :sid, binarize = false, no_dist = true, c = 100.0),
       () -> learn_projections(R; min_examples = 5, max_height = 100, n_projs = 25, t_proj = :sid, binarize = false, no_dist = true, single_mix = true, c = 100.0),
       () -> learn_projections(R; n_projs = 3, min_examples = 5, max_height = 15, t_proj = :max, binarize = false, no_dist = true, r = 1.0, c = 100.0),
       () -> learn_projections(R; min_examples = 5, max_height = 100, n_projs = 25, t_proj = :max, binarize = false, no_dist = true, r = 1.0, single_mix = true, c = 100.0),
      ]

load_from_file = false

names = ["sid", "sid_single", "max", "max_single"]
if load_from_file C_all = Circuit.("saved/sin_rot/" .* names .* ".pc")
else
  C_all = Vector{Circuit}(undef, length(names))
  #@qthreads 
  for i ∈ 1:length(C_all)
    println("Learning structure...")
    C = L_f[i]()
    println("Learning parameters...")
    learn_parameters!(C, R, V)
    ll = -NLL(C, T)
    println("Average log-likelihood: ", ll)
    open("results/sin_rot/$(names[i]).txt", "w") do out write(out, string(ll)) end
    C_all[i] = C
    println("Saving to file...")
    save(C, "saved/sin_rot/" * names[i] * ".pc")
  end
end
x_bounds, y_bounds = -1.5:0.05:1.5, -2.0:0.05:2.0
bounds = vec([[i, j] for i in x_bounds, j in y_bounds])
for (C, name) ∈ zip(C_all, names)
  L = leaves(C)
  n = length(L)÷2
  S = Matrix{Float64}(undef, n, 2)
  M = Matrix{Float64}(undef, n, 2)
  println("Exporting Gaussians...")
  for i ∈ 1:n
    j = 2*(i-1)+1
    S[i,1], S[i,2] = L[j].variance, L[j+1].variance
    M[i,1], M[i,2] = L[j].mean, L[j+1].mean
  end
  npzwrite("results/sin_rot/$(name)_mean.npy", M)
  npzwrite("results/sin_rot/$(name)_variance.npy", S)
  println("Exporting densities...")
  density = Vector{Float64}(undef, length(bounds))
  println("Computing densities...")
  @Threads.threads for i ∈ 1:length(bounds)
    density[i] = logpdf(C, bounds[i])
  end
  npzwrite("results/sin_rot/$(name)_density.npy", density)
  println(name, "... OK.")
end
