using Pkg; Pkg.activate("..")
using Random
using RPCircuits
using NPZ
# using ThreadPools

function learn_parameters!(C::Circuit, R, V; batchsize = 500)
  learner = SEM(C)
  # avgnll = 0.0
  # runnll = 0.0
  #indices = shuffle!(collect(1:size(R,1)))
  while learner.steps < 100
    #sid = rand(1:(length(indices)-batchsize))
    #batch = view(R, indices[sid:(sid+batchsize-1)], :)
    batch = R
    η = 1.0 # 0.975^learner.steps #max(0.95^learner.steps, 0.3)
    update(learner, batch, η, 1e-4, true, 0.1)
    testnll = NLL(C, V)
    batchnll = NLL(C, batch)
    # running average NLL
    # avgnll *= (learner.steps-1)/learner.steps # discards initial NLL
    # avgnll += batchnll/learner.steps
    # runnll = (1-η)*runnll + η*batchnll
    # println("It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $runnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
    println("It: $(learner.steps) \t train NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
  end
  return learner
end

println("Loading datasets")
R, V, T = npzread.("moons." .* ["train", "valid", "test"] .* ".npy")

L_f = [
       () -> learn_projections(R; min_examples = 5, max_height = 10, t_proj = :sid, binarize = false, no_dist = true, c = 10.0),
       () -> learn_projections(R; min_examples = 5, max_height = 100, n_projs = 30, t_proj = :sid, binarize = false, no_dist = true, single_mix = true, c = 10.0),
       () -> learn_projections(R; min_examples = 5, max_height = 10, t_proj = :max, binarize = false, no_dist = true, r = 1.0, c = 10.0),
       () -> learn_projections(R; min_examples = 5, max_height = 100, t_proj = :max, binarize = false, no_dist = true, r = 1.0, n_projs = 30, single_mix = true, c = 10.0),
      ]

load_from_file = false

names = ["sid", "sid_single", "max", "max_single"]
if load_from_file C_all = Circuit.("saved/moons/" .* names .* ".spn")
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
    open("results/moons/$(names[i]).txt", "w") do out write(out, string(ll)) end
    C_all[i] = C
    println("Saving to file...")
    save(C, "saved/moons/" * names[i] * ".spn")
  end
end
x_bounds, y_bounds = -1.5:0.05:2.5, -0.75:0.025:1.25
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
  npzwrite("results/moons/$(name)_mean.npy", M)
  npzwrite("results/moons/$(name)_variance.npy", S)
  println("Exporting densities...")
  density = Vector{Float64}(undef, length(bounds))
  println("Computing densities...")
  @Threads.threads for i ∈ 1:length(bounds)
    density[i] = logpdf(C, bounds[i])
  end
  npzwrite("results/moons/$(name)_density.npy", density)
  println(name, "... OK.")
end
