using Pkg; Pkg.activate("..")
using Random
using RPCircuits
using NPZ
# using ThreadPools

function learn_parameters!(C::Circuit, L, T; batchsize = 100, gauss = false, min_var = 0.1)
  learner = SEM(C)
  indices = shuffle!(collect(1:size(L,1)))
  while learner.steps < 100
    if batchsize > 0
      sid = rand(1:(length(indices)-batchsize))
      batch = view(L, indices[sid:(sid+batchsize-1)], :)
      η = 0.975^learner.steps #max(0.95^learner.steps, 0.3)
    else
      batch = L
      η = 1.0
    end
    update(learner, batch, η, 1e-4, gauss, min_var)
    testnll = NLL(C, T)
    batchnll = NLL(C, L)
    # running average NLL
    println("It: $(learner.steps) \t train NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
  end
  return learner
end

println("Loading datasets")
R, V, T = npzread.("moons." .* ["train", "valid", "test"] .* ".npy")

L_f = [
       () -> learn_projections(R; min_examples = 5, max_height = 10, t_proj = :sid, binarize = false, no_dist = true, c = 10.0, t_mix = :all),
       () -> learn_projections(R; min_examples = 5, max_height = 100, n_projs = 30, t_proj = :sid, binarize = false, no_dist = true, t_mix = :single, c = 10.0),
       () -> learn_projections(R; min_examples = 5, max_height = 10, t_proj = :max, binarize = false, no_dist = true, r = 1.0, c = 10.0, t_mix = :all),
       () -> learn_projections(R; min_examples = 5, max_height = 100, t_proj = :max, binarize = false, no_dist = true, r = 1.0, n_projs = 30, t_mix = :single, c = 10.0),
      ]

load_from_file = false

configs = [(false, 0.0, :online, "online_nogauss"), (true, 0.1, :online, "online_gauss01"), (true, 0.05, :online, "online_gauss005"), (true, 0.01, :online, "online_gauss001"),
           (false, 0.0, :batch, "nogauss"), (true, 0.1, :batch, "gauss01"), (true, 0.05, :batch, "gauss005"), (true, 0.01, :batch, "gauss001")]
names = ["sid", "sid_single", "max", "max_single"]
if load_from_file C_all = Circuit.("saved/moons/" .* names .* ".spn")
else
  C_all = Vector{Vector{Circuit}}(undef, length(configs))
  LL = Vector{Vector{Float64}}(undef, length(configs))
  for (j, (use_gauss, min_variance, em_type, pre)) ∈ enumerate(configs)
    C_t = Vector{Circuit}(undef, length(names))
    LL_t = Vector{Float64}(undef, length(names))
    for i ∈ 1:length(C_t)
      println(names[i])
      println("Learning structure...")
      C = L_f[i]()
      println("Learning parameters...")
      learn_parameters!(C, R, V; batchsize = em_type == :online ? 100 : -1, gauss = use_gauss, min_var = min_variance)
      ll = -NLL(C, T)
      LL_t[i] = ll
      open("results/moons/$(pre)_$(names[i]).txt", "w") do out write(out, string(ll)) end
      C_t[i] = C
      # println("Saving to file...")
      # save(C, "saved/synt1d/" * names[i] * ".pc")
    end
    C_all[j] = C_t
    LL[j] = LL_t
  end
end
x_bounds, y_bounds = -1.5:0.05:2.5, -0.75:0.025:1.25
bounds = vec([[i, j] for i in x_bounds, j in y_bounds])
for (l, (use_gauss, min_variance, em_type, pre)) ∈ enumerate(configs)
  C_t = C_all[l]
  for (k, name) ∈ enumerate(names)
    C = C_t[k]
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
    npzwrite("results/moons/$(pre)_$(name)_mean.npy", M)
    npzwrite("results/moons/$(pre)_$(name)_variance.npy", S)
    println("Exporting densities...")
    density = Vector{Float64}(undef, length(bounds))
    println("Computing densities...")
    @Threads.threads for i ∈ 1:length(bounds)
      density[i] = logpdf(C, bounds[i])
    end
    npzwrite("results/moons/$(pre)_$(name)_density.npy", density)
    println(name, "... OK.")
  end
end
