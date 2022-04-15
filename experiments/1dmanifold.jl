using Pkg; Pkg.activate("..")
using RPCircuits
using NPZ
#using ThreadPools
using Random

function learn_parameters!(C::Circuit, L, T; batchsize = 100, gauss = false, min_var = 0.1)
  learner = SEM(C)
  indices = shuffle!(collect(1:size(L,1)))
  while learner.steps < 30
    if batchsize > 0
      sid = rand(1:(length(indices)-batchsize))
      batch = view(L, indices[sid:(sid+batchsize-1)], :)
      η = 0.975^learner.steps #max(0.95^learner.steps, 0.3)
    else
      batch = L
      η = 1.0
    end
    update(learner, batch; learningrate=η, smoothing=1e-4, learngaussians=gauss, minimumvariance=min_var)
    # testnll = NLL(C, T)
    # batchnll = NLL(C, L)
    # running average NLL
    # println("It: $(learner.steps) \t train NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
    println("Iteration $(learner.steps)")
  end
  return learner
end

D = npzread("synt10d.npy")
L, T = D[1:end-100,:], D[end-99:end,:]

println("Learning with $(Threads.nthreads()) threads...")
L_f = [
       () -> learn_projections(L; n_projs = 3, min_examples = 10, max_height = 30, t_proj = :sid, binarize = false, no_dist = true, c = 10.0, t_mix = :all),
       () -> learn_projections(L; min_examples = 10, max_height = 50, n_projs = 100, t_proj = :sid, binarize = false, no_dist = true, t_mix = :single, c = 10.0),
       () -> learn_projections(L; n_projs = 3, min_examples = 10, max_height = 30, t_proj = :max, binarize = false, no_dist = true, r = 1.0, c = 10.0, t_mix = :all),
       () -> learn_projections(L; min_examples = 10, max_height = 50, n_projs = 100, t_proj = :max, binarize = false, no_dist = true, r = 1.0, t_mix = :single, c = 10.0),
      ]

load_from_file = false

names = [
         "sid", 
         "sid_single", 
         "max", 
         "max_single"
         ]
configs = [(false, 0.0, :online), (true, 0.1, :online), (true, 0.05, :online), (true, 0.01, :online),
           (false, 0.0, :batch), (true, 0.1, :batch), (true, 0.05, :batch), (true, 0.01, :batch)]
if load_from_file C_all = Circuit.("saved/synt1d/" .* names .* ".pc")
else
  C_all = Vector{Vector{Circuit}}(undef, length(configs))
  LL = Vector{Vector{Float64}}(undef, length(configs))
  for (j, (use_gauss, min_variance, em_type)) ∈ enumerate(configs)
    C_t = Vector{Circuit}(undef, length(names))
    LL_t = Vector{Float64}(undef, length(names))
    for i ∈ 1:length(C_t)
      println(names[i])
      println("Learning structure...")
      C = L_f[i]()
      println("Learning parameters...")
      learn_parameters!(C, L, T; batchsize = em_type == :online ? 100 : -1, gauss = use_gauss, min_var = min_variance)
      ll = NLL(C, T)
      LL_t[i] = ll
      open("results/synt1d/$(names[i])_rp_10.txt", "w") do out write(out, string(ll)) end
      C_t[i] = C
      # println("Saving to file...")
      # save(C, "saved/synt1d/" * names[i] * ".pc")
    end
    C_all[j] = C_t
    LL[j] = LL_t
  end
end
println(LL)
# x_bounds, y_bounds, z_bounds = -2:0.1:2, -2:0.1:2, -2:0.1:2
# bounds = vec([[i, j, l] for i in x_bounds, j in y_bounds, l in z_bounds])
# for (C, name) ∈ zip(C_all, names)
  # G = leaves(C)
  # n = length(G)÷3
  # S = Matrix{Float64}(undef, n, 3)
  # M = Matrix{Float64}(undef, n, 3)
  # println("Exporting Gaussians...")
  # for i ∈ 1:n
    # j = 3*(i-1)+1
    # S[i,1], S[i,2], S[i,3] = G[j].variance, G[j+1].variance, G[j+2].variance
    # M[i,1], M[i,2], M[i,3] = G[j].mean, G[j+1].mean, G[j+2].mean
  # end
  # npzwrite("results/synt1d/$(name)_mean.npy", M)
  # npzwrite("results/synt1d/$(name)_variance.npy", S)
  # println("Exporting densities...")
  # density = Vector{Float64}(undef, length(bounds))
  # println("Computing densities...")
  # Threads.@threads for i ∈ 1:length(bounds)
    # density[i] = logpdf(C, bounds[i])
  # end
  # npzwrite("results/synt1d/$(name)_density.npy", density)
  # println(name, "... OK.")
# end
