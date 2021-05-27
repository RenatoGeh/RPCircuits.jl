using Pkg; Pkg.activate("..")
using RPCircuits
using NPZ

function learn_parameters!(C::Circuit; batchsize = 100)
  learner = SEM(C)
  avgnll = 0.0
  runnll = 0.0
  indices = shuffle!(collect(1:size(R,1)))
  while !converged(learner) && learner.steps < 100
    sid = rand(1:(length(indices)-batchsize))
    batch = view(R, indices[sid:(sid+batchsize-1)], :)
    η = 0.975^learner.steps #max(0.95^learner.steps, 0.3)
    update(learner, batch, η)
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

println("Loading datasets")
R, V, T = npzread.("moons." .* ["train", "valid", "test"] .* ".npy")

C_sid = learn_projections(R; min_examples = 5, max_height = 10, t_proj = :sid, binarize = false, no_dist = true)
# learn_parameters!(C_sid)
println("RP-SID average log-likelihood: ", -NLL(C_sid, T))
C_sid_single = learn_projections(R; min_examples = 5, max_height = 10, n_projs = 20, t_proj = :sid, binarize = false, no_dist = true, single_mix = true)
println("RP-SID average log-likelihood: ", -NLL(C_sid_single, T))

C_max = learn_projections(R; min_examples = 5, max_height = 10, t_proj = :max, binarize = false, no_dist = true, r = 1.0)
# learn_parameters!(C_max)
println("RP-Max average log-likelihood: ", -NLL(C_max, T))
C_max_single = learn_projections(R; min_examples = 5, max_height = 10, t_proj = :max, binarize = false, no_dist = true, r = 1.0, n_projs = 20, single_mix = true)
println("RP-Max average log-likelihood: ", -NLL(C_max_single, T))

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
  npzwrite("results/moons/$(name)_mean.npy", M)
  npzwrite("results/moons/$(name)_variance.npy", S)
end
