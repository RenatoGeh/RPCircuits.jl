using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization

datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "mushrooms",
            "adult", "dna"]
LL = Vector{Float64}(undef, length(datasets))
for data_idx ∈ 1:length(datasets)
  println("Dataset: ", datasets[data_idx])
  R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
  println("Learning structure...")
  C = learn_projections(R; n_projs = 3, t_proj = :max, binarize = true, no_dist = true, single_mix = false)
  println("Learning parameters...")
  learner = SEM(C)
  batchsize = 500
  # avgnll = 0.0
  # runnll = 0.0
  # println("It: $(learner.steps) \t train NLL: $avgnll \t held-out NLL: $(NLL(C, V))")
  indices = shuffle!(collect(1:size(R,1)))
  while !converged(learner) && learner.steps < 30
    sid = rand(1:(length(indices)-batchsize))
    batch = view(R, indices[sid:(sid+batchsize-1)], :)
    η = 0.975^learner.steps #max(0.95^learner.steps, 0.3)
    update(learner, batch, η)
    println("Iteration: ", learner.steps)
    # testnll = NLL(C, V)
    # batchnll = NLL(C, batch)
    # running average NLL
    # avgnll *= (learner.steps-1)/learner.steps # discards initial NLL
    # avgnll += batchnll/learner.steps
    # runnll = (1-η)*runnll + η*batchnll
    # println("It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $runnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
  end
  LL[data_idx] = -NLL(C, T)
  println("LL: ", LL[data_idx])
  println("Saving results...")
  serialize("$(datasets[data_idx])_multi_max.data", LL[data_idx])
  println("Saving circuit...")
  save(C, "saved/sid/$(datasets[data_idx])_multi_max.spn")
end
