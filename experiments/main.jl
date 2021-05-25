using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization

LL = Vector{Float64}(undef, 20)
for data_idx ∈ 1:20
  println("Dataset: ", RPCircuits.twenty_dataset_names[data_idx])
  R, V, T = twenty_datasets(RPCircuits.twenty_dataset_names[data_idx]; as_df = false)
  C = learn_projections(R; t_proj = :sid, binarize = true)
  learner = SEM(C)
  initialize(learner)
  batchsize = 100
  avgnll = NLL(C, R)
  runnll = 0.0
  # println("It: $(learner.steps) \t train NLL: $avgnll \t held-out NLL: $(NLL(C, V))")
  indices = shuffle!(collect(1:size(R,1)))
  while !converged(learner) && learner.steps < 50
    sid = rand(1:(length(indices)-batchsize))
    batch = view(R, indices[sid:(sid+batchsize-1)], :)
    η = max(0.95^learner.steps, 0.3)
    update(learner, batch, η)
    testnll = NLL(C, V)
    batchnll = NLL(C, batch)
    # running average NLL
    avgnll *= (learner.steps-1)/learner.steps # discards initial NLL
    avgnll += batchnll/learner.steps
    runnll = (1-η)*runnll + η*batchnll
    println("It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $runnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
  end
  LL[data_idx] = logpdf(C, T)
  println("LL: ", LL[data_idx])
end
serialize("twenty_datasets.data", LL)
