using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization

default_value(i, x, f) = length(ARGS) < i ? x : f(ARGS[i])

name = ARGS[1]
t_proj = Symbol(ARGS[2])
n_projs = parse(Int, ARGS[3])
is_mix = parse(Bool, ARGS[4])
max_height = parse(Int, ARGS[5])
trials = parse(Int, ARGS[6])
batchsize = default_value(7, 500, x -> parse(Int, x))
em_steps = default_value(8, 50, x -> parse(Int, x))

datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]
LL = Vector{Float64}(undef, length(datasets))
for data_idx ∈ 1:length(datasets)
  println("Dataset: ", datasets[data_idx])
  R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
  println("Learning structure...")
  C = learn_projections(R; n_projs = n_projs, t_proj = t_proj, binarize = true, single_mix = is_mix,
                        max_height = max_height, trials = trials)
  println("Learning parameters...")
  learner = SEM(C)
  indices = shuffle!(collect(1:size(R,1)))
  while !converged(learner) && learner.steps < em_steps
    sid = rand(1:(length(indices)-batchsize))
    batch = view(R, indices[sid:(sid+batchsize-1)], :)
    η = 0.975^learner.steps #max(0.95^learner.steps, 0.3)
    update(learner, batch, η)
    println("Iteration: ", learner.steps)
  end
  LL[data_idx] = -NLL(C, T)
  println("LL: ", LL[data_idx])
end
println(LL)
serialize("results/$(name).data", LL)
