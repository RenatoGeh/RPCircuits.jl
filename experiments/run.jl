using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization

default_value(i, x, f) = length(ARGS) < i ? x : f(ARGS[i])

name = ARGS[1]
t_proj = Symbol(ARGS[2])
n_projs = parse(Int, ARGS[3])
t_mix = Symbol(ARGS[4])
max_height = parse(Int, ARGS[5])
trials = parse(Int, ARGS[6])
batchsize = default_value(7, 500, x -> parse(Int, x))
em_steps = default_value(8, 100, x -> parse(Int, x))
full_em_steps = em_steps + default_value(9, 30, x -> parse(Int, x))
learnspn_style = default_value(10, false, x -> parse(Bool, x))
dense_leaves = default_value(11, false, x -> parse(Bool, x))

function count_nodes(C::Circuit, D::AbstractMatrix{<:Real})::Tuple{Int, Int, Int}
  sums, prods, leaves = 0, 0, size(D, 2)*2
  for n ∈ C
    if issum(n) sums += 1
    elseif isprod(n) prods += 1 end
  end
  return sums, prods, leaves
end

tee(out, str) = (write(out, str * "\n"); println(str))
verbose = false

function run_reg_em()
  datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    println("Dataset: ", datasets[data_idx])
    R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
    println("Learning structure...")
    if learnspn_style
      C = learn_structured(R; binarize = true, n_projs = n_projs, t_proj = t_proj,
                           max_height = max_height, trials = trials, dense_leaves = dense_leaves)
    else
      C = learn_projections(R; n_projs = n_projs, t_proj = t_proj, binarize = true, t_mix = t_mix,
                            max_height = max_height, trials = trials, dense_leaves = dense_leaves)
    end
    println("Learning parameters...")
    learner = SEM(C)
    indices = shuffle!(collect(1:size(R,1)))
    out_data = open("logs/$(name)_dense_$(datasets[data_idx]).log", "w")
    tee(out_data, "Mini-batch EM...")
    avgnll = 0.0
    runnll = 0.0
    while learner.steps < em_steps
      sid = rand(1:(length(indices)-batchsize))
      batch = view(R, indices[sid:(sid+batchsize-1)], :)
      η = 0.975^learner.steps
      update(learner, batch, η)
      if verbose
        testnll = NLL(C, V)
        batchnll = NLL(C, batch)
        # running average NLL
        avgnll *= (learner.steps-1)/learner.steps # discards initial NLL
        avgnll += batchnll/learner.steps
        runnll = (1-η)*runnll + η*batchnll
        tee(out_data, "It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $runnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
      else
        tee(out_data, "It: $(learner.steps) \t Learning rate: $η")
      end
    end
    tee(out_data, "Full EM...")
    η = 1.0
    while learner.steps < full_em_steps
      update(learner, R, η)
      if verbose
        testnll = NLL(C, V)
        batchnll = NLL(C, R)
        # running average NLL
        avgnll *= (learner.steps-1)/learner.steps
        avgnll += batchnll/learner.steps
        tee(out_data, "It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $batchnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
      else
        tee(out_data, "It: $(learner.steps) \t Learning rate: $η")
      end
    end
    LL[data_idx] = -NLL(C, T)
    S[data_idx] = count_nodes(C, T)
    tee("LL: ", string(LL[data_idx]))
    tee("Size: ", string(S[data_idx]))
    close(out_data)
  end
  println(LL)
  serialize("results/reg_$(name).data", LL)
  serialize("results/reg_$(name)_size.data", S)
end

function run_em()
  datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    println("Dataset: ", datasets[data_idx])
    R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
    println("Learning structure...")
    C = learn_projections(R; n_projs = n_projs, t_proj = t_proj, binarize = true, t_mix = t_mix,
                          max_height = max_height, trials = trials)
    println("Learning parameters...")
    learner = SEM(C)
    if batchsize > 0 indices = shuffle!(collect(1:size(R,1))) end
    while learner.steps < em_steps
      if batchsize > 0
        sid = rand(1:(length(indices)-batchsize))
        batch = view(R, indices[sid:(sid+batchsize-1)], :)
        η = 0.975^learner.steps
      else
        batch = R
        η = 1.0
      end
      update(learner, batch, η, 1e-4, true, 0.01)
      println("Iteration: ", learner.steps)
    end
    LL[data_idx] = -NLL(C, T)
    S[data_idx] = count_nodes(C, T)
    println("LL: ", LL[data_idx])
  end
  println(LL)
  serialize("results/$(name).data", LL)
  serialize("results/$(name)_size.data", S)
end

run_reg_em()
