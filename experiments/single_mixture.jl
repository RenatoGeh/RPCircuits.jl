using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization
using NPZ
using ThreadPools

function learn_parameters!(C::Circuit, R::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}; batchsize = 500)
  Random.seed!(101)
  # SQUAREEM for two iterations
  # learner = SQUAREM(C)
  # Running Expectation Maximization
  # while !converged(learner) && learner.steps < 3
      # score, α = update(learner, R)
      # testnll = NLL(C, V)
      # trainnll = NLL(C, R)
      # println("It: $(learner.steps) \t NLL: $trainnll \t held-out NLL: $testnll \t α: $α")
  # end
  # SEM for the rest
  learner = SEM(C)
  avgnll = 0.0
  runnll = 0.0
  indices = shuffle!(collect(1:size(R,1)))
  while !converged(learner) && learner.steps < 40
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

datasets = ["accidents"]
configs = [[:n_projs => 100, :t_proj => :sid, :max_height = 100, :single_mix => true, :c => 0.0],
           [:n_projs => 100, :t_proj => :sid, :max_height = 100, :single_mix => true, :c => 1.0],
           [:n_projs => 100, :t_proj => :sid, :max_height = 100, :single_mix => true, :c => 10.0],
           [:n_projs => 200, :t_proj => :sid, :max_height = 100, :single_mix => true, :c => 0.0],
           [:n_projs => 200, :t_proj => :sid, :max_height = 100, :single_mix => true, :c => 1.0],
           [:n_projs => 200, :t_proj => :sid, :max_height = 100, :single_mix => true, :c => 10.0],
           [:n_projs => 3, :t_proj => :sid, :single_mix => false, :c => 0.0],
           [:n_projs => 3, :t_proj => :sid, :single_mix => false, :c => 1.0],
           [:n_projs => 3, :t_proj => :sid, :single_mix => false, :c => 10.0],
           [:n_projs => 5, :t_proj => :sid, :single_mix => false, :c => 0.0],
           [:n_projs => 5, :t_proj => :sid, :single_mix => false, :c => 1.0],
           [:n_projs => 5, :t_proj => :sid, :single_mix => false, :c => 10.0],
           [:n_projs => 8, :t_proj => :sid, :single_mix => false, :c => 0.0],
           [:n_projs => 8, :t_proj => :sid, :single_mix => false, :c => 1.0],
           [:n_projs => 8, :t_proj => :sid, :single_mix => false, :c => 10.0],
           [:n_projs => 10, :t_proj => :sid, :single_mix => false, :c => 0.0],
           [:n_projs => 10, :t_proj => :sid, :single_mix => false, :c => 1.0],
           [:n_projs => 10, :t_proj => :sid, :single_mix => false, :c => 10.0],
          ]
LL = Vector{Vector{Float64}}(undef, length(datasets))

for data_idx ∈ 1:length(datasets)
  println("Dataset: ", datasets[data_idx])
  LL[data_idx] = Vector{Float64}(undef, length(configs))
  R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
  @qthreads for i ∈ 1:length(configs)
    println("Learning structure...")
    C = learn_projections(R; binarize = true, no_dist = true, configs[i]...)
    println("Learning parameters...")
    learn_parameters!(C, R, V; batchsize = 1000)
    LL[data_idx][i] = -NLL(C, T)
    println("LL: ", LL[data_idx])
    println("Saving circuit...")
    save(C, "saved/single_sid/$(datasets[data_idx])_$(i).spn")
  end
  println("Saving results...")
  serialize("results/single_sid/$(datasets[data_idx]).data", LL[data_idx])
  open("results/single_sid/$(datasets[data_idx]).txt", "w") do out write(out, string(LL[data_idx])) end
end
