using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization
using Printf

using BlossomV
using ProbabilisticCircuits: learn_vtree
using LogicCircuits: Vtree
using DataFrames

default_value(i, x, f) = length(ARGS) < i ? x : f(ARGS[i])

name = ARGS[1]
split = default_value(2, :max, x -> Symbol(x))
max_height = default_value(3, 10, x -> parse(Int, x))
em_steps = default_value(4, 100, x -> parse(Int, x))
strategy = default_value(5, :em, Symbol)
batchsize = default_value(6, 500, x -> parse(Int, x))

tee(out, str) = (write(out, str * "\n"); println(str))
verbose = true

all_data = [#="accidents", "ad", "baudio", "bbc", "bnetflix", "book",=# "c20ng"#=, "cr52", "cwebkb",
            "dna", "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb_star",
            "tmovie", "tretail"=#]

function run_bin()
  # datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]
  datasets = all_data
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_ensemble_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
    println("Learning structure...")
    if strategy == :bma
      c_time = @elapsed C = BMC(3, R, 3, 3; split, max_height, bin = true)
    else
      c_time = @elapsed C = ensemble(R, 5; em_steps, batch_size = batchsize, max_diff = 0.1,
                                     bin = true, validation = V, split, max_height, strategy)
    end
    tee(out_data, """
        Name: $(name)
        Split type: $(split)
        Max height: $(max_height)
        EM steps: $(em_steps)
        Strategy: $(strategy)
        """)
    S[data_idx] = size(C)
    tee(out_data, "Size: " * string(S[data_idx]) * " -> " * string(sum(S[data_idx])))
    LL[data_idx] = -NLL(C, T)
    tee(out_data, "LL: " * string(LL[data_idx]))
    tee(out_data, "Size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Circuit time: %.8f", c_time))
    close(out_data)
  end
  println(LL)
  serialize("results/ensemble_$(name).data", LL)
  serialize("results/ensemble_$(name)_size.data", S)
end

cont_data = ["abalone", "banknote", "ca", "kinematics", "quake", "sensorless", "chemdiab", "flowsize",
             "oldfaithful", "iris"]

function run_cont()
  datasets = cont_data
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_ensemble_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    D = continuous_datasets(datasets[data_idx]; as_df = false)
    D = D[shuffle(1:size(D, 1)), :]
    P = partition_kfold(D, 10)
    lls = zeros(10)
    c_sizes = Vector{Tuple{Int, Int, Int}}(undef, 10)
    c_times = zeros(10)
    for (i, (T, R)) ∈ enumerate(P)
      if strategy == :bma
        c_time = @elapsed C = BMC(3, R, 3, 3; split, max_height, bin = false, learngaussians = true)
      else
        c_time = @elapsed C = ensemble(R, 10; em_steps, batch_size = batchsize, max_diff = 0.1,
                                       bin = false, validation = T, split, max_height, strategy,
                                       learngaussians = true, minimumvariance = 0.01,
                                       smoothing = 0.05, learnvtree = false)
      end
      tee(out_data, """
          Name: $(name)
          Split type: $(split)
          Max height: $(max_height)
          EM steps: $(em_steps)
          Strategy: $(strategy)
          K-fold partition: $(i)
          """)
      c_sizes[i] = size(C)
      tee(out_data, "Size: " * string(c_sizes[i]) * " -> " * string(sum(c_sizes[i])))
      tee(out_data, "LL: " * string(-NLL(C, T)))
      lls[i] = -NLL(C, T)
      tee(out_data, @sprintf("Circuit time: %.8f", c_time))
      c_times[i] = c_time
    end
    LL[data_idx] = mean(lls)
    _s = [0, 0, 0]; for x ∈ c_sizes _s .= (_s[1]+x[1], _s[2]+x[2], _s[3]+x[3]) end; _s /= length(c_sizes)
    S[data_idx] = Tuple(round.(Int, _s))
    tee(out_data, "\n\nDataset: $(datasets[data_idx])\n=======\nAverage LL: " * string(LL[data_idx]))
    tee(out_data, "Average size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Average times:\n-------\nCircuit time: %.8f\n=======\n\n",
                           mean(c_times)))
    close(out_data)
  end
  println(LL)
  serialize("results/ensemble_$(name).data", LL)
  serialize("results/ensemble_$(name)_size.data", S)
end

# run_bin()
run_cont()
