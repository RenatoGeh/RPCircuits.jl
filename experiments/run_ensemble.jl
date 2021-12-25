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

function run()
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

run()
