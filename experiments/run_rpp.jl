using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization
using Printf
using Statistics

using BlossomV
using ProbabilisticCircuits: learn_vtree
using LogicCircuits: Vtree
using DataFrames

default_value(i, x, f) = length(ARGS) < i ? x : f(ARGS[i])

name = ARGS[1]
split = default_value(2, :max, x -> Symbol(x))
max_height = default_value(3, 10, x -> parse(Int, x))
em_steps = default_value(4, 100, x -> parse(Int, x))
full_em_steps = em_steps + default_value(5, 30, x -> parse(Int, x))
batchsize = default_value(6, 50, x -> parse(Int, x))

tee(out, str) = (write(out, str * "\n"); println(str))
verbose = true

all_data = ["accidents", "ad", "baudio", "bbc", "bnetflix", "book", "c20ng", "cr52", "cwebkb",
            "dna", "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb_star",
            "tmovie", "tretail"]

function run_bin()
  # datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]
  datasets = all_data
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_rpp_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
    println("Learning structure...")
    v_time = @elapsed vtree = learn_vtree(DataFrame(R, :auto); alg = :bottomup)
    println("  Learning circuit...")
    c_time = @elapsed C, P = learn_rpp(R, vtree; split, max_height)
    println("  Number of projections: ", length(P))
    tee(out_data, """
        Name: $(name)
        Split type: $(split)
        Max height: $(max_height)
        EM steps: $(em_steps)
        Full EM steps: $(full_em_steps)
        """)
    S[data_idx] = size(C)
    tee(out_data, "Size: " * string(S[data_idx]) * " -> " * string(sum(S[data_idx])))
    tee(out_data, "LL: " * string(-NLL(C, T)))
    println("Learning parameters...")
    learner = SEM(C)
    indices = shuffle!(collect(1:size(R,1)))
    tee(out_data, "Mini-batch EM...")
    avgnll = 0.0
    runnll = 0.0
    batch_time = @elapsed while learner.steps < em_steps
      sid = rand(1:(length(indices)-batchsize))
      batch = view(R, indices[sid:(sid+batchsize-1)], :)
      η = 0.975^learner.steps
      update(learner, batch, η; verbose, validation = V)
    end
    tee(out_data, "Batch EM LL: " * string(-NLL(C, T)))
    tee(out_data, "Full EM...")
    η = 1.0
    em_time = @elapsed while learner.steps < full_em_steps
      update(learner, R, η; verbose, validation = V)
    end
    LL[data_idx] = -NLL(C, T)
    tee(out_data, "Full EM LL: " * string(LL[data_idx]))
    tee(out_data, "Size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Vtree time: %.8f\nCircuit time: %.8f\nBatch EM time: %.8f\nFull EM time: %.8f",
                           v_time, c_time, batch_time, em_time))
    close(out_data)
  end
  println(LL)
  serialize("results/rpp_$(name).data", LL)
  serialize("results/rpp_$(name)_size.data", S)
end

cont_data = ["abalone", "banknote", "ca", "kinematics", "quake", "sensorless", "chemdiab",
             "oldfaithful", "iris"]

function run_cont()
  datasets = cont_data
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_rpp_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    D = continuous_datasets(datasets[data_idx]; as_df = false)
    D = D[shuffle(1:size(D, 1)), :]
    P = partition_kfold(D, 10)
    lls = zeros(10)
    c_sizes = Vector{Tuple{Int, Int, Int}}(undef, 10)
    v_times, c_times, batch_times, em_times = zeros(10), zeros(10), zeros(10), zeros(10)
    for (i, (T, R)) ∈ enumerate(P)
      println("Learning structure...")
      v_time = @elapsed vtree = learn_vtree_cont(R; alg = :topdown)
      println("  Learning circuit...")
      c_time = @elapsed C, P = learn_rpp(R, vtree; split, max_height, bin = false, min_examples = 10)
      println("  Number of projections: ", length(P))
      tee(out_data, """
          Name: $(name)
          Split type: $(split)
          Max height: $(max_height)
          EM steps: $(em_steps)
          Full EM steps: $(full_em_steps)
          K-fold partition: $(i)
          """)
      c_sizes[i] = size(C)
      tee(out_data, "Size: " * string(c_sizes[i]) * " -> " * string(sum(c_sizes[i])))
      tee(out_data, "LL: " * string(-NLL(C, T)))
      println("Learning parameters...")
      learner = SEM(C; gauss = true)
      indices = shuffle!(collect(1:size(R,1)))
      tee(out_data, "Mini-batch EM...")
      avgnll = 0.0
      runnll = 0.0
      batch_time = @elapsed while learner.steps < em_steps
        sid = rand(1:(length(indices)-batchsize))
        batch = view(R, indices[sid:(sid+batchsize-1)], :)
        η = 0.975^learner.steps
        update(learner, batch, η, 0.05, true, 0.01; verbose, validation = T)
      end
      tee(out_data, "Batch EM LL: " * string(-NLL(C, T)))
      tee(out_data, "Full EM...")
      η = 1.0
      em_time = @elapsed while learner.steps < full_em_steps
        update(learner, R, η, 0.05, true, 0.01; verbose, validation = T)
      end
      lls[i] = -NLL(C, T)
      tee(out_data, "Full EM LL: " * string(lls[i]))
      tee(out_data, "Size: " * string(c_sizes[i]))
      tee(out_data, @sprintf("Vtree time: %.8f\nCircuit time: %.8f\nBatch EM time: %.8f\nFull EM time: %.8f",
                             v_time, c_time, batch_time, em_time))
      c_times[i], v_times[i], batch_times[i], em_times[i] = c_time, v_time, batch_time, em_time
    end
    LL[data_idx] = mean(lls)
    _s = [0, 0, 0]; for x ∈ c_sizes _s .= (_s[1]+x[1], _s[2]+x[2], _s[3]+x[3]) end; _s /= length(c_sizes)
    S[data_idx] = Tuple(round.(Int, _s))
    tee(out_data, "\n\nDataset: $(datasets[data_idx])\n=======\nAverage LL: " * string(LL[data_idx]))
    tee(out_data, "Average size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Average times:\n-------\nVtree time: %.8f\nCircuit time: %.8f\nBatch EM time: %.8f\nFull EM time: %.8f\n=======\n\n",
                           mean(v_times), mean(c_times), mean(batch_times), mean(em_times)))
    close(out_data)
  end
  println(LL)
  serialize("results/rpp_$(name).data", LL)
  serialize("results/rpp_$(name)_size.data", S)
end

# run_bin()
run_cont()
