using Pkg; Pkg.activate("..")
using RPCircuits
using Random
using Serialization
using Printf
using Statistics
using Logging

using BlossomV
using ProbabilisticCircuits: learn_vtree
using LogicCircuits: Vtree
using DataFrames

using GaussianMixtures

# using Debugger
# break_on(:error)

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
  history = Vector{Vector{Float64}}(undef, length(datasets))
  history_rand = Vector{Vector{Float64}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_rpp_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
    println("Learning structure...")
    # Random.seed!(1)
    v_time = @elapsed vtree = learn_vtree(DataFrame(R, :auto); alg = :bottomup)
    println("  Learning circuit...")
    c_time = @elapsed C = learn_rpp(R, vtree; split, max_height, bin = true, min_examples = 20, pseudocount = 0, trials = 100)
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
    ranges = prepare_step_indices(size(R, 1), batchsize)
    tee(out_data, "Mini-batch EM...")
    H = Vector{Float64}()
    # Random.seed!(1)
    indices = collect(1:size(R, 1))
    batch_time = @elapsed while learner.steps < em_steps
      if learner.steps % 2 == 0 shuffle!(indices) end
      I = view(indices, ranges[(learner.steps % length(ranges)) + 1])
      batch = view(R, I, :)
      η = 0.975^learner.steps
      update(learner, batch; learningrate=η, smoothing=smoothing, verbose=verbose, validation=T, history=H)
      # update(learner, batch; learningrate=η, smoothing=1e-4, verbose=false, validation=V)
    end
    tee(out_data, "Batch EM LL: " * string(-NLL(C, T)))
    tee(out_data, "Full EM...")
    η = 1.0
    em_time = @elapsed while learner.steps < full_em_steps
      if datasets[data_idx] == "c20ng" oupdate(learner, R, 2000; learningrate=η, smoothing=smoothing, verbose=verbose, validation=T)
      else update(learner, R; learningrate=η, smoothing=smoothing, verbose=verbose, validation=T) end
    end
    LL[data_idx] = -NLL(C, T)
    push!(H, -LL[data_idx])
    history[data_idx] = H
    tee(out_data, "Full EM LL: " * string(LL[data_idx]))
    tee(out_data, "Size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Vtree time: %.8f\nCircuit time: %.8f\nBatch EM time: %.8f\nFull EM time: %.8f",
                           v_time, c_time, batch_time, em_time))
    # randomize!(C)
    # println("Randomizing...")

    # learner = SEM(C)
    # indices = shuffle!(collect(1:size(R,1)))
    # avgnll = 0.0
    # runnll = 0.0
    # H = Vector{Float64}()
    # batch_time = @elapsed while learner.steps < em_steps
      # sid = rand(1:(length(indices)-batchsize))
      # batch = view(R, indices[sid:(sid+batchsize-1)], :)
      # η = 0.975^learner.steps
      # update(learner, batch; learningrate=η, smoothing=1e-4, verbose=verbose, validation=T, history=H)
    # end
    # η = 1.0
    # em_time = @elapsed while learner.steps < full_em_steps
      # update(learner, R; learningrate=η, verbose=false, validation=V)
    # end
    # push!(H, NLL(C, T))
    # history_rand[data_idx] = H
    close(out_data)
  end
  println(LL)
  serialize("results/rpp_$(name).data", LL)
  serialize("results/rpp_$(name)_size.data", S)
  serialize("results/rpp_$(name)_hist.data", history)
  # serialize("results/rpp_$(name)_rand_hist.data", history_rand)
end

cont_data = ["abalone", "banknote", "ca", "kinematics", "quake", "sensorless", "chemdiab", "flowsize",
             "oldfaithful", "iris"]

smoothing, minimumvariance = 1e-4, 1e-3

function run_cont()
  # datasets = cont_data
  datasets = ["flowsize"]
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_rpp_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    O = continuous_datasets(datasets[data_idx]; as_df = false, normalize = false)
    # O = O[shuffle(1:size(O, 1)), :]
    # D, E = normalize(O)
    # P = partition_kfold(D, 10)
    P = partition_kfold(O, 10)
    # Q = partition_kfold(O, 10)
    lls = zeros(10)
    c_sizes = Vector{Tuple{Int, Int, Int}}(undef, 10)
    v_times, c_times, batch_times, em_times = zeros(10), zeros(10), zeros(10), zeros(10)
    # lls_rand = zeros(10)
    for (i, (T, R)) ∈ enumerate(P)
      println("Learning structure...")
      l = -Inf
      C, v_time, c_time = nothing, nothing, nothing
      Random.seed!(2)
      v_time = @elapsed vtree = learn_vtree_cont(R; alg = :bottomup)
      println("  Learning circuit...")
      # c_time = @elapsed C = learn_rpp_auto(R, vtree; samples = 20, split, max_height, bin = false, min_examples = 20, gmm = true)
      c_time = @elapsed C = learn_rpp(R, vtree; max_projs = 1, split, max_height, bin = false, min_examples = 100, gmm = true)
      # C_r = randomize(C)
      # println("  Number of projections: ", length(P))
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
      l = -NLL(C, T)
      tee(out_data, "LL: " * string(l))

      println("Learning parameters...")
      learner = SEM(C; gauss = true)
      indices = shuffle!(collect(1:size(R,1)))
      tee(out_data, "Mini-batch EM...")
      n = size(R, 1)
      l_batch = batchsize
      if n < 5000 l_batch = 100 end
      if n < 300 l_batch = 50 end
      batch_time = @elapsed while learner.steps < em_steps
        sid = rand(1:(length(indices)-l_batch))
        batch = view(R, indices[sid:(sid+l_batch-1)], :)
        η = 0.975^learner.steps
        update(learner, batch; learningrate=η, smoothing=smoothing, learngaussians=true, minimumvariance=minimumvariance, verbose=verbose, validation=T)
      end
      tee(out_data, "Batch EM LL: " * string(-NLL(C, T)))
      tee(out_data, "Full EM...")
      η = 1.0
      em_time = @elapsed while learner.steps < full_em_steps
        update(learner, R; learningrate=η, smoothing=smoothing, learngaussians=true, minimumvariance=minimumvariance, verbose=verbose, validation=T)
      end
      # rescale_gauss!(C, E)
      # lls[i] = -NLL(C, Q[i][1])
      lls[i] = -NLL(C, T)
      # println("Training LL: ", -NLL(C, Q[i][2]))
      println("Training LL: ", -NLL(C, R))

      # println("Learning parameters from randomized circuit...")
      # learner = SEM(C_r; gauss = true)
      # indices = shuffle!(collect(1:size(R,1)))
      # tee(out_data, "Mini-batch EM...")
      # l_batch = batchsize > size(R, 1) ? size(R, 1) : batchsize
      # batch_time = @elapsed while learner.steps < em_steps
        # sid = rand(1:(length(indices)-l_batch))
        # batch = view(R, indices[sid:(sid+l_batch-1)], :)
        # η = 0.975^learner.steps
        # update(learner, batch; learningrate=η, smoothing=smoothing, learngaussians=true, minimumvariance=minimumvariance, verbose=verbose, validation=T)
      # end
      # tee(out_data, "Batch EM LL: " * string(-NLL(C_r, T)))
      # tee(out_data, "Full EM...")
      # η = 1.0
      # em_time = @elapsed while learner.steps < full_em_steps
        # update(learner, R; learningrate=η, smoothing=smoothing, learngaussians=true, minimumvariance=minimumvariance, verbose=verbose, validation=T)
      # end
      # lls_rand[i] = -NLL(C_r, T)

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
    # tee(out_data, "Average randomized LL: " * string(mean(lls_rand)))
    tee(out_data, "Average size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Average times:\n-------\nVtree time: %.8f\nCircuit time: %.8f\nBatch EM time: %.8f\nFull EM time: %.8f\n=======\n\n",
                           mean(v_times), mean(c_times), mean(batch_times), mean(em_times)))
    close(out_data)
  end
  println(LL)
  serialize("results/rpp_$(name).data", LL)
  serialize("results/rpp_$(name)_size.data", S)
end

function run_gmm()
  datasets = cont_data
  # datasets = ["flowsize"]
  L_v = Vector{Vector{Float64}}(undef, length(datasets))
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/gmm_$(name)_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    O = continuous_datasets(datasets[data_idx]; as_df = false, normalize = false)
    P = partition_kfold(O, 10)
    lls = zeros(10)
    for (i, (T, R)) ∈ enumerate(P)
      println("Learning structure...")
      M = GMM(5, R; method = :kmeans, kind = :diag, nInit = 100, nIter = 0)
      println("Learning parameters...")
      em!(M, R; nIter = 100, varfloor = 1e-3)
      lls[i] = sum(GaussianMixtures.logsumexpw(GaussianMixtures.llpg(M, T), M.w))/size(T, 1)
      tee(out_data, "LL: " * string(lls[i]))
    end
    LL[data_idx] = mean(lls)
    L_v[data_idx] = lls
    tee(out_data, "\n\nDataset: $(datasets[data_idx])\n=======\nAverage LL: " * string(LL[data_idx]))
    close(out_data)
  end
  println(LL)
  serialize("results/gmm_$(name).data", LL)
  return L_v
end

function run_gmm_circ()
  datasets = cont_data
  L_v = Vector{Vector{Float64}}(undef, length(datasets))
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/gmm_circ_$(name)_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    O = continuous_datasets(datasets[data_idx]; as_df = false, normalize = false)
    P = partition_kfold(O, 10)
    lls = zeros(10)
    for (i, (T, R)) ∈ enumerate(P)
      println("Partition ", i)
      M = learn_multi_gmm(R; k = 5, kmeans_iter = 100, em_iter = 100, validation = T, minvar = 1e-3)
      lls[i] = -NLL(M, T)
      tee(out_data, "LL: " * string(lls[i]))
    end
    LL[data_idx] = mean(lls)
    L_v[data_idx] = lls
    tee(out_data, "\n\nDataset: $(datasets[data_idx])\n=======\nAverage LL: " * string(LL[data_idx]))
    close(out_data)
  end
  println(LL)
  serialize("results/gmm_circ_$(name).data", LL)
  return L_v
end

function run_rand_bin()
  # datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]
  datasets = all_data
  LL = Vector{Float64}(undef, length(datasets))
  S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
  history = Vector{Vector{Float64}}(undef, length(datasets))
  history_rand = Vector{Vector{Float64}}(undef, length(datasets))
  for data_idx ∈ 1:length(datasets)
    out_data = open("logs/$(name)_rand_rpp_$(datasets[data_idx]).log", "w")
    tee(out_data, "Dataset: " * datasets[data_idx])
    R, V, T = twenty_datasets(datasets[data_idx]; as_df = false)
    println("Learning structure...")
    v_time = @elapsed vtree = learn_vtree(DataFrame(R, :auto); alg = :bottomup)
    println("  Learning circuit...")
    c_time = @elapsed C = learn_rpp(R, vtree; split, max_height, bin = true, min_examples = 20, pseudocount = 0)
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
    H = Vector{Float64}()
    batch_time = @elapsed while learner.steps < em_steps
      sid = rand(1:(length(indices)-batchsize))
      batch = view(R, indices[sid:(sid+batchsize-1)], :)
      η = 0.975^learner.steps
      update(learner, batch; learningrate=η, smoothing=1e-4, verbose=verbose, validation=T, history=H)
      # update(learner, batch; learningrate=η, smoothing=1e-4, verbose=false, validation=V)
    end
    tee(out_data, "Batch EM LL: " * string(-NLL(C, T)))
    tee(out_data, "Full EM...")
    η = 1.0
    em_time = @elapsed while learner.steps < full_em_steps
      update(learner, R; learningrate=η, verbose=false, validation=V)
    end
    LL[data_idx] = -NLL(C, T)
    push!(H, -LL[data_idx])
    history[data_idx] = H
    tee(out_data, "Full EM LL: " * string(LL[data_idx]))
    tee(out_data, "Size: " * string(S[data_idx]))
    tee(out_data, @sprintf("Vtree time: %.8f\nCircuit time: %.8f\nBatch EM time: %.8f\nFull EM time: %.8f",
                           v_time, c_time, batch_time, em_time))
  end
  println(LL)
  serialize("results/rand_$(name).data", LL)
  serialize("results/rand_$(name)_size.data", S)
  serialize("results/rand_$(name)_hist.data", history)
end

global_logger(SimpleLogger(stdout, Logging.Error))
run_bin()
# run_cont()
# LL = run_gmm()
# LL = run_gmm_circ()
