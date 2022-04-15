using Pkg; Pkg.activate("..")
using Random
using RPCircuits
using NPZ
# using ThreadPools

function learn_parameters!(C::Node, R::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real};
    full_steps::Integer = 100, batch_steps::Integer = 100, batchsize::Integer = 100,
    gauss::Bool = false, min_var::Real = 0.1, smoothing::Real = 1e-4, verbose::Bool = true)
  println("Compiling circuit...")
  L = SEM(C; gauss)
  I = shuffle!(collect(1:size(R, 1)))
  println("Batch EM...")
  while L.steps < batch_steps
    sid = rand(1:(length(I)-batchsize))
    B = view(R, I[sid:(sid+batchsize-1)], :)
    η = 0.975^L.steps
    update(L, B; learningrate=η, smoothing=smoothing, verbose=verbose, validation=T)
  end
  println("Full EM...")
  while L.steps < full_steps
    update(L, R; learningrate=1.0, smoothing=smoothing, verbose=verbose, validation=T)
  end
  return C
end

println("Loading datasets")
R, V, T = npzread.("moons." .* ["train", "valid", "test"] .* ".npy")

L_f = [
       () -> learn_projections(R; min_examples = 5, max_height = 5, n_projs = 3, t_proj = :sid, c = 10.0, t_mix = :all),
       () -> learn_projections(R; min_examples = 5, max_height = 10, t_proj = :sid, t_mix = :single, c = 10.0),
       () -> learn_projections(R; min_examples = 5, max_height = 5, n_projs = 3, t_proj = :max, r = 1.0, c = 10.0, t_mix = :all),
       () -> learn_projections(R; min_examples = 5, max_height = 10, t_proj = :max, r = 1.0, t_mix = :single, c = 10.0),
      ]

load_from_file = false

configs = [(false, 0.0, :batch, "batch_nogauss"), (true, 0.1, :batch, "batch_gauss01"), (true, 0.05, :batch, "batch_gauss005"), (true, 0.01, :batch, "batch_gauss001"),
           (false, 0.0, :full, "nogauss"), (true, 0.1, :full, "gauss01"), (true, 0.05, :full, "gauss005"), (true, 0.01, :full, "gauss001")]
names = ["sid", "sid_single", "max", "max_single"]
LL = Vector{Vector{Float64}}(undef, length(configs))
for (j, (use_gauss, min_variance, em_type, pre)) ∈ enumerate(configs)
  LL_t = Vector{Float64}(undef, length(names))
  for i ∈ 1:length(names)
    println(names[i])
    println("Learning structure...")
    C = L_f[i]()
    println("Learning parameters...")
    learn_parameters!(C, R, V; batchsize = em_type == :batch ? 100 : -1,
                      batch_steps = em_type == :batch ? 100 : 0, full_steps = em_type == :full ? 100 : 0,
                      gauss = use_gauss, min_var = min_variance)
    ll = -NLL(C, T)
    LL_t[i] = ll
    println("$(pre) $(names[i]) -> LL: ", ll)
    open("results/moons/$(pre)_$(names[i]).txt", "w") do out write(out, string(ll)) end
    # println("Saving to file...")
    # save(C, "saved/synt1d/" * names[i] * ".pc")
  end
  LL[j] = LL_t
end
# x_bounds, y_bounds = -1.5:0.05:2.5, -0.75:0.025:1.25
# bounds = vec([[i, j] for i in x_bounds, j in y_bounds])
# for (l, (use_gauss, min_variance, em_type, pre)) ∈ enumerate(configs)
  # C_t = C_all[l]
  # for (k, name) ∈ enumerate(names)
    # C = C_t[k]
    # L = leaves(C)
    # n = length(L)÷2
    # S = Matrix{Float64}(undef, n, 2)
    # M = Matrix{Float64}(undef, n, 2)
    # println("Exporting Gaussians...")
    # for i ∈ 1:n
      # j = 2*(i-1)+1
      # S[i,1], S[i,2] = L[j].variance, L[j+1].variance
      # M[i,1], M[i,2] = L[j].mean, L[j+1].mean
    # end
    # npzwrite("results/moons/$(pre)_$(name)_mean.npy", M)
    # npzwrite("results/moons/$(pre)_$(name)_variance.npy", S)
    # println("Exporting densities...")
    # density = Vector{Float64}(undef, length(bounds))
    # println("Computing densities...")
    # @Threads.threads for i ∈ 1:length(bounds)
      # density[i] = logpdf(C, bounds[i])
    # end
    # npzwrite("results/moons/$(pre)_$(name)_density.npy", density)
    # println(name, "... OK.")
  # end
# end
