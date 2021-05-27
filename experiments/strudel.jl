using Pkg; Pkg.activate("..")
using RPCircuits
using ProbabilisticCircuits
using Serialization

datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "mushrooms",
            "adult", "dna"]

LL = Vector{Float64}(undef, length(datasets))
for (i, data_name) ∈ enumerate(datasets)
  R, _, T = RPCircuits.twenty_datasets(data_name; as_df = true)
  C = learn_circuit(R; maxiter = 100)
  estimate_parameters(C, R; pseudocount = 1.0)
  LL[i] = log_likelihood_avg(C, T)
end
serialize("strudel.data", LL)
