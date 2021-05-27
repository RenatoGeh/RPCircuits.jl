using Pkg; Pkg.activate("..")
using RPCircuits
using Serialization

datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "mushrooms",
            "adult", "dna"]

LL = Vector{Float64}(undef, length(datasets))
C = Vector{Circuit}(undef, length(datasets))
for (i, data_name) ∈ enumerate(datasets)
  if data_name == "adult" LL[i] = -Inf; continue end
  _, _, T = twenty_datasets(data_name; as_df = false)
  C[i] = Circuit("learnspns/20-datasets/$(data_name)/$(data_name).spn"; offset = 1)
  LL[i] = -NLL(C[i], T)
  println(data_name, ": ", LL[i])
end
serialize("learnspn_ll.data", LL)
