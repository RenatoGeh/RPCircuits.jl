using Pkg; Pkg.activate("..")
using RPCircuits
using Serialization

datasets = ["nltcs", "book", "plants", "baudio", "jester", "bnetflix", "accidents", "dna"]

function count_nodes(c::Circuit, d::AbstractMatrix{<:Real})::Tuple{Int, Int, Int}
  sums, prods, leaves = 0, 0, size(d, 2)*2
  for n ∈ c
    if issum(n) sums += 1
    elseif isprod(n) prods += 1 end
  end
  return sums, prods, leaves
end

LL = Vector{Float64}(undef, length(datasets))
C = Vector{Circuit}(undef, length(datasets))
S = Vector{Tuple{Int, Int, Int}}(undef, length(datasets))
Threads.@threads for i ∈ 1:length(datasets)
  data_name = datasets[i]
  # if data_name == "adult" LL[i] = -Inf; continue end
  println(data_name)
  _, _, T = twenty_datasets(data_name; as_df = false)
  C[i] = Circuit("learnspns/20-datasets/$(data_name)/$(data_name).spn"; offset = 1, ind_offset = 1)
  # LL[i] = -NLL(C[i], T)
  S[i] = count_nodes(C[i], T)
  # println(data_name, ": ", LL[i])
end
# serialize("learnspn_ll.data", LL)
serialize("learnspn_size.data", S)
