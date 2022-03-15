using BlossomV
using SimpleWeightedGraphs
using DataFrames
using SparseArrays
using LogicCircuits: PlainVtree
import LogicCircuits
import ProbabilisticCircuits

function learn_vtree_cont(M::Matrix; α::Float64 = 0.0, alg = :bottomup)
  D = DataFrame(M, :auto)
  n, m = size(D)
  if alg == :topdown
    return PlainVtree(m, :topdown; f = top_down_v(D, M; α))
  elseif alg == :bottomup
    return PlainVtree(m, :bottomup; f = bottom_up_v(D, M; α))
  else
    return error("Vtree learner $(alg) not supported.")
  end
end
export learn_vtree_cont

"Metis top down method"
function top_down_v(data::DataFrame, M::Matrix; α)
    δINT = 999999
    MIN_INT = 1
    MAX_INT = δINT + MIN_INT
    n, m = size(data)

    weight=ones(Float64, n)
    C = abs.(cor(M))
    vars = UInt32.(collect(1:m))
    info = ProbabilisticCircuits.to_long_mi(C, MIN_INT, MAX_INT)

    function f(leafs::Vector{PlainVtreeLeafNode})::Tuple{Vector{PlainVtreeLeafNode}, Vector{PlainVtreeLeafNode}}
        var2leaf = Dict([(variable(x),x) for x in leafs])
        vertices = sort(variable.(leafs))
        sub_context = info[vertices, vertices]
        len = length(vertices)
        for i in 1 : len
            sub_context[i, i] = 0
        end
        g = convert(SparseMatrixCSC, sub_context)
        partition = ProbabilisticCircuits.my_partition(ProbabilisticCircuits.my_graph(g), 2)

        subsets = (Vector{PlainVtreeLeafNode}(), Vector{PlainVtreeLeafNode}())
        for (index, p) in enumerate(partition)
            push!(subsets[p], var2leaf[vertices[index]])
        end

        return subsets
    end
    return f
end

"Blossom bottom up method, vars are not used"
function bottom_up_v(data::DataFrame, M::Matrix; α)
    n, m = size(data)
    weight = ones(Float64, n)
    C = abs.(cor(M))
    vars = UInt32.(collect(1:m))
    info = round.(Int64, 1000001 .+ ProbabilisticCircuits.to_long_mi(C, -1, -1000000))

    function f(leaf::Vector{<:Vtree})
        variable_sets = collect.(variables.(leaf))

        # even number of nodes, use blossomv alg
        function blossom_bottom_up_even!(variable_sets)
            # 1. calculate pMI
            pMI = ProbabilisticCircuits.Utils.set_mutual_information(info, variable_sets)
            pMI = round.(Int64, pMI)

            # 2. solve by blossomv alg
            len = length(variable_sets)
            m = BlossomV.Matching(len)
            for i in 1 : len, j in i + 1 : len
                add_edge(m, i - 1, j - 1, pMI[i, j]) # blossomv index start from 0
            end

            solve(m)
            all_matches = Set{Tuple{UInt32, UInt32}}()
            for v in 1 : len
                push!(all_matches, LogicCircuits.Utils.order_asc(v, get_match(m, v - 1) + 1))
            end

            # 3. calculate scores, map index to var
            all_matches = Vector(collect(all_matches))
            score = 0

            for i in 1 : length(all_matches)
                (x, y) = all_matches[i]
                score += pMI[x, y]
            end

            return (all_matches, score)
        end

        # odd number of nodes, try every 2 combinations
        function blossom_bottom_up_odd!(variable_sets)
            # try all len - 1 conditions, find best score(minimun cost)
            (best_matches, best_score) = (nothing, typemax(Int64))
            len = length(variable_sets)
            for index in 1 : len
                indices = [collect(1:index-1);collect(index+1:len)]
                (matches, score) = blossom_bottom_up_even!(variable_sets[indices])
                if score < best_score
                    (best_matches, best_score) = ([[(indices[l], indices[r]) for (l,r) in matches];[index]], score)
                end
            end
            return (best_matches, best_score)
        end

        if length(variable_sets) % 2 == 0
            (matches, score) = blossom_bottom_up_even!(variable_sets)
        else
            (matches, score) = blossom_bottom_up_odd!(variable_sets)
        end

        pairs = []
        for x in matches
            if x isa Tuple
                push!(pairs, (leaf[x[1]], leaf[x[2]]))
            else
                push!(pairs, leaf[x])
            end
        end
        return pairs
    end
    return f
end
