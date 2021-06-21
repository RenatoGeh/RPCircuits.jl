using Random

abstract type RGNode end
abstract type Region <: RGNode end

struct Partition <: RGNode
  n::Int
  ch::Tuple{Region, Region}
  prods::Vector{Product}
  Partition(n::Integer, ch1::Region, ch2::Region) = new(n, (ch1, ch2), Vector{Product}(undef, n))
end

struct RootRegion <: Region
  n::Int
  ch::Vector{Partition}
  sums::Vector{Sum}
  RootRegion(n::Integer, ch::Vector{Partition}) = new(n, ch, Vector{Sum}(undef, n))
end

struct SumRegion <: Region
  n::Int
  sc::AbstractVector{<:Integer}
  ch::Partition
  sums::Vector{Sum}
  SumRegion(n::Integer, sc::AbstractVector{<:Integer}, ch::Partition) = new(n, sc, ch, Vector{Sum}(undef, n))
end

struct LeafRegion <: Region
  n::Int
  sc::AbstractVector{<:Integer}
end

function split_scope(X::AbstractVector{<:Integer})::Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}}
  k = length(X) ÷ 2
  perm = shuffle(X)
  return @views perm[1:k], perm[k+1:end]
end

function sample_regiongraph(Sc::AbstractVector{<:Integer}, C::Int, D::Int, R::Int, S::Int, I::Int)::RootRegion
  function passdown(Z::AbstractVector{<:Integer}, depth::Int)::RGNode
    X, Y = split_scope(Z)
    if length(X) == 1 || depth > D
      R_1 = LeafRegion(I, X)
    else
      R_1 = SumRegion(S, X, passdown(X, depth+1))
    end
    if length(Y) == 1 || depth > D
      R_2 = LeafRegion(I, Y)
    else
      R_2 = SumRegion(S, Y, passdown(Y, depth+1))
    end
    P = Partition(R_1.n*R_2.n, R_1, R_2)
  end
  return RootRegion(C, [passdown(Sc, 1) for i in 1:R])
end

function compile_regiongraph(data::AbstractMatrix{<:Real}, Sc::AbstractVector{<:Integer},
    root::RootRegion, C::Int, D::Int, R::Int, S::Int, I::Int; offset = 0, binary::Bool = true)::Circuit
  n_prods_sums = S*S
  n_prods_leaves = I*I
  node_id = offset + C
  if binary
    neg_leaves = [Indicator(x, 0) for x ∈ Sc]
    pos_leaves = [Indicator(x, 1) for x ∈ Sc]
    θ = vec(sum(data; dims = 1)) / size(data, 1)
  else
    μ = mean(data; dims = 1)
    σ = std(data; dims = 1)
    leaves = [Gaussian(Sc[i], μ[i], σ[i]*σ[i]) for i in 1:length(Sc)]
  end
  circ = Vector{Node}(undef, C)
  for i ∈ 1:C
    circ[i] = Sum(n_prods_sums*R)
    root.sums[i] = circ[i]
  end
  for (i, part) ∈ enumerate(root.ch)
    for j ∈ 1:part.n
      p = Product(2)
      part.prods[j] = p
      push!(circ, p)
      node_id += 1
      for pa ∈ root.sums
        l = (i-1)*part.n+j
        pa.children[l] = node_id
        pa.weights[l] = 1.0/(n_prods_sums*R)
      end
    end
  end
  Q = Vector{Tuple{Region, Partition, Int}}()
  for i ∈ 1:R
    append!(Q, ((root.ch[i].ch[1], root.ch[i], 1), (root.ch[i].ch[2], root.ch[i], 2)))
  end
  while !isempty(Q)
    Ch, Pa, which = popfirst!(Q)
    for i ∈ 1:length(Pa.prods)
      Pa.prods[i].children[which] = node_id + (i % Ch.n) + 1
    end
    if Ch isa LeafRegion
      n = length(Ch.sc)
      if n == 1
        x = first(Ch.sc)
        if binary
          ⊥, ⊤ = node_id+I+1, node_id+I+2
          for i ∈ 1:I
            push!(circ, Sum([⊥, ⊤], [1-θ[x], θ[x]]))
            node_id += 1
          end
          append!(circ, (neg_leaves[x], pos_leaves[x]))
          node_id += 2
        else
          for i ∈ 1:I
            push!(circ, leaves[x])
            node_id += 1
          end
        end
      else
        inputs = [Product(n) for i ∈ 1:I]
        append!(circ, inputs)
        node_id += I
        for p ∈ inputs
          for (i, x) ∈ enumerate(Ch.sc)
            node_id += 1
            if binary
              p.children[i] = node_id
              append!(circ, (Sum([node_id+1, node_id+2], [1-θ[x], θ[x]]), neg_leaves[x], pos_leaves[x]))
              node_id += 2
            else
              p.children[i] = node_id
              push!(circ, leaves[x])
            end
          end
        end
      end
    else # SumRegion
      m = Ch.ch.n
      s_ch = Vector{UInt}(undef, m)
      s_w = Vector{Float64}(undef, m)
      for i ∈ 1:m
        Ch.ch.prods[i] = Product(2)
        s_ch[i] = node_id+S+i
        s_w[i] = 1.0/m
      end
      for i ∈ 1:S
        Ch.sums[i] = Sum(s_ch, s_w)
        push!(circ, Ch.sums[i])
      end
      append!(circ, Ch.ch.prods)
      node_id += S+m
      append!(Q, ((Ch.ch.ch[1], Ch.ch, 1), (Ch.ch.ch[2], Ch.ch, 2)))
    end
  end
  return circ
end

"""
Generates a random dense circuit.

`C` is the number of root nodes.
`D` is the max depth.
`R` is the number of splits.
`S` is the number of sum nodes in each region.
`I` is the number of input nodes in each region.

See "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning", Peharz et al, UAI 2019
"""
@inline function sample_dense(data::AbstractMatrix{<:Real}, Sc::AbstractVector{<:Integer}, C::Int,
    D::Int, R::Int, S::Int, I::Int; offset::Int = 0, binary::Bool = true)::Circuit
  return compile_regiongraph(data, Sc, sample_regiongraph(Sc, C, D, R, S, I), C, D, R, S, I; offset, binary)
end
export sample_dense

