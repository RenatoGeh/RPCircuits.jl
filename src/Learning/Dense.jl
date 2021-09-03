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
    root::RootRegion, C::Int, D::Int, R::Int, S::Int, I::Int; binary::Bool = true,
    V::Function = x -> x, pseudocount::Int = 1)::Circuit
  n_prods_sums = S*S
  n_prods_leaves = I*I
  if binary
    neg_leaves = [Indicator(V(x), 0) for x ∈ Sc]
    pos_leaves = [Indicator(V(x), 1) for x ∈ Sc]
    θ = (vec(sum(data; dims = 1)) .+ pseudocount) / (size(data, 1)+pseudocount)
  else
    μ = mean(data; dims = 1)
    σ = std(data; dims = 1)
    leaves = [(V(Sc[i]), μ[i], σ[i]*σ[i]) for i in 1:length(Sc)]
  end
  roots = Vector{Sum}(undef, C)
  for i ∈ 1:C
    circ[i] = Sum(n_prods_sums*R)
    root.sums[i] = circ[i]
  end
  for (i, part) ∈ enumerate(root.ch)
    for j ∈ 1:part.n
      p = Product(2)
      part.prods[j] = p
      for pa ∈ root.sums
        l = (i-1)*part.n+j
        pa.children[l] = p
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
    if Ch isa LeafRegion
      N = Vector{Leaf}(undef, I)
      n = length(Ch.sc)
      if n == 1
        x = first(Ch.sc)
        if binary
          ⊥, ⊤ = neg_leaves[x], pos_leaves[x]
          for i ∈ 1:I N[i] = Sum([⊥, ⊤], [1-θ[x], θ[x]]) end
        else
          for i ∈ 1:I N[i] = Gaussian(leaves[x]...) end
        end
      else
        for i ∈ 1:I N[i] = Product(n) end
        for p ∈ N
          for (i, x) ∈ enumerate(Ch.sc)
            if binary p.children[i] = Sum([neg_leaves[x], pos_leaves[x]], [1-θ[x], θ[x]])
            else p.children[i] = Gaussian(leaves[x]...) end
          end
        end
      end
    else # SumRegion
      N = Vector{Sum}(undef, S)
      m = Ch.ch.n
      Ch.ch.prods = [Product(2) for i ∈ 1:m]
      for i ∈ 1:S
        s = Sum(Ch.ch.prods, [1.0/m for i ∈ 1:m])
        N[i] = s
        Ch.sums[i] = s
      end
      append!(Q, ((Ch.ch.ch[1], Ch.ch, 1), (Ch.ch.ch[2], Ch.ch, 2)))
    end
    for i ∈ 1:length(Pa.prods) Pa.prods[i].children[which] = N[((i-1) % Ch.n) + 1] end
  end
  return roots
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
    D::Int, R::Int, S::Int, I::Int; binary::Bool = true, V::Function = x -> x,
    pseudocount::Int = 1)::Vector{Sum}
  return compile_regiongraph(data, Sc, sample_regiongraph(Sc, C, D, R, S, I), C, D, R, S, I;
                             binary, V, pseudocount)
end
export sample_dense

