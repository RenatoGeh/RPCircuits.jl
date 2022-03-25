struct CSum <: Inner
  children::Vector{UInt}
  weights::Vector{Float64}
end

struct CProduct <: Inner
  children::Vector{UInt}
end

struct GStore
  G::Vector{UInt}
  μ::Dict{UInt, Float64}
  squares::Dict{UInt, Float64}
  denon::Dict{UInt, Float64}
end

mutable struct CCircuit
  C::Vector{Node} # current
  P::Vector{Node} # previous
  V::Vector{Float64} # logpdf's
  D::Vector{Float64} # derivatives
  L::Vector{Vector{UInt}} # layers
  S::Vector{UInt} # sums
  I::Union{Nothing, Vector{UInt}} # indicators
  gauss::Union{Nothing, GStore} # gaussians
end

@inline issum(n::CSum)::Bool = true
@inline isprod(n::CProduct)::Bool = true

@inline function swap!(bundle::CCircuit)
  bundle.C, bundle.P = bundle.P, bundle.C
  return nothing
end

function compile(::Type{CCircuit}, r::Node; gauss::Bool = false)::CCircuit
  N = nodes(r; rev = false)
  oL = layers(r)
  n = length(N)
  C = Vector{Node}(undef, n)
  P = Vector{Node}(undef, n)
  M = Dict{Node, UInt}(x => i for (i, x) ∈ enumerate(N))
  Threads.@threads for i ∈ 1:n
    u = N[i]
    if issum(u)
      m = length(u.children)
      ch = Vector{UInt}(undef, m)
      @inbounds for j ∈ 1:m ch[j] = M[u.children[j]] end
      C[i] = CSum(ch, u.weights)
      P[i] = CSum(ch, copy(u.weights))
    elseif isprod(u)
      m = length(u.children)
      ch = Vector{UInt}(undef, m)
      @inbounds for j ∈ 1:m ch[j] = M[u.children[j]] end
      C[i] = CProduct(ch)
      P[i] = CProduct(ch)
    else C[i], P[i] = u, u end
  end
  L = Vector{Vector{UInt}}(undef, length(oL))
  for (i, ol) ∈ enumerate(oL)
    m = length(ol)
    l = Vector{UInt}(undef, m)
    Threads.@threads for j ∈ 1:m l[j] = M[ol[j]] end
    L[i] = l
  end
  V, D = zeros(n), zeros(n)
  gs, I = nothing, nothing
  if gauss
    G = [i for (i, x) ∈ enumerate(C) if isa(x, Gaussian)]
    μ = Dict{UInt, Float64}(n => 0.0 for n ∈ G)
    squares = Dict{UInt, Float64}(n => 0.0 for n ∈ G)
    denon = Dict{UInt, Float64}(n => 0.0 for n ∈ G)
    gs = GStore(G, μ, squares, denon)
  else
    I = [i for (i, x) ∈ enumerate(C) if isa(x, Indicator)]
  end
  return CCircuit(C, P, V, D, L, [i for (i, x) ∈ enumerate(C) if isa(x, CSum)], I, gs)
end
