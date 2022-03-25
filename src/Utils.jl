# Helper functions

"""
  expw = normexp!(logw,expw)

Compute expw .= exp.(logw)/sum(exp.(logw)). Uses the log-sum-exp trick to control for overflow in exp function.

Smoothing adds constant value to each value prior to normalization (useful to avoid zero probabilities).
"""
function normexp!(logw, expw, smoothing = 0.0)
  offset = maximum(logw)
  expw .= exp.(logw .- offset) .+ smoothing
  s = sum(expw)
  return expw .*= 1 / s
end

# function logsumexp!(w,we)
#     offset = maximum(w)
#     we .= exp.(w .- offset)
#     s = sum(we)
#     w .-= log(s) + offset # this computes logw
#     we .*= 1/s
# end

#="""
  logsumexp(x)

Returns `log(sum(exp.(x)))`. Uses a numerically stable algorithm.

# References:

  - https://arxiv.org/pdf/1412.8695.pdf eq 3.8
  - https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp(x::AbstractVector{<:Real})
  offset, maxind = findmax(x) # offset controls for overflow
  w = exp.(x .- offset) # Note: maximum(w) = 1
  Σ = sum_all_but(w, maxind) # Σ = sum(w)-1
  return log1p(Σ) + offset #, Σ+1 # log1p controls for underflow
end=#

using StatsFuns: logsumexp

"""
    sum_all_but(x,i)

Computes `sum(x) - x[i]`.
"""
function sum_all_but(x, i)
  x[i] -= 1
  s = sum(x)
  x[i] += 1
  return s
end

"""
Truncate to `Integer`s and count how many are in each bin, returning the resulting probabilities.
"""
function bincount(Y::AbstractVector{<:Real}, n::Int)::Vector{Float64}
  C = Vector{Int}(undef, n)
  for i ∈ 1:length(Y)
    C[convert(Int, Y[i])] += 1
  end
  C /= n
  return C
end

"""
Compares if all elements are equal.
"""
function allequal(E::AbstractVector)::Bool
  @assert !isempty(E) "Collection must be non-empty."
  u = first(E)
  for i ∈ 2:length(E)
    if u != E[i] return false end
  end
  return true
end

"""
    indicators(X::Union{AbstractVector{<:Integer}, UnitRange})::Tuple{Vector{Indicator}, Vector{Indicator}}

Returns all ⊤ and ⊥ indicators for each variable in X.
"""
function indicators(X::Union{AbstractVector{<:Integer}, UnitRange})::Tuple{Vector{Indicator}, Vector{Indicator}}
  return [Indicator(i, 0) for i ∈ X], [Indicator(i, 1) for i ∈ X]
end

"""
    mapcopy(r::Node)::Tuple{Node, Dict{Node, Node}}

Returns a deep copy of the circuit rooted at `r` and a mapping of each node in `r`'s network to its copy and vice-versa.
"""
function mapcopy(r::Node; converse::Bool = false)::Tuple{Node, Dict{Node, Node}}
  M = Dict{Node, Node}()
  function passdown(n::Node)
    if issum(n)
      Ch = Vector{Node}(undef, length(n.children))
      for (i, c) ∈ enumerate(n.children)
        if !haskey(M, c)
          M[c] = c # Temporarily set this as visited.
          u = passdown(c)
          M[c] = u # On backtrack, set this to the correct value.
          M[u] = c
          Ch[i] = u
        else Ch[i] = M[c] end
      end
      return Sum(Ch, copy(n.weights))
    elseif isprod(n)
      Ch = Vector{Node}(undef, length(n.children))
      for (i, c) ∈ enumerate(n.children)
        if !haskey(M, c)
          M[c] = c
          u = passdown(c)
          M[c] = u
          M[u] = c
          Ch[i] = u
        else Ch[i] = M[c] end
      end
      return Product(Ch)
    end
    u = copy(n)
    M[n] = u
    M[u] = n
    return u
  end
  z = passdown(r)
  M[r] = z
  M[z] = r
  return z, M
end

"""Computes, inplace, the Hadamard product: A∘B=C, where C_ij = A_ij*B_ij."""
function hadamard!(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
  n, m = size(A)
  for j ∈ 1:m
    @simd for i ∈ 1:n
      @inbounds A[i,j] *= B[i,j]
    end
  end
  return nothing
end

"Returns a(n index) partitioning a la k-fold."
function kfold(n::Int, p::Int)::Vector{Tuple{UnitRange, Vector{Int}}}
  F = Vector{Tuple{UnitRange, Vector{Int}}}(undef, p)
  j = s = 1
  k = n÷p
  for i ∈ 1:n%p
    if s > 1
      I = collect(1:s-1)
      if s+k < n append!(I, s+k+1:n) end
    else I = collect(s+k+1:n) end
    F[j] = (s:s+k, I)
    s += k+1
    j += 1
  end
  k = n÷p-1
  for i ∈ 1:p-n%p
    if s > 1
      I = collect(1:s-1)
      if s+k < n append!(I, s+k+1:n) end
    else I = collect(s+k+1:n) end
    F[j] = (s:s+k, I)
    s += k+1
    j += 1
  end
  return F
end

function partition_kfold(D::AbstractMatrix, k::Int)::Vector{Tuple{AbstractMatrix, AbstractMatrix}}
  n = size(D, 1)
  K = kfold(n, k)
  P = Vector{Tuple{Matrix, Matrix}}(undef, k)
  for i ∈ 1:k
    P[i] = (view(D, K[i][1], :), view(D, K[i][2], :))
  end
  return P
end
export partition_kfold

function rescale_gauss!(r::Node, extrem::Vector{Tuple{Float64, Float64}}; scale::Real = 1.0)::Node
  G = leaves(r)
  scq = scale*scale
  for g ∈ G
    minim, maxim = extrem[g.scope]
    d = maxim-minim
    s = d*d
    g.mean = (g.mean*d)/scale+minim
    g.variance *= s/scq
  end
  return r
end
export rescale_gauss!

"""
Returns a `Vector{UnitRange}` of ranges tentatively equally sized. The size of this collection is
equal to `k`, which by default is the number of available threads `Thread.nthreads()`.

  Example:

  `n=23` and `k=10`

  ```julia
  julia> prepare_indices(23, 10)

  10-element Vector{UnitRange{Int64}}:
   1:3
   4:5
   6:7
   8:9
   10:12
   13:14
   15:16
   17:18
   19:20
   21:23
  ```
"""
function prepare_indices(n::Int, k::Int = Threads.nthreads())::Vector{UnitRange{Int64}}
  s = floor.(Int, collect(range(1, n, k+1)))
  return [i == 1 ? (s[i]:s[i+1]) : (s[i]+1:s[i+1]) for i ∈ 1:length(s)-1]
end
export prepare_indices
function prepare_step_indices(n::Int, s::Int)::Vector{UnitRange{Int64}}
  r = 1:s:n
  I = [i == 1 ? (r[i]:r[i+1]) : (r[i]+1:r[i+1]) for i ∈ 1:length(r)-1]
  return r.stop != n ? push!(I, r.stop+1:n) : I
end
export prepare_step_indices

@inline isinvalid(x::Real)::Bool = isinf(x) || isnan(x)
@inline setinvalid!(X::AbstractVector{<:Real}, y::Real = -floatmax(eltype(X)))::Nothing = (X[map(isinvalid, X)] .= y; nothing)
