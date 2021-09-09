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

"""
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
end

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
