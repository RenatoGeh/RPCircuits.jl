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
