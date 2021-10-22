# Routines for SPN evaluation and sampling

"""
(r::Node)(x::AbstractVector{<:Real})
(r::Node)(x...)

Evaluates the circuit rooted at `r` at a given instantiation `x` of the variables.
Summed-out variables are represented as `NaN`s.

# Parameters

  - `x`: vector of values of variables (integers or reals).

# Examples

To compute the probability of `P(b=1)` using circ rooted at `S`, use

```@example
S = Circuit(IOBuffer("1 + 2 0.2 3 0.5 4 0.3\n2 * 5 7\n3 * 5 8\n4 * 6 8\n5 categorical 1 0.6 0.4\n6 categorical 1 0.1 0.9\n7 categorical 2 0.3 0.7\n8 categorical 2 0.8 0.2"));
println(S([NaN, 2]))
println(S(NaN, 2))
```
"""
(r::Node)(x::Example) = exp(logpdf(r, x))
(r::Node)(x...) = exp(logpdf(r, [x...]))

"""
    logpdf!(values, circ, X)

Computes log-probabilities of instances `x` in `X` and stores the results in a given vector. Uses
multithreading to speed up computations.

# Parameters

  - `results`: vector to store results (log-probabilities)
  - `circ`: Sum-Product Network
  - `X`: matrix of values of variables (integers or reals). Summed-out variables are represented as `NaN`s.
"""
function logpdf!(results::AbstractVector{Float64}, r::Node, X::AbstractMatrix{<:Real})
  n = size(X, 1)
  @assert length(results) == n
  values = [Dict{Node, Float64}() for i ∈ 1:Threads.nthreads()]
  N = nodes(r; rev = false)
  Threads.@threads for i ∈ 1:n
    @inbounds results[i] = logpdf!(values[Threads.threadid()], N, view(X, i, :))
  end
  return nothing
end

"""
    logpdf!(values::Dict{Node, Float64}, N::Vector{Node}, x::AbstractVector{<:Real})::Float64

Evaluates the reverse-topologically-sorted circuit `N` in log domain at configuration `x` and
stores values of each node in the dictionary `values`.
"""
function logpdf!(values::Dict{Node, Float64}, N::Vector{Node}, x::AbstractVector{<:Real})::Float64
  for n ∈ N
    if isprod(n)
      lval = 0.0
      for c ∈ n.children lval += values[c] end
      values[n] = isfinite(lval) ? lval : -Inf
    elseif issum(n)
      m = -Inf
      for c ∈ n.children
        u = values[c]
        (u > m) && (m = u)
      end
      lval = 0.0
      for (i, c) ∈ enumerate(n.children) lval += exp(values[c]-m) * n.weights[i] end
      values[n] = isfinite(lval) ? m + log(lval) : -Inf
    else
      values[n] = logpdf(n, x[n.scope])
    end
  end
  return values[last(N)]
end
export logpdf!

"""
    logpdf(r::Node, X::Data)

Returns the sums of the log-probabilities of instances `x` in `X`. Uses multithreading to speed up
computations if `JULIA_NUM_THREADS > 1`.

# Parameters

  - `r`: Sum-Product Network
  - `X`: matrix of values of variables (integers or reals). Summed-out variables are represented as
    `NaN`s.
"""
function logpdf(r::Node, X::Data)::Float64
  k = Threads.nthreads()
  N = nodes(r; rev = false)
  n = size(X, 1)
  if k == 1
    values = Dict{Node, Float64}()
    return sum(logpdf!(values, N, view(X, i, :)) for i ∈ 1:n)
  end
  values = [Dict{Node, Float64}() for i ∈ 1:Threads.nthreads()]
  s = Threads.Atomic{Float64}(0.0)
  Threads.@threads for i ∈ 1:n
    v = logpdf!(values[Threads.threadid()], N, view(X, i, :))
    Threads.atomic_add!(s, v)
  end
  return s[]
end
export logpdf

"""
    logpdf(r::Node, x::AbstractVector{<:Real})::Float64

Evaluates the circuit `circ` in log domain at configuration `x`.

# Parameters

  - `x`: vector of values of variables (integers or reals). Summed-out variables are represented as `NaN`s

# Examples

To compute the probability of `P(b=1)` using circ `S`, use

```@example
S = Circuit(IOBuffer("1 + 2 0.2 3 0.5 4 0.3\n2 * 5 7\n3 * 5 8\n4 * 6 8\n5 categorical 1 0.6 0.4\n6 categorical 1 0.1 0.9\n7 categorical 2 0.3 0.7\n8 categorical 2 0.8 0.2"));
logpdf(S, [NaN, 2])
```
"""
logpdf(r::Node, x::AbstractVector{<:Real})::Float64 = logpdf!(Dict{Node, Float64}(), nodes(r; rev = false), x)

function cplogpdf!(
    V::Vector{Float64},
    N::Vector{Node},
    nlayers::Vector{Vector{UInt}},
    x::AbstractVector{<:Real},
)::Float64
  # visit layers from last (leaves) to first (root)
  @inbounds for l in length(nlayers):-1:1
    # parallelize computations within layer
    Threads.@threads for i in nlayers[l]
      n = N[i]
      if isprod(n)
        lval = 0.0
        for c in n.children lval += V[c] end
        V[i] = isfinite(lval) ? lval : -Inf
      elseif issum(n)
        # log-sum-exp trick to improve numerical stability (assumes weights are normalized)
        m = -Inf
        # get maximum incoming value
        for c in n.children
          u = V[c]
          (u > m) && (m = u)
        end
        lval = 0.0
        for (j, c) in enumerate(n.children)
          # ensure exp in only computed on nonpositive arguments (avoid overflow)
          lval += exp(V[c] - m) * n.weights[j]
        end
        # if something went wrong (e.g. incoming value is NaN or Inf) return -Inf, otherwise
        # return maximum plus lval (adding m ensures signficant digits are numerically precise)
        V[i] = isfinite(lval) ? m + log(lval) : -Inf
      else # is a leaf node
        V[i] = logpdf(n, x[n.scope])
      end
    end
  end
  return V[first(first(nlayers))]
end

"""
    plogpdf!(values::Dict{Node, Float64}, nlayers::Vector{Vector{Node}}, x::AbstractVector{<:Real})::Float64

Evaluates the circuit in log domain at configuration `x` using the scheduling in `nlayers` as
obtained by the method `layers(circ)`. Stores values of each node in the dictionary `values`.

# Parameters

  - `values`: dictionary to cache node values
  - `nlayers`: Vector of vector of nodes determinig the layers of the circuit; each node in a layer is computed based on values of nodes in smaller layers.
  - `x`: Vector containing assignment
"""
function plogpdf!(
    I::Dict{Node, Int},
    V::Vector{Float64},
    nlayers::Vector{Vector{Node}},
    x::AbstractVector{<:Real},
)::Float64
  # visit layers from last (leaves) to first (root)
  @inbounds for l in length(nlayers):-1:1
    # parallelize computations within layer
    Threads.@threads for n in nlayers[l]
      if isprod(n)
        lval = 0.0
        for c in n.children lval += V[I[c]] end
        lval = isfinite(lval) ? lval : -Inf
        V[I[n]] = lval
      elseif issum(n)
        # log-sum-exp trick to improve numerical stability (assumes weights are normalized)
        m = -Inf
        # get maximum incoming value
        for c in n.children
          u = V[I[c]]
          (u > m) && (m = u)
        end
        lval = 0.0
        for (j, c) in enumerate(n.children)
          # ensure exp in only computed on nonpositive arguments (avoid overflow)
          u = V[I[c]]
          lval += exp(u - m) * n.weights[j]
        end
        # if something went wrong (e.g. incoming value is NaN or Inf) return -Inf, otherwise
        # return maximum plus lval (adding m ensures signficant digits are numerically precise)
        lval = isfinite(lval) ? m + log(lval) : -Inf
        V[I[n]] = lval
      else # is a leaf node
        lval = logpdf(n, x[n.scope])
        V[I[n]] = lval
      end
    end
  end
  return V[I[first(first(nlayers))]]
end

@inline function plogpdf!(values::Dict{Node, Float64}, L::Vector{Vector{Node}}, x::AbstractVector{<:Real})::Float64
  I, i = Dict{Node, Int}(), 1
  for v ∈ L
    for n ∈ v I[n] = i; i += 1 end
  end
  V = Vector{Float64}(undef, sum(length.(L)))
  p = plogpdf!(I, V, L, x)
  for v ∈ L for n ∈ v values[n] = V[I[n]] end end
  return p
end
export plogpdf!

"""
    plogpdf(r, x)

Parallelized version of `logpdf(r, x)`. Set `JULIA_NUM_THREADS` > 1 to make this effective.
"""
@inline function plogpdf(r::Node, x::AbstractVector{<:Real})::Float64
  I, L, i = Dict{Node, Int}(), layers(r), 1
  for v ∈ L
    for n ∈ v I[n] = i; i += 1 end
  end
  V = Vector{Float64}(undef, sum(length.(L)))
  return plogpdf!(I, V, L, x)
end
@inline function plogpdf(r::Node, X::AbstractMatrix{<:Real})::Float64
  I, L, i = Dict{Node, Int}(), layers(r), 1
  for v ∈ L
    for n ∈ v I[n] = i; i += 1 end
  end
  V = Vector{Float64}(undef, sum(length.(L)))
  return sum(plogpdf!(I, V, L, view(X, i, :)) for i ∈ 1:size(X, 1))
end
export plogpdf

"""
    sample(weights)::UInt

Sample integer with probability proportional to given `weights`.

# Parameters

  - `weights`: vector of nonnegative reals.
"""
function sample(weights::AbstractVector{Float64})::UInt
  Z = sum(weights)
  u = rand()
  cum = 0.0
  for i in 1:length(weights)
    @inbounds cum += weights[i] / Z
    if u < cum
      return i
    end
  end
end

"""
    rand(n::Indicator)
    rand(n::Categorical)
    rand(n::Gaussian)

Sample values from circuit leaves.
"""
@inline Base.rand(n::Indicator) = n.value
@inline Base.rand(n::Categorical) = sample(n.values)
@inline Base.rand(n::Gaussian) = n.mean + sqrt(n.variance) * randn()

"""
    rand(r::Node)

Returns a sample of values of the variables generated according
to the probability defined by the network rooted at `r`. Stores the sample
as a vector of values
"""
function Base.rand(r::Node)
  if length(scope(r)) > 0
    numvars = length(scope(r))
  else
    numvars = length(union(n.scope for n in leaves(r)))
  end
  a = Vector{Float64}(undef, numvars)
  # sample induced tree
  queue = Node[r]
  while !isempty(queue)
    n = popfirst!(queue)
    if issum(n)
      # sample one child to visit
      c = sample(n.weights)
      push!(queue, n.children[c])
    elseif isprod(n)
      # visit every child
      append!(queue, n.children)
    else
      # draw value from node distribution
      a[n.scope] = rand(n)
    end
  end
  return a
end

"""
    rand(r::Node, N::Integer)

Returns a matrix of samples generated according to the probability
defined by the network `circ`.
"""
function Base.rand(r::Node, N::Integer)
  if length(scope(r)) > 0
    numvars = length(scope(r))
  else
    numvars = length(union(n.scope for n in leaves(r)))
  end
  Sample = Array{Float64}(undef, N, numvars)
  # get sample
  for i in 1:N
    queue = Node[r]
    while length(queue) > 0
      n = popfirst!(queue)
      if issum(n)
        # sample one child to visit
        c = sample(n.weights) # sparse array to vector inserts 0 at first coordinate
        push!(queue, n.children[c])
      elseif isprod(n)
        # visit every child
        append!(queue, n.children)
      else
        # draw value from distribution
        Sample[i, n.scope] = rand(n)
      end
    end
  end
  return Sample
end

"""
    ncircuits(r::Node)::Int

Counts the number of induced circuits of the circuit rooted at `r`.
"""
ncircuits(r::Node)::Int = ncircuits!(Dict{Node, Int}(), r)
export ncircuits

"""
    ncircuits!(values::Dict{Node, Int}, r::Node)::Int

Counts the number of induced circuits of the circuit rooted at `r`, caching intermediate values.
"""
function ncircuits!(values::Dict{Node, Int}, r::Node)
  # traverse nodes in reverse topological order (bottom-up)
  N = nodes(r; rev = false)
  for n ∈ N
    if isa(n, Product)
      values[n] = mapreduce(u -> values[u], *, n.children)
    elseif isa(n, Sum)
      values[n] = mapreduce(u -> values[u], +, n.children)
    else # is a leaf node
      values[n] = 1
    end
  end
  return values[r]
end

@inline avgll(r::Node, data::AbstractMatrix{<:Real}) = logpdf(r, data) / size(data, 1)
export avgll

"""
    NLL(r::Node, data::AbstractMatrix{<:Real})

Computes the average negative loglikelihood of a dataset `data` assigned by circ.
"""
NLL(r::Node, data::AbstractMatrix{<:Real}) = -logpdf(r, data) / size(data, 1)
"""Computes the NLL by parallelizing the circuit instead of instances."""
@inline function pNLL(V::Vector{Float64}, N::Vector{Node}, L::Vector{Vector{UInt}}, D::AbstractMatrix{<:Real})::Float64
  n = size(D, 1)
  return -sum(cplogpdf!(V, N, L, view(D, i, :)) for i in 1:n)/n
end

export NLL

"""
    MAE(S1::Node, S2::Node, data::AbstractMatrix{<:Real})

Computes the Mean Absolute Error of circuits `S1` and `S2` on given `data`.
"""
MAE(S1::Node, S2::Node, data::AbstractMatrix{<:Real}) =
  sum(abs(logpdf(S1, view(data, i, :)) - logpdf(S2, view(data, i, :))) for i in 1:size(data, 1)) /
  size(data, 1)
