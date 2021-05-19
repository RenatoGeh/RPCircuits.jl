# Routines for SPN evaluation and sampling

"""
(circ::Circuit)(x::AbstractVector{<:Real})
(circ::Circuit)(x...)

Evaluates the circuit at a given instantiation `x` of the variables.
Summed-out variables are represented as `NaN`s.

# Parameters

  - `x`: vector of values of variables (integers or reals).

# Examples

To compute the probability of `P(b=1)` using circ `S`, use

```@example
S = Circuit(IOBuffer("1 + 2 0.2 3 0.5 4 0.3\n2 * 5 7\n3 * 5 8\n4 * 6 8\n5 categorical 1 0.6 0.4\n6 categorical 1 0.1 0.9\n7 categorical 2 0.3 0.7\n8 categorical 2 0.8 0.2"));
println(S([NaN, 2]))
println(S(NaN, 2))
```
"""
(circ::Circuit)(x::Example) = exp(logpdf(circ, x))
(circ::Circuit)(x...) = exp(logpdf(circ, [x...]))

"""
logpdf(circ, X)

Returns the sums of the log-probabilities of instances `x` in `X`. Uses multithreading to speed up
computations if `JULIA_NUM_THREADS > 1`.

# Parameters

  - `circ`: Sum-Product Network
  - `X`: matrix of values of variables (integers or reals). Summed-out variables are represented as
    `NaN`s.
"""
function logpdf(circ::Circuit, X::Data)::Float64
  # single-threaded version
  if Threads.nthreads() == 1
    values = Array{Float64}(undef, length(circ))
    return sum(logpdf!(values, circ, view(X, i, :)) for i in 1:size(X, 1))
  end
  # multi-threaded version
  values = Array{Float64}(undef, length(circ), Threads.nthreads())
  s = Threads.Atomic{Float64}(0.0)
  Threads.@threads for i in 1:size(X, 1)
    Threads.atomic_add!(s, logpdf!(view(values, :, Threads.threadid()), circ, view(X, i, :)))
  end
  return s[]
end
export logpdf

"""
logpdf!(values, circ, X)

Computes log-probabilities of instances `x` in `X` and stores the results in a given vector. Uses
multithreading to speed up computations.

# Parameters

  - `results`: vector to store results (log-probabilities)
  - `circ`: Sum-Product Network
  - `X`: matrix of values of variables (integers or reals). Summed-out variables are represented as `NaN`s.
"""
function logpdf!(results::AbstractVector{<:Real}, circ::Circuit, X::AbstractMatrix{<:Real})
  @assert length(results) == size(X, 1)
  # multi-threaded version
  values = Array{Float64}(undef, length(circ), Threads.nthreads())
  Threads.@threads for i in 1:size(X, 1)
    @inbounds results[i] = logpdf!(view(values, :, Threads.threadid()), circ, view(X, i, :))
  end
  return Nothing
end
export logpdf!

"""
logpdf(circ, x)

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
function logpdf(circ::Circuit, x::AbstractVector{<:Real})::Float64
  values = Array{Float64}(undef, length(circ))
  return logpdf!(values, circ, x)
end

"""
logpdf!(values,circ,x)

Evaluates the circuit `circ` in log domain at configuration `x` and stores values of each node in the vector `values`.
"""
function logpdf!(values::AbstractVector{Float64}, circ::Circuit, x::AbstractVector{<:Real})::Float64
  # @assert length(values) == length(circ)
  # traverse nodes in reverse topological order (bottom-up)
  @inbounds for i in length(circ):-1:1
    node = circ[i]
    if isprod(node)
      lval = 0.0
      for j in node.children
        lval += values[j]
      end
      values[i] = isfinite(lval) ? lval : -Inf
    elseif issum(node)
      # log-sum-exp trick to improve numerical stability
      m = -Inf
      # get maximum incoming value
      for j in node.children
        m = values[j] > m ? values[j] : m
      end
      lval = 0.0
      for (k, j) in enumerate(node.children)
        # ensure exp in only computed on nonpositive arguments (avoid overflow)
        lval += exp(values[j] - m) * node.weights[k]
      end
      # println("m: ", m, ", lval: ", lval)
      # println("  values: ", values[node.children])
      # println("  weights: ", node.weights)
      # if something went wrong (e.g. incoming value is NaN or Inf) return -Inf
      values[i] = isfinite(lval) ? m + log(lval) : -Inf
    elseif isproj(node)
      values[i] = node.hyperplane(x) ? log(node.λ) + values[node.pos] : log(1-node.λ) + values[node.neg]
      if isnan(values[i])
        println("Node: ", i)
        println("  pos: ", node.pos, ", neg: ", node.neg)
        println("  pos_val: ", values[node.pos], ", neg_val: ", values[node.neg])
        println("  λ = ", node.λ, ", activated? ", node.hyperplane(x) ? "pos" : "neg")
      end
    else # is a leaf node
      values[i] = logpdf(node, x[node.scope])
    end
  end
  @inbounds return values[1]
end

"""
plogpdf!(values,circ,layers,x)

Evaluates the circuit `circ` in log domain at configuration `x` using the scheduling in `nlayers`
as obtained by the method `layers(circ)`. Stores values of each node in the vector `values`.

# Parameters

  - `values`: vector to cache node values
  - `circ`: the sum product network
  - `nlayers`: Vector of vector of node indices determinig the layers of the `circ`; each node in a layer is computed based on values of nodes in smaller layers.
  - `x`: Vector containing assignment
"""
function plogpdf!(
  values::AbstractVector{Float64},
  circ::Circuit,
  nlayers,
  x::AbstractVector{<:Real},
)::Float64
  # visit layers from last (leaves) to first (root)
  @inbounds for l in length(nlayers):-1:1
    # parallelize computations within layer
    Threads.@threads for i in nlayers[l]
      node = circ[i]
      if isprod(node)
        lval = 0.0
        for j in node.children
          lval += values[j]
        end
        values[i] = isfinite(lval) ? lval : -Inf
      elseif issum(node)
        # log-sum-exp trick to improve numerical stability (assumes weights are normalized)
        m = -Inf
        # get maximum incoming value
        for j in node.children
          m = values[j] > m ? values[j] : m
        end
        lval = 0.0
        for (k, j) in enumerate(node.children)
          # ensure exp in only computed on nonpositive arguments (avoid overflow)
          lval += exp(values[j] - m) * node.weights[k]
        end
        # if something went wrong (e.g. incoming value is NaN or Inf) return -Inf, otherwise
        # return maximum plus lval (adding m ensures signficant digits are numerically precise)
        values[i] = isfinite(lval) ? m + log(lval) : -Inf
      else # is a leaf node
        values[i] = logpdf(node, x[node.scope])
      end
    end
  end
  @inbounds return values[1]
end
export plogpdf!

"""
plogpdf(circ, x)

Parallelized version of `logpdf(circ, x)`. Set `JULIA_NUM_THREADS` > 1 to make this effective.
"""
function plogpdf(circ::Circuit, x::AbstractVector{<:Real})::Float64
  values = Array{Float64}(undef, length(circ))
  return plogpdf!(values, circ, layers(circ), x)
end
@inline function plogpdf(circ::Circuit, X::AbstractMatrix{<:Real})::Float64
  s = 0.0
  @inbounds for i ∈ 1:size(X)[1]
    s += plogpdf(circ, view(X, i, :))
  end
  return s
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
rand(circ)

Returns a sample of values of the variables generated according
to the probability defined by the network `circ`. Stores the sample
as a vector of values
"""
function Base.rand(circ::Circuit)
  if length(scope(circ)) > 0
    numvars = length(scope(circ))
  else
    numvars = length(union(n.scope for n in leaves(circ)))
  end
  a = Vector{Float64}(undef, numvars)
  # sample induced tree
  queue = [1]
  while !isempty(queue)
    n = popfirst!(queue)
    if issum(circ[n])
      # sample one child to visit
      c = sample(circ[n].weights)
      push!(queue, circ[n].children[c])
    elseif isprod(circ[n])
      # visit every child
      append!(queue, children(circ, n))
    else
      # draw value from node distribution
      a[circ[n].scope] = rand(circ[n])
    end
  end
  return a
end

"""
rand(circ::Circuit, N::Integer)

Returns a matrix of samples generated according to the probability
defined by the network `circ`.
"""
function Base.rand(circ::Circuit, N::Integer)
  if length(scope(circ)) > 0
    numvars = length(scope(circ))
  else
    numvars = length(union(n.scope for n in leaves(circ)))
  end
  Sample = Array{Float64}(undef, N, numvars)
  # get sample
  for i in 1:N
    queue = [1]
    while length(queue) > 0
      n = popfirst!(queue)
      if issum(circ[n])
        # sample one child to visit
        c = sample(circ[n].weights) # sparse array to vector inserts 0 at first coordinate
        push!(queue, circ[n].children[c])
      elseif isprod(circ[n])
        # visit every child
        append!(queue, children(circ, n))
      else
        # draw value from distribution
        Sample[i, circ[n].scope] = rand(circ[n])
      end
    end
  end
  return Sample
end

"""
ncircuits(circ::Circuit)

Counts the number of induced circuits of the circuit `circ`.
"""
ncircuits(circ::Circuit) = ncircuits!(Array{Int}(undef, length(circ)), circ)
export ncircuits

"""
ncircuits!(values,circ)

Counts the number of induced circuits of the circuit `circ`, caching intermediate values.
"""
function ncircuits!(values::AbstractVector{Int}, circ::Circuit)
  @assert length(values) == length(circ)
  # traverse nodes in reverse topological order (bottom-up)
  for i in length(circ):-1:1
    @inbounds node = circ[i]
    if isa(node, Product)
      @inbounds values[i] = mapreduce(j -> values[j], *, node.children)
    elseif isa(node, Sum)
      @inbounds values[i] = sum(values[node.children])
    else # is a leaf node
      @inbounds values[i] = 1
    end
  end
  @inbounds values[1]
end

"""
NLL(circ::Circuit,data::AbstractMatrix{<:Real})

Computes the average negative loglikelihood of a dataset `data` assigned by circ.
"""
NLL(circ::Circuit, data::AbstractMatrix{<:Real}) = -logpdf(circ, data) / size(data, 1)

"""
MAE(S1::Circuit,S2::Circuit,data::AbstractMatrix{<:Real})

Computes the Mean Absolute Error of circuits `S1` and `S2` on given `data`.
"""
MAE(S1::Circuit, S2::Circuit, data::AbstractMatrix{<:Real}) =
  sum(abs(logpdf(S1, view(data, i, :)) - logpdf(S2, view(data, i, :))) for i in 1:size(data, 1)) /
  size(data, 1)
