using Random

"""
    Parameter Learning Algorithm
"""
abstract type ParameterLearner end

"""
    backpropagate!(diff::Dict{Node, Float64}, r::Node, values::Dict{Node, Float64})

Computes derivatives for values propagated in log domain and stores results in given vector diff.
"""
function backpropagate!(diff::Dict{Node, Float64}, r::Node, values::Dict{Node, Float64})
  # Assumes network has been evaluted at some assignment using logpdf!
  # Backpropagate derivatives
  diff[r] = 1.0
  function dx(i::Int, n::Node)
    if issum(n)
      for (j, c) in enumerate(n.children) diff[c] += node.weights[j] * diff[n] end
    elseif isprod(n)
      for c in node.children
        if isfinite(values[c])
          # @assert isfinite(exp(values[i]-values[j]))  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(exp(values[i]-values[j]))"
          diff[c] += diff[n] * exp(values[n] - values[c])
        else
          δ = exp(sum(values[k] for k in n.children if k ≠ j))
          # @assert isfinite(δ)  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(δ)"
          diff[c] += diff[n] * δ
        end
      end
    end
  end
  foreach(dx, r)
end

# compute log derivatives (not working!)
function logbackpropagate!(r::Node, values::Vector{Float64}, diff::Vector{Float64})
  # Assumes network has been evaluted at some assignment using logpdf!
  # Backpropagate derivatives
  @assert length(diff) == length(circ) == length(values)
  # create storage for each computed value (message from parent to child)
  from = []
  to = []
  for i in 1:length(circ)
    if isa(circ[i], Sum) || isa(circ[i], Product)
      for j in children(circ, i)
        push!(from, i)
        push!(to, j)
      end
    end
  end
  cache = sparse(to, from, ones(Float64, length(to)))
  #fill!(diff,0.0)
  logdiff = zeros(Float64, length(diff))
  for i in 1:length(circ)
    if i == 1
      diff[i] == 1.0
    else
      cache_vals = nonzeros(cache[i, :]) # incoming arc values
      offset = maximum(cache_vals)
      logdiff[i] = offset + log(sum(exp.(cache_vals .- offset)))
      diff[i] = isfinite(logdiff[i]) ? exp(logdiff[i]) : 0.0
    end
    if isa(circ[i], Sum)
      for j in children(circ, i)
        #@inbounds diff[j] += getweight(circ,i,j)*diff[i]
        cache[j, i] = logweight(circ, i, j) + logdiff[i]
      end
    elseif isa(circ[i], Product)
      for j in children(circ, i)
        #@inbounds diff[j] += diff[i]*exp(values[i]-values[j])
        cache[j, i] = logdiff[i] + values[i] - values[j]
      end
    end
  end
end

"""
    backpropagate(r::Node, values::Dict{Node, Float64})::Dict{Node, Float64}

Computes derivatives for given vector of values propagated in log domain.

Returns vector of derivatives.
"""
function backpropagate(r::Node, values::Dict{Node, Float64})::Dict{Node, Float64}
  diff = Dict{Node, Float64}()
  backpropagate!(diff, r, values)
  return diff
end

"""
Random initialization of weights
"""
function initialize(learner::ParameterLearner)
  r = learner.root
  function f(i::Int, n::Node)
    if issum(n)
      @inbounds Random.rand!(n.weights)
      @inbounds n.weights ./= sum(n.weights)
      @assert sum(n.weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(n.weights)) | $(n.weights)"
    elseif n isa Gaussian
      @inbounds n.mean = rand()
      @inbounds n.variance = 1.0
    end
  end
  foreach(f, r)
end
export initialize
