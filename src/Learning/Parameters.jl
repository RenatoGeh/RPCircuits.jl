using Random

"""
    Parameter Learning Algorithm
"""
abstract type ParameterLearner end

"""
    backpropagate!(diff,circ,values)

Computes derivatives for values propagated in log domain and stores results in given vector diff.
"""
function backpropagate!(diff::Vector{Float64}, circ::Circuit, values::Vector{Float64})
  # Assumes network has been evaluted at some assignment using logpdf!
  # Backpropagate derivatives
  @assert length(diff) == length(circ) == length(values)
  fill!(diff, 0.0)
  @inbounds diff[1] = 1.0
  for i in 1:length(circ)
    @inbounds node = circ[i]
    if issum(node)
      @inbounds for (k, j) in enumerate(node.children)
        diff[j] += node.weights[k] * diff[i]
      end
    elseif isprod(node)
      @inbounds for j in node.children
        if isfinite(values[j])
          # @assert isfinite(exp(values[i]-values[j]))  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(exp(values[i]-values[j]))"
          diff[j] += diff[i] * exp(values[i] - values[j])
        else
          δ = exp(sum(values[k] for k in node.children if k ≠ j))
          # @assert isfinite(δ)  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(δ)"
          diff[j] += diff[i] * δ
        end
      end
    end
  end
end

# compute log derivatives (not working!)
function logbackpropagate(circ::Circuit, values::Vector{Float64}, diff::Vector{Float64})
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
Computes derivatives for given vector of values propagated in log domain.

Returns vector of derivatives.
"""
function backpropagate(circ::Circuit, values::Vector{Float64})::Vector{Float64}
  diff = Array{Float64}(undef, length(circ))
  backpropagate!(diff, circ, values)
  return diff
end

"""
Random initialization of weights
"""
function initialize(learner::ParameterLearner) 
  circ = learner.circ
  sumnodes = filter(i -> isa(circ[i], Sum), 1:length(circ))
  for i in sumnodes
    #@inbounds ch = circ[i].children  # children(circ,i)
    @inbounds Random.rand!(circ[i].weights)
    @inbounds circ[i].weights ./= sum(circ[i].weights)
    @assert sum(circ[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(circ[i].weights)) | $(circ[i].weights)"
  end
  gaussiannodes = filter(i -> isa(circ[i], Gaussian), 1:length(circ))
  for i in gaussiannodes
    @inbounds circ[i].mean = rand()
    @inbounds circ[i].variance = 1.0
  end
end
export initialize
