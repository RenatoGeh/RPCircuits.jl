
# Parameter learning by Stochastic Mini-Batch Expectation-Maximimzation

"""
Learn weights using the Expectation Maximization algorithm.
"""
mutable struct SEM <: ParameterLearner
  circ::CCircuit
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  function SEM(r::Node; gauss::Bool = false)
    return new(compile(CCircuit, r; gauss), NaN, NaN, 1e-4, 0, 0.5)
  end
end
export SEM

"""
Verifies if algorithm has converged.

Convergence is defined as absolute difference of score.
Requires at least 2 steps of optimization.
"""
converged(learner::SEM) =
  learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance
export converged

# TODO: take filename os CSV File object as input and iterate over file to decrease memory footprint
"""
Improvement step for learning weights using the Expectation Maximization algorithm. Returns improvement in negative loglikelihood.

circ[i].weights[j] = circ[i].weights[k] * backpropagate(circ)[i]/sum(circ[i].weights[k] * backpropagate(circ)[i] for k=1:length(circ[i].children))

## Arguments

  - `learner`: EMParamLearner struct
  - `data`: Data Matrix
  - `learningrate`: learning inertia, used to avoid forgetting in minibatch learning [default: 1.0 correspond to no inertia, i.e., previous weights are discarded after update]
  - `smoothing`: weight smoothing factor (= pseudo expected count) [default: 1e-4]
  - `learngaussians`: whether to also update the parameters of Gaussian leaves [default: false]
  - `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(
  learner::SEM,
  Data::AbstractMatrix,
  learningrate::Float64 = 1.0,
  smoothing::Float64 = 1e-4,
  learngaussians::Bool = false,
  minimumvariance::Float64 = learner.minimumvariance;
  verbose::Bool = false, validation::AbstractMatrix = Data, history = nothing
)

  numrows, numcols = size(Data)

  curr = learner.circ.C
  prev = learner.circ.P
  score = 0.0
  sumnodes = learner.circ.S
  if learngaussians
    gaussiannodes = learner.circ.gauss.G
  end

  V = Matrix{Float64}(undef, numrows, length(curr))
  Δ = Matrix{Float64}(undef, numrows, length(curr))

  # Compute backward pass (values)
  LL = mlogpdf!(V, curr, Data)
  # Compute forward pass (derivatives)
  backpropagate_tree!(Δ, curr, V)
  if !isnothing(findfirst(isnan, LL)) || !isnothing(findfirst(isnan, Δ)) return -2, V, Δ end

  # Update sum weights
  Threads.@threads for i ∈ 1:length(sumnodes)
    s = sumnodes[i]
    S, Z = curr[s], prev[s]
    β = Vector{Float64}(undef, length(S.children))
    for (j, c) ∈ enumerate(S.children)
      β[j] = log(S.weights[j]) + logsumexp(view(Δ, :, s) .+ view(V, :, c) .- LL)
    end
    u = exp.(β .- logsumexp(β))# .+ smoothing
    # u ./= sum(u)
    # u .= learningrate .* u .+ (1.0-learningrate) .* Z.weights
    S.weights .= u# ./ sum(u)
  end

  # Update Gaussian parameters
  if learngaussians
    mean_u, var_u = Vector{Float64}(undef, length(gaussiannodes)), Vector{Float64}(undef, length(gaussiannodes))
    Threads.@threads for i ∈ 1:length(gaussiannodes)
      g = gaussiannodes[i]
      G, H = curr[g], prev[g]
      α = view(Δ, :, g) .+ view(V, :, g) .- LL
      α .= exp.(α .- logsumexp(α))
      X = view(Data, :, G.scope)
      mean_u[i], var_u[i] = learningrate*sum(α .* X) + (1-learningrate)*H.mean, learningrate*sum(α .* ((X .- G.mean) .^ 2)) + (1-learningrate)*H.variance
      # G.mean = learningrate*sum(α .* X) + (1-learningrate)*H.mean
      # G.variance = learningrate*sum(α .* ((X .- G.mean) .^ 2)) + (1-learningrate)*H.variance
      # !(G.variance > minimumvariance) && (G.variance = minimumvariance)
    end
    if !isnothing(findfirst(isnan, mean_u)) || !isnothing(findfirst(isnan, var_u)) return -1, V, Δ end
    for i ∈ 1:length(gaussiannodes)
      G = curr[gaussiannodes[i]]
      G.mean, G.variance = mean_u[i], var_u[i]
      !(G.variance > minimumvariance) && (G.variance = minimumvariance)
    end
  end

  if verbose && (learner.steps % 2 == 0)
    print("Training LL: ", sum(LL)/numrows, " | ")
    ll = pNLL(LL, curr, learner.circ.L, validation)
    if !isnothing(history) push!(history, ll) end
    println("Iteration $(learner.steps). η: $(learningrate), NLL: $(ll)")
  end

  score = sum(LL)
  swap!(learner.circ)
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -score / numrows

  return learner.prevscore - learner.score, V, Δ
end
export update

# Parameter learning by Accelerated Expectation-Maximimzation (SQUAREM)
# RAVI VARADHAN & CHRISTOPHE ROLAND, Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm, J. Scand J Statist 2008
# Yu Du & Ravi Varadhan, SQUAREM: An R Package for Off-the-Shelf Acceleration of EM, MM and Other EM-Like Monotone Algorithms, J. Statistical Software, 2020.
"""
Learn weights using the Accelerated Expectation Maximization algorithm.
"""
mutable struct SQUAREM <: ParameterLearner
  root::Node
  layers::Vector{Vector{Node}}
  layersp::Vector{Vector{Node}}
  cache1::Node
  cache2::Node
  cache3::Node
  cache4::Node
  cache1_map::Dict{Node, Node}
  cache2_map::Dict{Node, Node}
  cache3_map::Dict{Node, Node}
  cache4_map::Dict{Node, Node}
  diff::Dict{Node, Float64} # to store derivatives
  values::Dict{Node, Float64} # to store logprobabilities
  # dataset::AbstractMatrix
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  function SQUAREM(r::Node)
    a, amap = mapcopy(r)
    b, bmap = mapcopy(r)
    c, cmap = mapcopy(r)
    d, dmap = mapcopy(r)
    return new(r, layers(r), layers(a), a, b, c, d, amap, bmap, cmap, dmap, Dict{Node, Float64}(),
               Dict{Node, Float64}(), NaN, NaN, 1e-3, 0, 0.5)
  end
end
export SQUAREM

"""
Verifies if algorithm has converged.

Convergence is defined as absolute difference of score.
Requires at least 2 steps of optimization.
"""
converged(learner::SQUAREM) =
  learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance
export converged

# TODO: take filename os CSV File object as input and iterate over file to decrease memory footprint
"""
Improvement step for learning weights using the Squared Iterative Expectation Maximization algorithm. Returns improvement in negative loglikelihood.

circ[i].weights[j] = circ[i].weights[k] * backpropagate(circ)[i]/sum(circ[i].weights[k] * backpropagate(circ)[i] for k=1:length(circ[i].children))

## Arguments

  - `learner`: SQUAREM struct
  - `data`: Data Matrix
  - `smoothing`: weight smoothing factor (= pseudo expected count) [default: 0.0001]
  - `learngaussians`: whether to also update the parameters of Gaussian leaves [default: false]
  - `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(
  learner::SQUAREM,
  Data::AbstractMatrix,
  smoothing::Float64 = 0.0001,
  learngaussians::Bool = false, # not implemented
  minimumvariance::Float64 = learner.minimumvariance,
)

  numrows, numcols = size(Data)

  θ_0 = learner.circ
  θ_1 = learner.cache1
  θ_2 = learner.cache2
  r = learner.cache3
  v = learner.cache4
  μ_1 = learner.cache1_map
  μ_2 = learner.cache2_map
  μ_r = learner.cache3_map
  μ_v = learner.cache4_map
  sumnodes = sums(θ_0)
  # if learngaussians
  #   gaussiannodes = filter(i -> isa(circ_p[i], Gaussian), 1:length(circ_p))
  #   if length(gaussiannodes) > 0
  #       means = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
  #       squares = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
  #       denon = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
  #   end
  # end
  diff = learner.diff
  values = learner.values
  # Compute theta1 = EM_Update(theta0)
  for t in 1:numrows
    datum = view(Data, t, :)
    lv = plogpdf!(values, learner.layers, datum) # parallelized version
    @assert isfinite(lv) "1. logvalue of datum $t is not finite: $lv"
    backpropagate!(diff, θ_0, values) # backpropagate derivatives
    Threads.@threads for n in sumnodes # update each node in parallel
      @inbounds for (j, c) in enumerate(n.children)
        if isfinite(values[c])
          δ = n.weights[j] * diff[n] * exp(values[c] - lv) # improvement
          @assert isfinite(δ) "1. improvement to weight ($n,$c):$(n.weights[j]) is not finite: $δ, $(diff[n]), $(values[j]), $(exp(values[c]-lv))"
        else
          δ = 0.0
        end
        u = μ_1[n]
        u.weights[j] = ((t - 1) / t) * u.weights[j] + δ / t # running average for improved precision
        @assert u.weights[j] ≥ 0
      end
    end
    # if learngaussians
    #   Threads.@threads for i in gaussiannodes
    #       @inbounds α = diff[i]*exp(values[i]-lv)
    #       @inbounds denon[i] += α
    #       @inbounds means[i] += α*datum[θ_0[i].scope]
    #       @inbounds squares[i] += α*datum[θ_0[i].scope]^2
    #   end
    # end
  end
  @inbounds Threads.@threads for n in sumnodes
    # println(θ_1[i].weights)
    u = μ_1[n]
    u.weights .+= smoothing / length(u.weights) # smoothing factor to prevent degenerate probabilities
    # println("  ", θ_1[i].weights)
    u.weights ./= sum(u.weights)
    # println("    ", θ_1[i].weights)
    @assert sum(u.weights) ≈ 1.0 "1. Unnormalized weight vector at node $n: $(sum(u.weights)) | $(u.weights)"
  end
  # if learngaussians
  #   Threads.@threads for i in gaussiannodes
  #       @inbounds θ_1[i].mean = means[i]/denon[i]
  #       @inbounds θ_1[i].variance = squares[i]/denon[i] - (θ_1[i].mean)^2
  #       @inbounds if θ_1[i].variance < minimumvariance
  #           @inbounds θ_1[i].variance = minimumvariance
  #       end
  #       # reset values for next update
  #       means[i] = 0.0
  #       squares[i] = 0.0
  #       denon[i] = 0.0
  #   end
  # end  
  # Compute theta2 = EM_Update(theta1)
  for t in 1:numrows
    datum = view(Data, t, :)
    lv = plogpdf!(values, learner.layersp, datum) # parallelized version
    @assert isfinite(lv) "2. logvalue of datum $t is not finite: $lv"
    backpropagate!(diff, θ_1, values) # backpropagate derivatives
    Threads.@threads for n in sumnodes # update each node in parallel
      @inbounds for (j, c) in enumerate(n.children)
        if isfinite(values[c])
          u = μ_1[n]
          δ = u.weights[j] * diff[c] * exp(values[c] - lv) # improvement
          @assert isfinite(δ) "2. improvement to weight ($n,$c):$(u.weights[j]) is not finite: $δ, $(diff[n]), $(values[c]), $(exp(values[c]-lv))"
        else δ = 0.0 end
        u = μ_2[n]
        u.weights[j] = ((t - 1) / t) * u.weights[j] + δ / t
        @assert u.weights[j] ≥ 0
      end
    end
    # if learngaussians
    #   Threads.@threads for i in gaussiannodes
    #       @inbounds α = diff[i]*exp(values[i]-lv)
    #       @inbounds denon[i] += α
    #       @inbounds means[i] += α*datum[θ_0[i].scope]
    #       @inbounds squares[i] += α*datum[θ_0[i].scope]^2
    #   end
    # end
  end
  @inbounds Threads.@threads for n in sumnodes
    # println(θ_2[i].weights)
    u = μ_2[n]
    u.weights .+= smoothing / length(u.weights) # smoothing factor to prevent degenerate probabilities
    # println("  ", θ_2[i].weights)
    u.weights ./= sum(u.weights)
    # println("    ", θ_2[i].weights)
    @assert sum(u.weights) ≈ 1.0 "2. Unnormalized weight vector at node $n: $(sum(u.weights)) | $(u.weights)"
  end
  # if learngaussians
  #   Threads.@threads for i in gaussiannodes
  #       @inbounds θ_2[i].mean = means[i]/denon[i]
  #       @inbounds θ_2[i].variance = squares[i]/denon[i] - (θ_2[i].mean)^2
  #       @inbounds if θ_2[i].variance < minimumvariance
  #           @inbounds θ_2[i].variance = minimumvariance
  #       end
  #   end  
  # Compute r, v, |r| and |v|
  r_norm, v_norm = 0.0, 0.0
  @inbounds Threads.@threads for n in sumnodes
    # r[i].weights .= θ_1[i].weights .- θ_0[i].weights
    # v[i].weights .= θ_2[i].weights .- θ_1[i].weights .- r[i].weights
    p, q = μ_r[n], μ_v[n]
    a, b = μ_1[n], μ_2[n]
    c = μ_0[n]
    for k in 1:length(u.weights)
      p.weights[k] = a.weights[k] - c.weights[k]
      q.weights[k] = b.weights[k] - a.weights[k] - p.weights[k]
      r_norm += p.weights[k] * p.weights[k]
      v_norm += q.weights[k] * q.weights[k]
    end
    # r_norm += sum(r[i].weights .* r[i].weights)
    # v_norm += sum(v[i].weights .* v[i].weights)
  end
  # steplength
  α = -max(sqrt(r_norm) / sqrt(v_norm), 1)
  #println("α: $α")
  # Compute θ' (reuse θ_1 for that matter)
  @inbounds Threads.@threads for n in sumnodes
    # θ' = θ0 - 2αr + α^2v
    p, q = μ_1[n], μ_0[n]
    a, b = μ_r[n], μ_v[n]
    p.weights .= q.weights
    p.weights .-= ((2 * α) .* a.weights)
    p.weights .+= ((α * α) .* b.weights)
    p.weights .+ smoothing / length(p.weights) # add term to prevent negative weights due to numerical imprecision
    p.weights ./= sum(p.weights)
    @assert sum(p.weights) ≈ 1.0 "3. Unnormalized weight vector at node $n: $(sum(p.weights)) | $(p.weights)"
    for w in p.weights @assert w ≥ 0 "Negative weight at node $n: $(p.weights)" end
  end
  # Final EM Update: θ_0 = EM_Update(θ')
  score = 0.0 # data loglikelihood
  for t in 1:numrows
    datum = view(Data, t, :)
    lv = plogpdf!(values, learner.layersp, datum) # parallelized version
    @assert isfinite(lv) "4. logvalue of datum $t is not finite: $lv"
    score += lv
    backpropagate!(diff, θ_1, values) # backpropagate derivatives
    Threads.@threads for n in sumnodes # update each node in parallel
      @inbounds for (j, c) in enumerate(n.children)
        if isfinite(values[c])
          u = μ_1[n]
          δ = u.weights[j] * diff[n] * exp(values[n] - lv) # improvement
          @assert isfinite(δ) "4. improvement to weight ($n,$c):$(u.weights[j]) is not finite: $δ, $(diff[n]), $(values[n]), $(exp(values[n]-lv))"
        else
          δ = 0.0
        end
        n.weights[j] = ((t - 1) / t) * n.weights[j] + δ / t
        @assert n.weights[j] ≥ 0
      end
    end
  end
  @inbounds Threads.@threads for n in sumnodes
    n.weights .+= smoothing / length(n.weights) # smoothing factor to prevent degenerate probabilities
    n.weights ./= sum(n.weights)
    @assert sum(n.weights) ≈ 1.0 "4. Unnormalized weight vector at node $n: $(sum(n.weights)) | $(n.weights)"
  end
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -score / numrows
  return learner.prevscore - learner.score, α
end
export update
