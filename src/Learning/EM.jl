
# Parameter learning by Stochastic Mini-Batch Expectation-Maximimzation

"""
Learn weights using the Expectation Maximization algorithm.
"""
mutable struct SEM <: ParameterLearner
  root::Node
  N_r::Vector{Node}
  layers::Vector{Vector{Node}}
  sums_r::Vector{Sum}
  previous::Node # for performing temporary updates and applying momentum
  N_p::Vector{Node}
  layersp::Vector{Vector{Node}}
  sums_p::Vector{Sum}
  cmap::Dict{Node, Node} # mapping of each node in root to corresponding copy in previous
  imap::Dict{Node, Int} # mapping from node to vector index for root
  jmap::Dict{Node, Int} # mapping from node to vector index for previous
  diff::Vector{Float64} # to store derivatives
  values::Vector{Float64} # to store logprobabilities
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  function SEM(r::Node)
    p, M = mapcopy(r; converse = true)
    N_r, N_p = nodes(r), nodes(p)
    n = length(N_r)
    imap, jmap = Dict{Node, Int}(x => i for (i, x) ∈ enumerate(N_r)), Dict{Node, Int}(x => i for (i, x) ∈ enumerate(N_p))
    sums_r, sums_p = filter(x -> isa(x, Sum), N_r), filter(x -> isa(x, Sum), N_p)
    return new(r, N_r, layers(r), sums_r, p, N_p, layers(p), sums_p, M, imap, jmap,
               Vector{Float64}(undef, n), Vector{Float64}(undef, n), NaN, NaN, 1e-4, 0, 0.5)
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
  minimumvariance::Float64 = learner.minimumvariance
)

  numrows, numcols = size(Data)

  root_p = learner.root # current parameters
  root_n = learner.previous # updated parameters
  imap = learner.imap
  M = learner.cmap
  # # @assert numcols == circ._numvars "Number of columns should match number of variables in network."
  score = 0.0 # data loglikelihood
  sumnodes = learner.sums_r
  if learngaussians
    gaussiannodes = nodes(root_p; f = Base.Fix2(isa, Gaussian), rev = false)
    if length(gaussiannodes) > 0
        means = Dict{Node, Float64}(n => 0.0 for n in gaussiannodes)
        squares = Dict{Node, Float64}(n => 0.0 for n in gaussiannodes)
        denon = Dict{Node, Float64}(n => 0.0 for n in gaussiannodes)
    end
  end
  #diff = zeros(Float64, length(circ))
  diff = learner.diff
  values = learner.values
  #values = similar(diff)
  # TODO beter exploit multithreading
  # Compute expected weights
  for t in 1:numrows
    datum = view(Data, t, :)
    #lv = logpdf!(values,circ,datum) # propagate input Data[i,:]
    lv = plogpdf!(imap, values, learner.layers, datum) # parallelized version
    @assert isfinite(lv) "logvalue of datum $t is not finite: $lv"
    score += lv
    #TODO: implement multithreaded version of backpropagate
    backpropagate!(imap, diff, learner.N_r, values) # backpropagate derivatives
    Threads.@threads for l in 1:length(sumnodes) # update each node in parallel
      n = sumnodes[l]
      i = imap[n]
      p = learner.sums_p[l]
      @inbounds for (j, c) in enumerate(n.children)
        # @assert isfinite(diff[i]) "derivative of node $i is not finite: $(diff[i])"
        # @assert !isnan(values[j]) "value of node $j is NaN: $(values[j])"
        u = values[imap[c]]
        if isfinite(u)
          δ = n.weights[j] * diff[i] * exp(u - lv) # improvement
          # @assert isfinite(δ) "improvement to weight ($i,$j):$(circ_p[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
          if !isfinite(δ) δ = 0.0 end
        else
          δ = 0.0
        end
        p.weights[j] = ((t - 1) / t) * p.weights[j] + δ / t # running average for improved precision
      end
    end
    if learngaussians
      Threads.@threads for n in gaussiannodes
        i = imap[n]
        α = diff[i]*exp(values[i]-lv)
        u = datum[n.scope]
        denon[n] += α
        means[n] += α*u
        squares[n] += α*u*u
      end
    end
  end
  # Do update
  # TODO: implement momentum acceleration (nesterov acceleration, Adam, etc)
  # newweights =  log.(newweights) .+ maxweights
  Threads.@threads for i in 1:length(sumnodes)
    n = sumnodes[i]
    p = learner.sums_p[i]
    p.weights .+= smoothing / length(p.weights) # smoothing factor to prevent degenerate probabilities
    p.weights .*= learningrate / sum(p.weights) # normalize weights
    # online update: θ[t+1] = (1-η)*θ[t] + η*update(θ[t])
    p.weights .+= (1.0 - learningrate) * n.weights
    p.weights ./= sum(p.weights)
    # @assert sum(circ_n[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(circ_n[i].weights)) | $(circ_n[i].weights) | $(circ_p[i].weights)"
  end
  if learngaussians
    Threads.@threads for n in gaussiannodes
      p = M[n]
      # online update: θ[t+1] = (1-η)*θ[t] + η*update(θ[t])
      @inbounds p.mean = learningrate*means[n]/denon[n] + (1-learningrate)*n.mean
      @inbounds p.variance = learningrate*(squares[n]/denon[n] - (p.mean)^2) + (1-learningrate)*n.variance
      if p.variance < minimumvariance p.variance = minimumvariance end
    end
  end
  learner.previous = root_p
  learner.root = root_n
  learner.layers, learner.layersp = learner.layersp, learner.layers
  learner.sums_r, learner.sums_p = learner.sums_p, learner.sums_r
  learner.imap, learner.jmap = learner.jmap, learner.imap
  learner.N_r, learner.N_p = learner.N_p, learner.N_r
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -score / numrows
  return learner.prevscore - learner.score
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
