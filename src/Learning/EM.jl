
# Parameter learning by Stochastic Mini-Batch Expectation-Maximimzation

"""
Learn weights using the Expectation Maximization algorithm.
"""
mutable struct SEM <: ParameterLearner
  circ::Circuit
  layers::Vector{Vector{Int}}
  previous::Circuit # for performing temporary updates and applying momentum
  diff::Vector{Float64} # to store derivatives
  values::Vector{Float64} # to store logprobabilities
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  SEM(circ::Circuit) = new(
    circ,
    layers(circ),
    deepcopy(circ),
    Array{Float64}(undef, length(circ)),
    Array{Float64}(undef, length(circ)),
    NaN,
    NaN,
    1e-4,
    0,
    0.5,
  )
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

  circ_p = learner.circ      # current parameters
  circ_n = learner.previous # updated parameters
  # # @assert numcols == circ._numvars "Number of columns should match number of variables in network."
  score = 0.0 # data loglikelihood
  sumnodes = filter(i -> isa(circ_p[i], Sum), 1:length(circ_p))
  if learngaussians
    gaussiannodes = filter(i -> isa(circ_p[i], Gaussian), 1:length(circ_p))
    if length(gaussiannodes) > 0
        means = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
        squares = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
        denon = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
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
    lv = plogpdf!(values, circ_p, learner.layers, datum) # parallelized version
    @assert isfinite(lv) "logvalue of datum $t is not finite: $lv"
    score += lv
    #TODO: implement multithreaded version of backpropagate
    backpropagate!(diff, circ_p, values) # backpropagate derivatives
    Threads.@threads for i in sumnodes # update each node in parallel
      @inbounds for (k, j) in enumerate(circ_p[i].children)
        # @assert isfinite(diff[i]) "derivative of node $i is not finite: $(diff[i])"
        # @assert !isnan(values[j]) "value of node $j is NaN: $(values[j])"
        if isfinite(values[j])
          δ = circ_p[i].weights[k] * diff[i] * exp(values[j] - lv) # improvement
          # @assert isfinite(δ) "improvement to weight ($i,$j):$(circ_p[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
          if !isfinite(δ) δ = 0.0 end
        else
          δ = 0.0
        end
        circ_n[i].weights[k] = ((t - 1) / t) * circ_n[i].weights[k] + δ / t # running average for improved precision
      end
    end
    if learngaussians
      Threads.@threads for i in gaussiannodes
          @inbounds α = diff[i]*exp(values[i]-lv)
          @inbounds denon[i] += α
          @inbounds means[i] += α*datum[circ_p[i].scope]
          @inbounds squares[i] += α*(datum[circ_p[i].scope]^2)
      end
    end
  end
  # Do update
  # TODO: implement momentum acceleration (nesterov acceleration, Adam, etc)
  # newweights =  log.(newweights) .+ maxweights
  @inbounds Threads.@threads for i in sumnodes
    circ_n[i].weights .+= smoothing / length(circ_n[i].weights) # smoothing factor to prevent degenerate probabilities
    circ_n[i].weights .*= learningrate / sum(circ_n[i].weights) # normalize weights
    # online update: θ[t+1] = (1-η)*θ[t] + η*update(θ[t])
    circ_n[i].weights .+= (1.0 - learningrate) * circ_p[i].weights
    circ_n[i].weights ./= sum(circ_n[i].weights)
    # @assert sum(circ_n[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(circ_n[i].weights)) | $(circ_n[i].weights) | $(circ_p[i].weights)"
  end
  if learngaussians
    Threads.@threads for i in gaussiannodes
        # online update: θ[t+1] = (1-η)*θ[t] + η*update(θ[t])
        @inbounds circ_n[i].mean = learningrate*means[i]/denon[i] + (1-learningrate)*circ_p[i].mean
        @inbounds circ_n[i].variance = learningrate*(squares[i]/denon[i] - (circ_n[i].mean)^2) + (1-learningrate)*circ_p[i].variance
        @inbounds if circ_n[i].variance < minimumvariance
            @inbounds circ_n[i].variance = minimumvariance
        end
    end
  end
  learner.previous = circ_p
  learner.circ = circ_n
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
  circ::Circuit
  layers::Vector{Vector{Int}}
  cache1::Circuit
  cache2::Circuit
  cache3::Circuit
  cache4::Circuit
  diff::Vector{Float64} # to store derivatives
  values::Vector{Float64} # to store logprobabilities
  # dataset::AbstractMatrix
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  SQUAREM(circ::Circuit) = new(
    circ,
    layers(circ),
    deepcopy(circ),
    deepcopy(circ),
    deepcopy(circ),
    deepcopy(circ),
    Array{Float64}(undef, length(circ)),
    Array{Float64}(undef, length(circ)),
    NaN,
    NaN,
    1e-3,
    0,
    0.5,
  )
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
  sumnodes = filter(i -> isa(learner.circ[i], Sum), 1:length(learner.circ))
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
    lv = plogpdf!(values, θ_0, learner.layers, datum) # parallelized version
    @assert isfinite(lv) "1. logvalue of datum $t is not finite: $lv"
    backpropagate!(diff, θ_0, values) # backpropagate derivatives
    Threads.@threads for i in sumnodes # update each node in parallel
      @inbounds for (k, j) in enumerate(learner.circ[i].children)
        if isfinite(values[j])
          δ = θ_0[i].weights[k] * diff[i] * exp(values[j] - lv) # improvement
          @assert isfinite(δ) "1. improvement to weight ($i,$j):$(θ_0[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
        else
          δ = 0.0
        end
        θ_1[i].weights[k] = ((t - 1) / t) * θ_1[i].weights[k] + δ / t # running average for improved precision
        @assert θ_1[i].weights[k] ≥ 0
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
  @inbounds Threads.@threads for i in sumnodes
    # println(θ_1[i].weights)
    θ_1[i].weights .+= smoothing / length(θ_1[i].weights) # smoothing factor to prevent degenerate probabilities
    # println("  ", θ_1[i].weights)
    θ_1[i].weights ./= sum(θ_1[i].weights)
    # println("    ", θ_1[i].weights)
    @assert sum(θ_1[i].weights) ≈ 1.0 "1. Unnormalized weight vector at node $i: $(sum(θ_1[i].weights)) | $(θ_1[i].weights)"
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
    lv = plogpdf!(values, θ_1, learner.layers, datum) # parallelized version
    @assert isfinite(lv) "2. logvalue of datum $t is not finite: $lv"
    backpropagate!(diff, θ_1, values) # backpropagate derivatives
    Threads.@threads for i in sumnodes # update each node in parallel
      @inbounds for (k, j) in enumerate(learner.circ[i].children)
        if isfinite(values[j])
          δ = θ_1[i].weights[k] * diff[i] * exp(values[j] - lv) # improvement
          @assert isfinite(δ) "2. improvement to weight ($i,$j):$(θ_1[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
        else
          δ = 0.0
        end
        θ_2[i].weights[k] = ((t - 1) / t) * θ_2[i].weights[k] + δ / t
        @assert θ_2[i].weights[k] ≥ 0
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
  @inbounds Threads.@threads for i in sumnodes
    # println(θ_2[i].weights)
    θ_2[i].weights .+= smoothing / length(θ_2[i].weights) # smoothing factor to prevent degenerate probabilities
    # println("  ", θ_2[i].weights)
    θ_2[i].weights ./= sum(θ_2[i].weights)
    # println("    ", θ_2[i].weights)
    @assert sum(θ_2[i].weights) ≈ 1.0 "2. Unnormalized weight vector at node $i: $(sum(θ_2[i].weights)) | $(θ_2[i].weights)"
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
  @inbounds Threads.@threads for i in sumnodes
    # r[i].weights .= θ_1[i].weights .- θ_0[i].weights
    # v[i].weights .= θ_2[i].weights .- θ_1[i].weights .- r[i].weights
    for k in 1:length(r[i].weights)
      r[i].weights[k] = θ_1[i].weights[k] - θ_0[i].weights[k]
      v[i].weights[k] = θ_2[i].weights[k] - θ_1[i].weights[k] - r[i].weights[k]
      r_norm += r[i].weights[k] * r[i].weights[k]
      v_norm += v[i].weights[k] * v[i].weights[k]
    end
    # r_norm += sum(r[i].weights .* r[i].weights)
    # v_norm += sum(v[i].weights .* v[i].weights)
  end
  # steplength
  α = -max(sqrt(r_norm) / sqrt(v_norm), 1)
  #println("α: $α")
  # Compute θ' (reuse θ_1 for that matter)
  @inbounds Threads.@threads for i in sumnodes
    # θ' = θ0 - 2αr + α^2v
    θ_1[i].weights .= θ_0[i].weights
    θ_1[i].weights .-= ((2 * α) .* r[i].weights)
    θ_1[i].weights .+= ((α * α) .* v[i].weights)
    θ_1[i].weights .+ smoothing / length(θ_1[i].weights) # add term to prevent negative weights due to numerical imprecision
    θ_1[i].weights ./= sum(θ_1[i].weights)
    @assert sum(θ_1[i].weights) ≈ 1.0 "3. Unnormalized weight vector at node $i: $(sum(θ_1[i].weights)) | $(θ_1[i].weights)"
    for w in θ_1[i].weights
      @assert w ≥ 0 "Negative weight at node $i: $(θ_1[i].weights)"
    end
  end
  # Final EM Update: θ_0 = EM_Update(θ')
  score = 0.0 # data loglikelihood
  for t in 1:numrows
    datum = view(Data, t, :)
    lv = plogpdf!(values, θ_1, learner.layers, datum) # parallelized version
    @assert isfinite(lv) "4. logvalue of datum $t is not finite: $lv"
    score += lv
    backpropagate!(diff, θ_1, values) # backpropagate derivatives
    Threads.@threads for i in sumnodes # update each node in parallel
      @inbounds for (k, j) in enumerate(learner.circ[i].children)
        if isfinite(values[j])
          δ = θ_1[i].weights[k] * diff[i] * exp(values[j] - lv) # improvement
          @assert isfinite(δ) "4. improvement to weight ($i,$j):$(θ_1[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
        else
          δ = 0.0
        end
        θ_0[i].weights[k] = ((t - 1) / t) * θ_0[i].weights[k] + δ / t
        @assert θ_0[i].weights[k] ≥ 0
      end
    end
  end
  @inbounds Threads.@threads for i in sumnodes
    θ_0[i].weights .+= smoothing / length(θ_0[i].weights) # smoothing factor to prevent degenerate probabilities
    θ_0[i].weights ./= sum(θ_0[i].weights)
    @assert sum(θ_0[i].weights) ≈ 1.0 "4. Unnormalized weight vector at node $i: $(sum(θ_0[i].weights)) | $(θ_0[i].weights)"
  end
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -score / numrows
  return learner.prevscore - learner.score, α
end
export update
