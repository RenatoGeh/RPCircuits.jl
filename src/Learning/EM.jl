
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
  # dataset::AbstractMatrix
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

# TODO: take filename os CSV File object as input and iterate over file to decreae memory footprint
"""
Improvement step for learning weights using the Expectation Maximization algorithm. Returns improvement in negative loglikelihood.

circ[i].weights[j] = circ[i].weights[k] * backpropagate(circ)[i]/sum(circ[i].weights[k] * backpropagate(circ)[i] for k=1:length(circ[i].children))

## Arguments

  - `learner`: EMParamLearner struct
  - `data`: Data Matrix
  - `learningrate`: learning inertia, used to avoid forgetting in minibatch learning [default: 1.0 correspond to no inertia, i.e., previous weights are discarded after update]
  - `smoothing`: weight smoothing factor (= pseudo expected count) [default: 1e-4]
  - `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(
  learner::SEM,
  Data::AbstractMatrix,
  learningrate::Float64 = 1.0,
  smoothing::Float64 = 1e-4,
  minimumvariance::Float64 = learner.minimumvariance,
)

  numrows, numcols = size(Data)

  circ_p = learner.circ      # current parameters
  circ_n = learner.previous # updated parameters

  # # @assert numcols == circ._numvars "Number of columns should match number of variables in network."
  # m, n = size(circ._weights)
  # weights = nonzeros(circ._weights)
  # childrens = rowvals(circ._weights)
  # # oldweights = fill(τ, length(weights))
  # newweights = fill(τ, length(weights))
  # maxweights = similar(newweights)
  score = 0.0 # data loglikelihood
  sumnodes = filter(i -> isa(circ_p[i], Sum), 1:length(circ_p))
  # gaussiannodes = filter(i -> isa(circ[i],GaussianDistribution), 1:length(circ))
  # if length(gaussiannodes) > 0
  #     means = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
  #     squares = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
  #     denon = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
  # end
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
          @assert isfinite(δ) "improvement to weight ($i,$j):$(circ_p[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
        else
          δ = 0.0
        end
        circ_n[i].weights[k] = ((t - 1) / t) * circ_n[i].weights[k] + δ / t # running average for improved precision
      end
      #         # for j in children(circ,i) #circ._backward[i]
      #         #     oldweights[j,i] += circ._weights[j,i]*diff[i]*exp(values[j]-lv)
      #         # end
      #         for k in nzrange(circ._weights,i)
      #             j = childrens[k]
      #             #oldweights[k] += weights[k]*diff[i]*exp(values[j]-lv)
      #             Δ = log(weights[k]) + log(diff[i]) + values[j] - lv
      #             if t == 1
      #                 maxweights[k] = Δ
      #                 newweights[k] = 1.0
      #             else
      #                 if Δ > maxweights[k]
      #                     newweights[k] = exp(log(newweights[k])+maxweights[k]-Δ)+1.0
      #                     maxweights[k] = Δ
      #                 elseif isfinite(Δ) && isfinite(maxweights[k])
      #                     newweights[k] += exp(Δ-maxweights[k])
      #                 end
      #                 @assert isfinite(newweights[k]) "Infinite weight: $(newweights[k])"
      #             end
      #         end
    end
    #     for i in gaussiannodes
    #         α = diff[i]*exp(values[i]-lv)
    #         denon[i] += α
    #         means[i] += α*datum[circ[i].scope]
    #         squares[i] += α*datum[circ[i].scope]^2
    #     end
  end
  # # add regularizer to avoid degenerate distributions
  # Do update
  # TODO: implement momentum acceleration (nesterov acceleration, Adam, etc)
  # newweights =  log.(newweights) .+ maxweights
  @inbounds Threads.@threads for i in sumnodes
    circ_n[i].weights .+= smoothing / length(circ_n[i].weights) # smoothing factor to prevent degenerate probabilities
    circ_n[i].weights .*= learningrate / sum(circ_n[i].weights) # normalize weights
    # online update: θ[t+1] = (1-η)*θ[t] + η*update(θ[t])
    circ_n[i].weights .+= (1.0 - learningrate) * circ_p[i].weights
    # cache[i].weights .*= learningrate/sum(cache[i].weights) # normalize weights
    # circ[i].weights .*= 1.0-learningrate # apply update with inertia strenght given by learning rate
    # @assert sum(circ[i].weights) ≈ 1.0-learningrate "Unnormalized weight vector at node $i: $(sum(circ[i].weights)) | $(circ[i].weights)"
    # circ[i].weights .+= cache[i].weights
    @assert sum(circ_n[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(circ_n[i].weights)) | $(circ_n[i].weights) | $(circ_p[i].weights)"
    # for (k,j) in enumerate(circ[i].children)
    #     # circ[i].weights .*= cache[i].weights
    # Z = sum(circ[i].weights)
    # circ[i].weights ./= Z
    # normexp!(circ[i].weights, prev[i].weights, τ) # if weights are in log
    # end
    #     chval = nzrange(circ._weights,i)
    #     normexp!(view(newweights,chval), view(weights,chval), τ)
    #     if !isfinite(sum(weights[chval])) # some numerical problem occurred, set weights to uniform
    #         weights[chval] .= 1/length(chval)
    #     end
    #     # @assert sum(weights[chval]) ≈ 1.0 "Unormalized weight vector: $(sum(weights[chval])) | $(weights[chval])"
    #     # add regularizer to avoid degenerate distributions
    #     # Z = sum(oldweights[:,i]) + τ*length(children(circ,i))
    #     # for j in  children(circ,i) #circ._backward[i]
    #     #    circ._weights[j,i] = (oldweights[j,i]+τ)/Z
    #     #    # if isnan(Z)
    #     #    #    @warn "Not a number for weight $i -> $j"
    #     #    # end
    #     # end
  end
  #println(circ)

  # for i in gaussiannodes
  #     circ[i].mean = means[i]/denon[i]
  #     circ[i].variance = squares[i]/denon[i] - (circ[i].mean)^2
  #     if circ[i].variance < minimumvariance
  #         circ[i].variance = minimumvariance
  #     end
  # end
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
Learn weights using the Expectation Maximization algorithm.
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

# TODO: take filename os CSV File object as input and iterate over file to decreae memory footprint
"""
Improvement step for learning weights using the Squared Iterative Expectation Maximization algorithm. Returns improvement in negative loglikelihood.

circ[i].weights[j] = circ[i].weights[k] * backpropagate(circ)[i]/sum(circ[i].weights[k] * backpropagate(circ)[i] for k=1:length(circ[i].children))

## Arguments

  - `learner`: SQUAREM struct
  - `data`: Data Matrix
  - `smoothing`: weight smoothing factor (= pseudo expected count) [default: 0.0001]
  - `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(
  learner::SQUAREM,
  Data::AbstractMatrix,
  smoothing::Float64 = 0.0001,
  minimumvariance::Float64 = learner.minimumvariance,
)

  numrows, numcols = size(Data)

  θ_0 = learner.circ
  θ_1 = learner.cache1
  θ_2 = learner.cache2
  r = learner.cache3
  v = learner.cache4
  # smooth out estaimtors to avoid degenerate probabilities
  sumnodes = filter(i -> isa(learner.circ[i], Sum), 1:length(learner.circ))
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
  end
  @inbounds Threads.@threads for i in sumnodes
    θ_1[i].weights .+= smoothing / length(θ_1[i].weights) # smoothing factor to prevent degenerate probabilities
    θ_1[i].weights ./= sum(θ_1[i].weights)
    @assert sum(θ_1[i].weights) ≈ 1.0 "1. Unnormalized weight vector at node $i: $(sum(θ_1[i].weights)) | $(θ_1[i].weights)"
  end
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
  end
  @inbounds Threads.@threads for i in sumnodes
    θ_2[i].weights .+= smoothing / length(θ_2[i].weights) # smoothing factor to prevent degenerate probabilities
    θ_2[i].weights ./= sum(θ_2[i].weights)
    @assert sum(θ_2[i].weights) ≈ 1.0 "2. Unnormalized weight vector at node $i: $(sum(θ_2[i].weights)) | $(θ_2[i].weights)"
  end
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
