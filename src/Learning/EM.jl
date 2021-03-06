
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
  verbose::Bool = false, validation::AbstractMatrix = Data, history = nothing, binary::Bool = false
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
  ?? = Matrix{Float64}(undef, numrows, length(curr))

  # Compute backward pass (values)
  LL = mplogpdf!(V, curr, Data)
  # Compute forward pass (derivatives)
  pbackpropagate_tree!(??, curr, V; indicators = learner.circ.I)

  # if any(isnan, V) || any(isnan, ??) println("break -1"); return -1, V, ?? end

  # Update sum weights
  log_n = log(numrows)
  Threads.@threads for i ??? 1:length(sumnodes)
    s = sumnodes[i]
    S, Z = curr[s], prev[s]
    ?? = Vector{Float64}(undef, length(S.children))
    # ?? = zeros(length(S.children))
    # infs = 0
    for (j, c) ??? enumerate(S.children)
      v = view(V, :, c)
      ??[j] = log(S.weights[j]) + logsumexp(view(??, :, s) .+ view(V, :, c) .- LL)
      # for t ??? 1:numrows
        # ??[j] += S.weights[j] * exp(??[t,s]) * exp(V[t,c] - LL[t])
      # end
      # isinf(??[j]) && (infs += 1)
    end
    # if infs == length(S.children) continue end

    # u = exp.(?? .- log_n) .+ smoothing
    # u ./= sum(u)
    # u .*= learningrate
    # u .+= (1.0 - learningrate) * S.weights
    # Z.weights .= u ./ sum(u)

    u = exp.(?? .- log_n) .+ smoothing / length(??)
    u .*= learningrate / sum(u)
    u .+= (1.0 - learningrate) * S.weights
    Z.weights .= u ./ sum(u)

    # u = exp.(?? .- logsumexp(??)) .+ smoothing
    # if any(isnan, u ./ sum(u)) println("break -2.1 ", u, ??); return -2, V, ??, ??, u end
    # u ./= sum(u)
    # u .= learningrate .* u .+ (1.0-learningrate) .* Z.weights
    # if any(isnan, u ./ sum(u)) println("break -2.2 ", u); return -2, V, ??, ??, u end
    # S.weights .= u ./ sum(u)
  end

  # Update Gaussian parameters
  if learngaussians
    mean_u, var_u = Vector{Float64}(undef, length(gaussiannodes)), Vector{Float64}(undef, length(gaussiannodes))
    Threads.@threads for i ??? 1:length(gaussiannodes)
      g = gaussiannodes[i]
      G, H = curr[g], prev[g]
      ?? = view(??, :, g) .+ view(V, :, g) .- LL
      ?? .= exp.(?? .- logsumexp(??))
      X = view(Data, :, G.scope)
      mean_u[i] = learningrate*sum(?? .* X) + (1-learningrate)*H.mean
      var_u[i] = learningrate*sum(?? .* ((X .- mean_u[i]) .^ 2)) + (1-learningrate)*H.variance
    end
    # if any(isnan, mean_u) || any(isnan, var_u) return -2, V, ?? end
    for i ??? 1:length(gaussiannodes)
      G = curr[gaussiannodes[i]]
      G.mean, G.variance = mean_u[i], var_u[i] < minimumvariance ? minimumvariance : var_u[i]
    end
  end

  if verbose && (learner.steps % 2 == 0)
    print("Training LL: ", sum(LL)/numrows, " | ")
    n = size(validation, 1)
    if n <= numrows
      ll = mLL!(view(V, 1:n, :), curr, validation)
    else
      ll = mLL!(Matrix{Float64}(undef, n, length(curr)), curr, validation)
    end
    # if any(isnan, ll) println("break -3"); return -3, V, ??, ll end
    if !isnothing(history) push!(history, ll) end
    println("Iteration $(learner.steps). ??: $(learningrate), LL: $(ll)")
  end

  swap!(learner.circ)
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -sum(LL) / numrows

  return learner.prevscore - learner.score
end
export update

"Full EM, but computed online so that memory does not explode."
function oupdate(
  learner::SEM,
  Data::AbstractMatrix,
  batchsize::Integer,
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

  batches = prepare_step_indices(numrows, batchsize)
  B = [zeros(length(curr[s].children)) for s ??? sumnodes]

  if learngaussians
    ??_norm = 0.0
    ?? = Vector{Float64}(undef, length(batchsize))
    ?? = zeros(length(gaussiannodes))
    ?? = zeros(length(gaussiannodes))
  end

  V = Matrix{Float64}(undef, batchsize, length(curr))
  ?? = Matrix{Float64}(undef, batchsize, length(curr))
  n = size(V, 1)
  ll = 0.0
  for (b, I) ??? enumerate(batches)
    k = length(I)
    if k < n
      V_b, ??_b = view(V, 1:k, :), view(??, 1:k, :)
    else
      V_b, ??_b = V, ??
    end
    batch = view(Data, I, :)
    LL = mplogpdf!(V_b, curr, batch)
    pbackpropagate_tree!(??_b, curr, V_b)

    Threads.@threads for i ??? 1:length(sumnodes)
      s = sumnodes[i]
      S = curr[s]
      for (j, c) ??? enumerate(S.children)
        B[i][j] += sum(exp.(view(??_b, :, s) .+ view(V_b, :, c) .- LL))
      end
    end

    if learngaussians
      Threads.@threads for i ??? 1:length(gaussiannodes)
        g = gaussiannodes[i]
        ?? .= exp.(view(??_b, :, g) .+ view(V_b, :, g) .- LL)
        ??_norm += sum(??)
        ??[i] .+= ?? .* batch
        ??[i] .+= ?? .* (batch .- a)
      end
    end

    ll += sum(LL)
  end

  Threads.@threads for i ??? 1:length(sumnodes)
    s = sumnodes[i]
    S, Z = curr[s], prev[s]
    B[i] .*= S.weights
    # u = (B[i] / numrows) .+ smoothing
    # u ./= sum(u)
    # u .*= learningrate
    # u .+= (1.0 - learningrate) * S.weights
    # Z.weights .= u ./ sum(u)

    u = (B[i] / numrows) .+ smoothing / length(B[i])
    u .*= learningrate / sum(u)
    u .+= (1.0 - learningrate) * S.weights
    Z.weights .= u ./ sum(u)

    # u = (?? ./ sum(??)) .+ smoothing
    # u .= learningrate .* (u / sum(u)) .+ (1.0-learningrate) .* Z.weights
    # S.weights .= u / sum(u)
  end

  if verbose && (learner.steps % 2 == 0)
    print("Training LL: ", ll/numrows, " | ")
    m = size(validation, 1)
    if m <= batchsize
      ll = mLL!(view(V, 1:m, :), curr, validation)
    else
      ll = mLL!(Matrix{Float64}(undef, m, length(curr)), curr, validation)
    end
    if !isnothing(history) push!(history, ll) end
    println("Iteration $(learner.steps). ??: $(learningrate), LL: $(ll)")
  end

  swap!(learner.circ)
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -ll / numrows

  return learner.prevscore - learner.score, V, ??
end
export oupdate

function old_update(
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
    gs = learner.circ.gauss
    gaussiannodes = gs.G
    if length(gaussiannodes) > 0
        ?? = Dict{UInt, Vector{Float64}}(x => Vector{Float64}(undef, numrows) for x ??? gaussiannodes)
    end
  end
  diff = learner.circ.D
  values = learner.circ.V

  for t in 1:numrows
    datum = view(Data, t, :)
    lv = cplogpdf!(values, curr, learner.circ.L, datum) # parallelized version
    @assert isfinite(lv) "logvalue of datum $t is not finite: $lv"
    score += lv
    backpropagate_tree!(diff, curr, learner.circ.L, values)
    Threads.@threads for l in 1:length(sumnodes) # update each node in parallel
      i = sumnodes[l]
      n = curr[i]
      p = prev[i]
      @inbounds for (j, c) in enumerate(n.children)
        u = values[c]
        if isfinite(u)
          ?? = n.weights[j] * diff[i] * exp(u - lv) # improvement
          if !isfinite(??) ?? = 0.0 end
        else
          ?? = 0.0
        end
        p.weights[j] = ((t - 1) / t) * p.weights[j] + ?? / t # running average for improved precision
      end
    end
    if learngaussians
      Threads.@threads for n in gaussiannodes
        ??[n][t] = log(diff[n])+(values[n]-lv)
      end
    end
  end

  Threads.@threads for i in 1:length(sumnodes)
    j = sumnodes[i]
    n = curr[j]
    p = prev[j]
    p.weights .+= smoothing / length(p.weights) # smoothing factor to prevent degenerate probabilities
    p.weights .*= learningrate / sum(p.weights) # normalize weights
    # online update: ??[t+1] = (1-??)*??[t] + ??*update(??[t])
    p.weights .+= (1.0 - learningrate) * n.weights
    p.weights ./= sum(p.weights)
    # @assert sum(circ_n[i].weights) ??? 1.0 "Unnormalized weight vector at node $i: $(sum(circ_n[i].weights)) | $(circ_n[i].weights) | $(circ_p[i].weights)"
  end
  if learngaussians
    Threads.@threads for n in gaussiannodes
      p = prev[n]
      cg = curr[n]
      ?? = ??[n]
      ?? .= exp.(?? .- logsumexp(??))
      X = view(Data, :, p.scope)
      mean_u = learningrate*sum(?? .* X) + (1-learningrate)*cg.mean
      var_u = learningrate*sum(?? .* ((X .- mean_u) .^ 2)) + (1-learningrate)*cg.variance
      if !(isnan(mean_u) || isnan(var_u))
        @inbounds p.mean = mean_u
        @inbounds p.variance = var_u
      end
      if !(p.variance > minimumvariance) p.variance = minimumvariance end
    end
  end

  if verbose && (learner.steps % 2 == 0)
    ll = pNLL(values, curr, learner.circ.L, validation)
    if !isnothing(history) push!(history, ll) end
    println("Iteration $(learner.steps). ??: $(learningrate), NLL: $(ll)")
  end

  swap!(learner.circ)
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -score / numrows

  return learner.prevscore - learner.score
end
export old_update

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

  ??_0 = learner.circ
  ??_1 = learner.cache1
  ??_2 = learner.cache2
  r = learner.cache3
  v = learner.cache4
  ??_1 = learner.cache1_map
  ??_2 = learner.cache2_map
  ??_r = learner.cache3_map
  ??_v = learner.cache4_map
  sumnodes = sums(??_0)
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
    backpropagate!(diff, ??_0, values) # backpropagate derivatives
    Threads.@threads for n in sumnodes # update each node in parallel
      @inbounds for (j, c) in enumerate(n.children)
        if isfinite(values[c])
          ?? = n.weights[j] * diff[n] * exp(values[c] - lv) # improvement
          @assert isfinite(??) "1. improvement to weight ($n,$c):$(n.weights[j]) is not finite: $??, $(diff[n]), $(values[j]), $(exp(values[c]-lv))"
        else
          ?? = 0.0
        end
        u = ??_1[n]
        u.weights[j] = ((t - 1) / t) * u.weights[j] + ?? / t # running average for improved precision
        @assert u.weights[j] ??? 0
      end
    end
    # if learngaussians
    #   Threads.@threads for i in gaussiannodes
    #       @inbounds ?? = diff[i]*exp(values[i]-lv)
    #       @inbounds denon[i] += ??
    #       @inbounds means[i] += ??*datum[??_0[i].scope]
    #       @inbounds squares[i] += ??*datum[??_0[i].scope]^2
    #   end
    # end
  end
  @inbounds Threads.@threads for n in sumnodes
    # println(??_1[i].weights)
    u = ??_1[n]
    u.weights .+= smoothing / length(u.weights) # smoothing factor to prevent degenerate probabilities
    # println("  ", ??_1[i].weights)
    u.weights ./= sum(u.weights)
    # println("    ", ??_1[i].weights)
    @assert sum(u.weights) ??? 1.0 "1. Unnormalized weight vector at node $n: $(sum(u.weights)) | $(u.weights)"
  end
  # if learngaussians
  #   Threads.@threads for i in gaussiannodes
  #       @inbounds ??_1[i].mean = means[i]/denon[i]
  #       @inbounds ??_1[i].variance = squares[i]/denon[i] - (??_1[i].mean)^2
  #       @inbounds if ??_1[i].variance < minimumvariance
  #           @inbounds ??_1[i].variance = minimumvariance
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
    backpropagate!(diff, ??_1, values) # backpropagate derivatives
    Threads.@threads for n in sumnodes # update each node in parallel
      @inbounds for (j, c) in enumerate(n.children)
        if isfinite(values[c])
          u = ??_1[n]
          ?? = u.weights[j] * diff[c] * exp(values[c] - lv) # improvement
          @assert isfinite(??) "2. improvement to weight ($n,$c):$(u.weights[j]) is not finite: $??, $(diff[n]), $(values[c]), $(exp(values[c]-lv))"
        else ?? = 0.0 end
        u = ??_2[n]
        u.weights[j] = ((t - 1) / t) * u.weights[j] + ?? / t
        @assert u.weights[j] ??? 0
      end
    end
    # if learngaussians
    #   Threads.@threads for i in gaussiannodes
    #       @inbounds ?? = diff[i]*exp(values[i]-lv)
    #       @inbounds denon[i] += ??
    #       @inbounds means[i] += ??*datum[??_0[i].scope]
    #       @inbounds squares[i] += ??*datum[??_0[i].scope]^2
    #   end
    # end
  end
  @inbounds Threads.@threads for n in sumnodes
    # println(??_2[i].weights)
    u = ??_2[n]
    u.weights .+= smoothing / length(u.weights) # smoothing factor to prevent degenerate probabilities
    # println("  ", ??_2[i].weights)
    u.weights ./= sum(u.weights)
    # println("    ", ??_2[i].weights)
    @assert sum(u.weights) ??? 1.0 "2. Unnormalized weight vector at node $n: $(sum(u.weights)) | $(u.weights)"
  end
  # if learngaussians
  #   Threads.@threads for i in gaussiannodes
  #       @inbounds ??_2[i].mean = means[i]/denon[i]
  #       @inbounds ??_2[i].variance = squares[i]/denon[i] - (??_2[i].mean)^2
  #       @inbounds if ??_2[i].variance < minimumvariance
  #           @inbounds ??_2[i].variance = minimumvariance
  #       end
  #   end  
  # Compute r, v, |r| and |v|
  r_norm, v_norm = 0.0, 0.0
  @inbounds Threads.@threads for n in sumnodes
    # r[i].weights .= ??_1[i].weights .- ??_0[i].weights
    # v[i].weights .= ??_2[i].weights .- ??_1[i].weights .- r[i].weights
    p, q = ??_r[n], ??_v[n]
    a, b = ??_1[n], ??_2[n]
    c = ??_0[n]
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
  ?? = -max(sqrt(r_norm) / sqrt(v_norm), 1)
  #println("??: $??")
  # Compute ??' (reuse ??_1 for that matter)
  @inbounds Threads.@threads for n in sumnodes
    # ??' = ??0 - 2??r + ??^2v
    p, q = ??_1[n], ??_0[n]
    a, b = ??_r[n], ??_v[n]
    p.weights .= q.weights
    p.weights .-= ((2 * ??) .* a.weights)
    p.weights .+= ((?? * ??) .* b.weights)
    p.weights .+ smoothing / length(p.weights) # add term to prevent negative weights due to numerical imprecision
    p.weights ./= sum(p.weights)
    @assert sum(p.weights) ??? 1.0 "3. Unnormalized weight vector at node $n: $(sum(p.weights)) | $(p.weights)"
    for w in p.weights @assert w ??? 0 "Negative weight at node $n: $(p.weights)" end
  end
  # Final EM Update: ??_0 = EM_Update(??')
  score = 0.0 # data loglikelihood
  for t in 1:numrows
    datum = view(Data, t, :)
    lv = plogpdf!(values, learner.layersp, datum) # parallelized version
    @assert isfinite(lv) "4. logvalue of datum $t is not finite: $lv"
    score += lv
    backpropagate!(diff, ??_1, values) # backpropagate derivatives
    Threads.@threads for n in sumnodes # update each node in parallel
      @inbounds for (j, c) in enumerate(n.children)
        if isfinite(values[c])
          u = ??_1[n]
          ?? = u.weights[j] * diff[n] * exp(values[n] - lv) # improvement
          @assert isfinite(??) "4. improvement to weight ($n,$c):$(u.weights[j]) is not finite: $??, $(diff[n]), $(values[n]), $(exp(values[n]-lv))"
        else
          ?? = 0.0
        end
        n.weights[j] = ((t - 1) / t) * n.weights[j] + ?? / t
        @assert n.weights[j] ??? 0
      end
    end
  end
  @inbounds Threads.@threads for n in sumnodes
    n.weights .+= smoothing / length(n.weights) # smoothing factor to prevent degenerate probabilities
    n.weights ./= sum(n.weights)
    @assert sum(n.weights) ??? 1.0 "4. Unnormalized weight vector at node $n: $(sum(n.weights)) | $(n.weights)"
  end
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -score / numrows
  return learner.prevscore - learner.score, ??
end
export update
