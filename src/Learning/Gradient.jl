# Parameter learning by batch Gradient Descent

"""
Learn weights using the Gradient algorithm.
"""
mutable struct GRAD <: ParameterLearner
  circ::CCircuit
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  function GRAD(r::Node; gauss::Bool = false)
    return new(compile(CCircuit, r; gauss), NaN, NaN, 1e-4, 0, 0.5)
  end
end
export GRAD

"""
Verifies if algorithm has converged.

Convergence is defined as absolute difference of score.
Requires at least 2 steps of optimization.
"""
converged(learner::GRAD) =
  learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance
export converged

# TODO: take filename os CSV File object as input and iterate over file to decrease memory footprint
"""
Improvement step for learning weights using the Gradient Descent algorithm. Returns improvement in negative loglikelihood.

circ[i].weights[j] = circ[i].weights[k] * backpropagate(circ)[i]/sum(circ[i].weights[k] * backpropagate(circ)[i] for k=1:length(circ[i].children))

## Arguments

  - `learner`: GRADParamLearner struct
  - `data`: Data Matrix
  - `learningrate`: learning inertia, used to avoid forgetting in minibatch learning [default: 1.0 correspond to no inertia, i.e., previous weights are discarded after update]
  - `smoothing`: weight smoothing factor (= pseudo expected count) [default: 1e-4]
  - `learngaussians`: whether to also update the parameters of Gaussian leaves [default: false]
  - `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(
  learner::GRAD,
  Data::AbstractMatrix;
  learningrate::Float64 = 1e-4,
  smoothing::Float64 = 1e-4,
  learngaussians::Bool = false,
  minimumvariance::Float64 = learner.minimumvariance,
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

  V = Matrix{Float64}(undef, numrows, length(curr)) # V[i][j] = Sj(Di)
  Δ = Matrix{Float64}(undef, numrows, length(curr)) # Δ = ΔS(Di)/ΔSj

  norm_V = Vector{Float64}(undef, length(curr)) # norm_v = Sj(1)
  norm_Δ = Vector{Float64}(undef, length(curr)) # norm_Δ = ΔS(1)/ΔSj

  # Compute backward pass (values), considering normalization
  # LL = ∑d log(S(d)/S(1)) = ∑d [log S(d)] - |D|log S(1)
  LL = mplogpdf!(V, curr, Data)
  # Updated norm_V with values Si(1) and assigns norm_const to the normalizing constant of the circuit, S(1).
  norm_const = log_norm_const!(norm_V, curr)

  # Compute forward pass (derivatives)
  pbackpropagate_tree!(Δ, curr, V)
  # Compute forward pass of marginalized variables
  norm_backpropagate_tree!(norm_Δ , curr, norm_V) # norm_Δ[i] = ΔS(1)/ΔSi
  
  # Update sum weights
  log_n = log(numrows)
  Threads.@threads for i ∈ 1:length(sumnodes)
    s = sumnodes[i]
    S = curr[s]
    u = Vector{Float64}(undef, length(S.children))
    for (j, c) ∈ enumerate(S.children)
      # Δw_ij = Δu = \sum
      u[j] = sum(exp.(view(Δ, :, s) .+ view(V, :, c) .- LL)) - exp(log_n + norm_Δ[i] + norm_V[j] - norm_const)
    end
    u ./= numrows
    S.weights .+= learningrate.*u
  end

  # Update Gaussian parameters
  if learngaussians
    Threads.@threads for i ∈ 1:length(gaussiannodes)
      g = gaussiannodes[i]
      G = curr[g]
      # Gaussian scope
      X = view(Data, :, G.scope)
      # Gaussian parameters
      μ, σ = G.mean, G.variance
      # Δμ = 1/N ∑ (x-μ)/σ f(x)
      Δμ = sum(((X .- μ) ./ σ) .* view(V, :, g))/numrows
      G.mean -= learningrate*Δμ
      # Derivate (w.r.t σ) of log f(x)
      log_d = ((X .- μ).^2)./ (2*σ^2) .- 1/(2*σ)
      # Δσ = 1/N ∑ [(x-μ)^2/(2z^2) - 1/(2*σ)] f(x)
      Δσ = sum(log_d .* view(V, :, g))/numrows
      G.variance -= learningrate*Δσ
      !(G.variance > minimumvariance) && (G.variance = minimumvariance)
    end
  end

  if verbose && (learner.steps % 100 == 0)
    # TODO: Mudar o nome de LL para outra coisa que represente S(d), valor do circuito no dado
    print("Training LL: ", sum(LL)/numrows - norm_const, " | ")
    m = size(validation, 1)
    # loglikelihood of validation set
    if m <= numrows
      ll = mLL!(view(V, 1:m, :), curr, validation) - norm_const
    else
      ll = mLL!(Matrix{Float64}(undef, m, length(curr)), curr, validation) - norm_const
    end
    if !isnothing(history) push!(history, ll) end
    println("Iteration $(learner.steps). η: $(learningrate), LL: $(ll)")
  end

  swap!(learner.circ)
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -sum(LL) / numrows + norm_const

  return learner.prevscore - learner.score, V, Δ
end
export update
