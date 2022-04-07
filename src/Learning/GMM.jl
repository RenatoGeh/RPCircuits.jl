import GaussianMixtures

"""
    learn_gmm(D::AbstractMatrix{<:Real}, x::Integer; k::Int = 3, kmeans_iter::Int = 50, em_iter::Int = 10)::Node

Learns a Gaussian mixture model with `k` components from data `D`.
"""
@inline function learn_gmm(D::Vector{Float64}, x::Integer; k::Int = 3, kmeans_iter::Int = 50,
    em_iter::Int = 0, minvar::Real = 1e-3, smoothing::Real = 1e-1)::Node
  if size(D, 1) < k D = repeat(D, 2*k) end
  # G = @suppress begin
    # GaussianMixtures.GMM(k, D; method = :kmeans, nInit = 50, nIter = 10, nFinal = 10)
  # end
  G = GaussianMixtures.GMM(k, D; method = :kmeans, nInit = kmeans_iter, nIter = em_iter)
  w = GaussianMixtures.weights(G) .+ smoothing
  μ = GaussianMixtures.means(G)
  σ = GaussianMixtures.covars(G)
  K = [Gaussian(x, μ[i], σ[i] < minvar ? 1e-3 : σ[i]) for i ∈ 1:length(μ)]
  # println(n, ", ", w, ", ", μ, ", ", σ)
  return Sum(K, w / sum(w))
end

"""
  learn_multi_gmm(D::AbstractMatrix{<:Real}; k::Int = 3, kmeans_iter::Int = 50, em_iter::Int = 100)::Node

Learns a multivariate Gaussian mixture model with `k` components from data `D`.
"""
function learn_multi_gmm(D::AbstractMatrix{<:Real}; k::Int = 3, kmeans_iter::Integer = 50,
    minibatch_iter::Integer = 100, em_iter::Integer = 30, smoothing::Float64 = 1e-3,
    minvar::Float64 = 1e-3, verbose::Bool = true, validation::AbstractMatrix{<:Real} = D,
    V::Union{UnitRange, AbstractVector{<:Integer}} = 1:size(D, 2), batchsize::Integer = 100)::Node
  if size(D, 1) < k D = repeat(D, 2*k) end
  gmm = GaussianMixtures.GMM(k, Matrix(D); kind = :diag, method = :kmeans, nInit = kmeans_iter, nIter = 0)
  # GaussianMixtures.em!(gmm, D; nIter = 100, varfloor = 1e-3)
  # println("GMM LL: ", sum(GaussianMixtures.logsumexpw(GaussianMixtures.llpg(gmm, validation), gmm.w))/size(validation, 1))
  # println("GMM avll: ", GaussianMixtures.avll(gmm, validation))
  w = GaussianMixtures.weights(gmm)
  μ = GaussianMixtures.means(gmm)
  Σ = GaussianMixtures.covars(gmm)
  n, m = size(D)
  K = Vector{Node}(undef, k)
  for i ∈ 1:k
    G = [Gaussian(V[j], μ[i,j], Σ[i,j] < minvar ? minvar : Σ[i,j]) for j ∈ 1:m]
    K[i] = Product(G)
  end
  C = Sum(K, w)
  # println("Initial LL: ", -NLL(C, validation))
  L = SEM(C; gauss = true)
  if minibatch_iter > 0
    ranges = prepare_step_indices(n, batchsize)
    indices = collect(1:n)
    while L.steps < minibatch_iter
      if L.steps % length(ranges) == 0 shuffle!(indices) end
      I = view(indices, ranges[(L.steps % length(ranges)) + 1])
      batch = view(D, I, :)
      η = 0.975^L.steps
      update(L, batch, η, smoothing, true, minvar; verbose, validation)
    end
  end
  if em_iter <= 0 return C end
  while L.steps < em_iter
    # print("Training LL: ", -NLL(C, D), " -> ")
    update(L, D, 1.0, smoothing, true, minvar; verbose, validation)
  end
  return C
end
export learn_multi_gmm
