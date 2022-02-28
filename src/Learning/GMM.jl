import GaussianMixtures

"""
    learn_gmm(D::AbstractMatrix{<:Real}; k::Int, iter::Int)

Learns a Gaussian mixture model with `k` components from data `D`.
"""
@inline function learn_gmm(D::Vector{Float64}, x::Integer; k::Int = 3)::Node
  if size(D, 1) < k D = repeat(D, 2*k) end
  # G = @suppress begin
    # GaussianMixtures.GMM(k, D; method = :kmeans, nInit = 50, nIter = 10, nFinal = 10)
  # end
  G = GaussianMixtures.GMM(k, D; method = :kmeans, nInit = 50, nIter = 10, nFinal = 10)
  w = GaussianMixtures.weights(G)
  μ = GaussianMixtures.means(G)
  σ = GaussianMixtures.covars(G)
  K = [Gaussian(x, μ[i], σ[i]) for i ∈ 1:length(μ)]
  # println(n, ", ", w, ", ", μ, ", ", σ)
  return Sum(K, w)
end
