import GaussianMixtures

"""
    learn_gmm(D::AbstractMatrix{<:Real}, x::Integer; k::Int = 3, kmeans_iter::Int = 50, em_iter::Int = 10)::Node

Learns a Gaussian mixture model with one component from data `D`.
"""
@inline function learn_gauss(D::Vector{Float64}, x::Integer; kmeans_iter::Int = 50, em_iter::Int = 0, 
    minvar::Real = 1e-3, smoothing::Real = 1e-1)::Node
  G = GaussianMixtures.GMM(1, D; method = :kmeans, nInit = kmeans_iter, nIter = em_iter)
  μ = GaussianMixtures.means(G)
  σ = GaussianMixtures.covars(G)
  # Circuit with only a gaussian node
  C = Gaussian(x, μ[1], σ[1] < minvar ? 1e-3 : σ[1])
  return C
end;
export learn_gauss

"""
  Learns, by Gradient Descent, a Gaussian mixture model with one component from data `D`.
"""
@inline function learn_gauss_grad(D::Vector{Float64}, x::Integer; kmeans_iter::Int = 50, grad_iter::Int = 100, 
    minvar::Real = 1e-3, smoothing::Real = 1e-6, learning_rate::Float64 = 0.1, verbose::Bool =true, 
    validation::AbstractVector{<:Real} = D)::Node
  G = GaussianMixtures.GMM(1, D; method = :kmeans, nInit = kmeans_iter, nIter = 0)
  μ = GaussianMixtures.means(G)
  σ = GaussianMixtures.covars(G)
  # Circuit with only a gaussian node
  C = Gaussian(x, μ[1], σ[1] < minvar ? 1e-3 : σ[1])

  if grad_iter <= 0 return C end
  # println("Initial LL: ", -NLL(C, validation))
  println("Learning weights...")
  L = GRAD(C; gauss = true)
  D = reshape(D, length(D), 1)
  validation = reshape(validation, length(validation), 1)
  while L.steps < grad_iter
    # print("Training LL: ", -NLL(C, D), " -> ")
    update(L, D; learningrate=learning_rate, smoothing=smoothing, learngaussians=true, minimumvariance=minvar, verbose=verbose, validation=validation)
  end
  return C
end
export learn_gauss_grad

"""
  learn_multi_gmm(D::AbstractMatrix{<:Real}; k::Int = 3, kmeans_iter::Int = 50, em_iter::Int = 100)::Node

Learns a multivariate Gaussian mixture model with one component from data `D`.
"""
function learn_multi_gauss(D::AbstractMatrix{<:Real}; kmeans_iter::Int = 50, em_iter::Int = 100, smoothing::Float64 = 0.1, 
    minvar::Float64 = 1e-3, verbose::Bool = true, validation::AbstractMatrix{<:Real} = D, 
    V::Union{UnitRange, AbstractVector{<:Integer}} = 1:size(D, 2))::Node
  gmm = GaussianMixtures.GMM(1, Matrix(D); kind = :diag, method = :kmeans, nInit = kmeans_iter, nIter = 0)
  μ = GaussianMixtures.means(gmm)
  Σ = GaussianMixtures.covars(gmm)
  n, m = size(D)
  G = [Gaussian(V[j], μ[1,j], Σ[1,j] < minvar ? minvar : Σ[1,j]) for j ∈ 1:m]
  # Product of independent univariate gaussians
  C = Product(G)
  if em_iter <= 0 return C end
  # println("Initial LL: ", -NLL(C, validation))
  println("Learning weights...")
  L = SEM(C; gauss = true)
  while L.steps < em_iter
    # print("Training LL: ", -NLL(C, D), " -> ")
    update(L, D; learningrate=1.0, smoothing=smoothing, learngaussians=true, minimumvariance=minvar, verbose=verbose, validation=validation)
  end
  return C
end
export learn_multi_gmm

function learn_multi_gauss_grad(D::AbstractMatrix{<:Real}; kmeans_iter::Int = 50, grad_iter::Int = 100,
    smoothing::Float64 = 1e-6, minvar::Float64 = 1e-3, verbose::Bool = true, validation::AbstractMatrix{<:Real} = D, 
    init_weights::Float64 = 0.1, learning_rate::Float64 = 0.1)::Node
  println("Finding initial means and variances...")
  gmm = GaussianMixtures.GMM(1, Matrix(D); kind = :diag, method = :kmeans, nInit = kmeans_iter, nIter = 0)
  μ = GaussianMixtures.means(gmm)
  Σ = GaussianMixtures.covars(gmm)
  n, m = size(D)
  G = [Gaussian(V[j], μ[1,j], Σ[1,j] < minvar ? minvar : Σ[1,j]) for j ∈ 1:m]
  # Product of independent univariate gaussians
  C = Product(G)
  if grad_iter <= 0 return C end
  # println("Initial LL: ", -NLL(C, validation))
  println("Learning weights...")
  L = GRAD(C; gauss = true)
  while L.steps < grad_iter
    # print("Training LL: ", -NLL(C, D), " -> ")
    update(L, D; learningrate=learning_rate, smoothing=smoothing, learngaussians=true, minimumvariance=minvar, verbose=verbose, validation=validation)
  end
  return C
end
export learn_multi_gauss_grad
