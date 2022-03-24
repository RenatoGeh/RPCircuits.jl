using LogicCircuits: Vtree, PlainVtreeLeafNode, variable, variables
using ProbabilisticCircuits: learn_vtree
using LinearAlgebra
using ThreadPools
using Distributions: Dirichlet

BLAS.set_num_threads(Threads.nthreads())

@inline isleaf(v::Vtree)::Bool = isa(v, PlainVtreeLeafNode)

function learn_rpp_proj!(S::SubArray{<:Real, 2}, V::Vtree, f::Function, height::Int,
    bin::Bool, min_examples::Int, max_height::Int, rep_projs::Int, max_projs::Int,
    I::Vector{Tuple{Indicator, Indicator}}, gmm::Bool, pseudocount::Int)::Node
  n, m = size(S)
  if (n < min_examples) || (height > max_height)
    @label ff
    ch = Vector{Node}(undef, m)
    U = S.indices[2]
    if bin
      W = (sum(S; dims = 1) .+ pseudocount) ./ (n + pseudocount*2)
      for i ∈ 1:m
        u, w = U[i], W[i]
        ⊥, ⊤ = I[u]
        ch[i] = Sum([⊥, ⊤], [1.0-w, w])
      end
    elseif gmm return learn_multi_gmm(S; em_iter = 0, V = U)
    else
      μ = mean(S; dims = 1)
      s = std(S; dims = 1, mean = μ)
      map!(x -> !(x > 0.03) ? 1e-3 : x*x, s, s)
      for i ∈ 1:m ch[i] = Gaussian(U[i], μ[i], s[i]) end
    end
    return Product(ch)
  end
  a, _, g = f(S)
  if isnothing(a) @goto ff end
  # Negatives (A) and positives (B).
  A, B = select(g, S)
  # Initially give weights as the data proportion.
  k = size(A, 1)/n; w = [k, 1-k]
  # Be sure to pass w as a pointer (actually a reference) to the sum's weights so we can
  # efficiently compute during parameter learning.
  # proj = Projection(pa_id, Ref(w), a, θ)
  # push!(P, proj)
  # IDs are the indexes of the parent projections.
  if rep_projs > 0
    return Sum([learn_rpp_proj!(A, V, f, height+1, bin, min_examples, max_height, rep_projs-1,
                                max_projs, I, gmm, pseudocount),
                learn_rpp_proj!(B, V, f, height+1, bin, min_examples, max_height, rep_projs-1,
                                max_projs, I, gmm, pseudocount)], w)
  end
  return Sum([learn_rpp_part!(A, V, f, height+1, bin, min_examples, max_height, max_projs, I, gmm, pseudocount),
              learn_rpp_part!(B, V, f, height+1, bin, min_examples, max_height, max_projs, I, gmm, pseudocount)], w)
end

function learn_rpp_part!(S::SubArray{<:Real, 2}, V::Vtree, f::Function, height::Int, bin::Bool,
    min_examples::Int, max_height::Int, max_projs::Int, I::Vector{Tuple{Indicator, Indicator}},
    gmm::Bool, pseudocount::Real)::Node
  n, m = size(S)
  # Single variable. Return univariate distribution.
  if isleaf(V)
    u = variable(V)
    Z = reshape(S, :)#view(S, :, u)
    if bin
      w = (sum(Z)+pseudocount)/(n+pseudocount*2)
      ⊥, ⊤ = I[u]
      return Sum([⊥, ⊤], [1.0-w, w])
    elseif gmm return learn_gmm(S[:,1], u) end
    μ = mean(Z)
    s = std(Z; mean = μ)
    # This deals with NaNs, Infs and 0.
    if !(s < 0.03) s = 1e-3 else s *= s end
    return Gaussian(u, μ, s)
  # Small dataset or max height. Return fully factorized circuit.
  elseif (n < min_examples) || (height > max_height)
    @label ff
    ch = Vector{Node}(undef, m)
    U = S.indices[2]
    if bin
      W = (sum(S; dims = 1) .+ pseudocount) ./ (n + pseudocount*2)
      for i ∈ 1:m
        u, w = U[i], W[i]
        ⊥, ⊤ = I[u]
        ch[i] = Sum([⊥, ⊤], [1.0-w, w])
      end
    elseif gmm
      for i ∈ 1:m ch[i] = learn_gmm(S[:,i], U[i]) end
    else
      μ = mean(S; dims = 1)
      s = std(S; dims = 1, mean = μ)
      map!(x -> !(x > 0.03) ? 1e-3 : x*x, s, s)
      for i ∈ 1:m ch[i] = Gaussian(U[i], μ[i], s[i]) end
    end
    return Product(ch)
  end
  # Create projection and subsequent sum node.
  a, _, g = f(S)
  if isnothing(a) @goto ff end
  # Negatives (A) and positives (B).
  A, B = select(g, S)
  # Initially give weights as the data proportion.
  k = size(A, 1)/n; w = [k, 1-k]
  Sc_sub, Sc_prime = variables(V.left), variables(V.right)
  U = S.indices[2]
  X, Y = findall(∈(Sc_sub), U), findall(∈(Sc_prime), U)
  # Negatives, each on the subs and primes.
  neg = Product([learn_rpp_proj!(view(A, :, X), V.left, f, height + 1, bin, min_examples,
                                 max_height, max_projs, max_projs, I, gmm, pseudocount),
                 learn_rpp_proj!(view(A, :, Y), V.right, f, height + 1, bin, min_examples,
                                 max_height, max_projs, max_projs, I, gmm, pseudocount)])
  # Positives, each on the subs and primes.
  pos = Product([learn_rpp_proj!(view(B, :, X), V.left, f, height + 1, bin, min_examples,
                                 max_height, max_projs, max_projs, I, gmm, pseudocount),
                 learn_rpp_proj!(view(B, :, Y), V.right, f, height + 1, bin, min_examples,
                                 max_height, max_projs, max_projs, I, gmm, pseudocount)])
  # Finally, return the resulting sum node, with same weights as the ones given to the projection.
  return Sum([neg, pos], w)
end

function learn_rpp!(S::SubArray{<:Real, 2}, V::Vtree, f::Function, height::Int, bin::Bool,
    min_examples::Int, max_height::Int, I::Vector{Tuple{Indicator, Indicator}}, gmm::Bool,
    pseudocount::Int)::Node
  n, m = size(S)
  # Single variable. Return univariate distribution.
  if isleaf(V)
    u = variable(V)
    Z = reshape(S, :)#view(S, :, u)
    if bin
      w = (sum(Z)+pseudocount)/(n+pseudocount*2)
      ⊥, ⊤ = I[u]
      return Sum([⊥, ⊤], [1.0-w, w])
    elseif gmm return learn_gmm(S[:,1], u) end
    μ = mean(Z)
    s = std(Z; mean = μ)
    # This deals with NaNs, Infs and 0.
    if !(s > 0.03) s = 1e-3 else s *= s end
    return Gaussian(u, μ, s)
  # Small dataset or max height. Return fully factorized circuit.
  # TODO: fully factorized circuit that follows a vtree (contiguous 2-prods following vtree)
  elseif (n < min_examples) || (height > max_height)
    @label ff
    ch = Vector{Node}(undef, m)
    U = S.indices[2]
    if bin
      W = (sum(S; dims = 1) .+ pseudocount) ./ (n + pseudocount*2)
      for i ∈ 1:m
        u, w = U[i], W[i]
        ⊥, ⊤ = I[u]
        ch[i] = Sum([⊥, ⊤], [1.0-w, w])
      end
    elseif gmm
      for i ∈ 1:m ch[i] = learn_gmm(S[:,i], U[i]) end
    else
      μ = mean(S; dims = 1)
      s = std(S; dims = 1, mean = μ)
      map!(x -> !(x > 0.03) ? 1e-3 : x*x, s, s)
      for i ∈ 1:m ch[i] = Gaussian(U[i], μ[i], s[i]) end
    end
    return Product(ch)
  end
  # Create projection and subsequent sum node.
  a, _, g = f(S)
  if isnothing(a) @goto ff end
  # Negatives (A) and positives (B).
  A, B = select(g, S)
  # Initially give weights as the data proportion.
  k = size(A, 1)/n; w = [k, 1-k]
  Sc_sub, Sc_prime = variables(V.left), variables(V.right)
  U = S.indices[2]
  X, Y = findall(∈(Sc_sub), U), findall(∈(Sc_prime), U)
  # Negatives, each on the subs and primes.
  neg = Product([learn_rpp!(view(A, :, X), V.left, f, height + 1, bin, min_examples, max_height, I, gmm, pseudocount),
                 learn_rpp!(view(A, :, Y), V.right, f, height + 1, bin, min_examples, max_height, I, gmm, pseudocount)])
  # Positives, each on the subs and primes.
  pos = Product([learn_rpp!(view(B, :, X), V.left, f, height + 1, bin, min_examples, max_height, I, gmm, pseudocount),
                 learn_rpp!(view(B, :, Y), V.right, f, height + 1, bin, min_examples, max_height, I, gmm, pseudocount)])
  # Finally, return the resulting sum node, with same weights as the ones given to the projection.
  return Sum([neg, pos], w)
end

function learn_rpp_auto!(S::SubArray{<:Real, 2}, V::Vtree, f::Function, height::Int, bin::Bool,
    min_examples::Int, max_height::Int, I::Vector{Tuple{Indicator, Indicator}}, gmm::Bool,
    threshold::Float64, pseudocount::Int)::Node
  n, m = size(S)
  # Single variable. Return univariate distribution.
  if isleaf(V)
    u = variable(V)
    Z = reshape(S, :)#view(S, :, u)
    if bin
      w = (sum(Z)+pseudocount)/(n+pseudocount*2)
      ⊥, ⊤ = I[u]
      return Sum([⊥, ⊤], [1.0-w, w])
    elseif gmm return learn_gmm(S[:,1], u) end
    μ = mean(Z)
    s = std(Z; mean = μ)
    # This deals with NaNs, Infs and 0.
    if !(s > 0.2) s = 0.04 else s *= s end
    return Gaussian(u, μ, s)
  # Small dataset or max height. Return fully factorized circuit.
  elseif (n < min_examples) || (height > max_height)
    @label ff
    ch = Vector{Node}(undef, m)
    U = S.indices[2]
    if bin
      W = (sum(S; dims = 1) .+ pseudocount)/(n+pseudocount*2)
      for i ∈ 1:m
        u, w = U[i], W[i]
        ⊥, ⊤ = I[u]
        ch[i] = Sum([⊥, ⊤], [1.0-w, w])
      end
    elseif gmm
      for i ∈ 1:m ch[i] = learn_gmm(S[:,i], U[i]) end
    else
      μ = mean(S; dims = 1)
      s = std(S; dims = 1, mean = μ)
      map!(x -> !(x > 0.2) ? 0.04 : x*x, s, s)
      for i ∈ 1:m ch[i] = Gaussian(U[i], μ[i], s[i]) end
    end
    return Product(ch)
  end
  if avgdiam(S) < threshold
    a, θ, g = f(S)
    if isnothing(a) @goto ff end
    A, B = select(g, S)
    k = (size(A, 1)+pseudocount)/(n+pseudocount*2); w = [k, 1-k]
    return Sum([learn_rpp_auto!(A, V, f, height+1, bin, min_examples, max_height, I, gmm, threshold, pseudocount),
                learn_rpp_auto!(B, V, f, height+1, bin, min_examples, max_height, I, gmm, threshold, pseudocount)], w)
  end
  Sc_sub, Sc_prime = variables(V.left), variables(V.right)
  U = S.indices[2]
  X, Y = findall(∈(Sc_sub), U), findall(∈(Sc_prime), U)
  return Product([learn_rpp_auto!(view(S, :, X), V.left, f, height+1, bin, min_examples, max_height, I, gmm, threshold, pseudocount),
                  learn_rpp_auto!(view(S, :, Y), V.right, f, height+1, bin, min_examples, max_height, I, gmm, threshold, pseudocount)])
end

@inline sample_thresholds(D::AbstractMatrix{<:Real}, n::Int)::Vector{Float64} = ((rand(n)/2) .+ 0.5) * avgdiam(D) * 3
@inline equal_thresholds(D::AbstractMatrix{<:Real}, n::Int)::Vector{Float64} = collect(0.0:1.0/(n-1):1.0) * avgdiam(D)

function learn_rpp_auto(D::AbstractMatrix{<:Real}, V::Vtree; split::Symbol = :max, c::Real = 1.0,
    r::Real = 2.0, trials::Int = 10, bin::Bool = true, min_examples::Int = 30, max_height::Int =
    10, I::Vector{Tuple{Indicator, Indicator}} = bin ? [(Indicator(u, 0), Indicator(u, 1)) for u ∈ 1:size(D, 2)] : Tuple{Indicator, Indicator}[],
    gmm::Bool = false, samples::Int = 10, validation::AbstractMatrix{<:Real} = D,
    pseudocount::Int = 2, kwargs...)
  f = split == :max ? (x -> max_rulep(x, r, trials)) : (x -> sid_rulep(x, c, trials))
  # R = Vector{Node}(undef, samples)
  # T = equal_thresholds(D, samples)
  # Threads.@threads for i ∈ 1:samples
    # R[i] = learn_rpp_auto!(view(D, :, :), V, f, -1, bin, min_examples, max_height, I, gmm, T[i], pseudocount)
  # end
  # L = NLL.(R, Ref(validation))
  # t = argmin(L)
  # display(L)
  # println("\nBest threshold: ", T[t])
  # display(T); println()
  # mini_em(R[t], D; validation, learngaussians = !bin, kwargs...)
  # return R[t]
  return learn_rpp_auto!(view(D, :, :), V, f, -1, bin, min_examples, max_height, I, gmm, 0.5*avgdiam(D), pseudocount)
end
export learn_rpp_auto

"""Learns a PC by projections and returns the root and a vector with all projections."""
function learn_rpp(D::AbstractMatrix{<:Real}, V::Vtree; split::Symbol = :max, c::Real = 1.0, r::Real = 2.0,
    trials::Int = 10, bin::Bool = true, min_examples::Int = 30, max_height::Int = 10,
    I::Vector{Tuple{Indicator, Indicator}} = bin ? [(Indicator(u, 0), Indicator(u, 1)) for u ∈ 1:size(D, 2)] : Tuple{Indicator, Indicator}[],
    max_projs::Int = 0, gmm::Bool = false, pseudocount::Int = 2)::Node
  f = split == :max ? (x -> max_rulep(x, r, trials)) : (x -> sid_rulep(x, c, trials))
  return max_projs > 0 ? learn_rpp_proj!(view(D, :, :), V, f, -1, bin, min_examples, max_height, max_projs, max_projs, I, gmm, pseudocount) :
                         learn_rpp!(view(D, :, :), V, f, -1, bin, min_examples, max_height, I, gmm, pseudocount)
end
export learn_rpp

"""
    extract_a(P::Vector{Projection})::Matrix{Float64}

Constructs a matrix containing all projections' parameters.

Let m and k be the number of variables and projections resp. The resulting matrix is:

        | a_1^1 a_1^2 … a_1^k |
        | a_2^1 a_2^2 … a_2^k |
    A = |   ⋮     ⋮   … ⋱ ⋮   |
        | a_m^1 a_m^2 … a_m^k |
        |  θ_1   θ_2  …  θ_k  |

where `a_i^j` is the j-th projection's i-th dimension parameter.
"""
@inline function extract_a(P::Vector{Projection})::Matrix{Float64}
  m, k = length(P[1].a)+1, length(P)
  M = Matrix{Float64}(undef, m, k)
  extract_a!(P, M)
  return M
end
@inline function extract_a!(P::Vector{Projection}, M::Matrix{Float64})
  m, k = length(P[1].a)+1, length(P)
  Threads.@threads for i ∈ 1:k M[1:m-1,i] .= P[i].a; M[m,i] = P[i].θ end
  return nothing
end

"""
    pad_data(D::Matrix{<:Real})::Matrix{Float64}

Pads the data matrix with an additional column of one's so that we can perform the following matrix
operation:

            | x_11 x_12 … x_1m 1 |   | a_1^1 a_1^2 … a_1^k |   | σ(x_1|e_1) σ(x_1|e_2) … σ(x_1|e_k) |
            | x_21 x_22 … x_2m 1 |   | a_2^1 a_2^2 … a_2^k |   | σ(x_2|e_1) σ(x_2|e_2) … σ(x_2|e_k) |
      D⋅A = | x_31 x_32 … x_3m 1 | ⋅ |   ⋮     ⋮   … ⋱ ⋮   | = | σ(x_3|e_1) σ(x_3|e_2) … σ(x_3|e_k) |
            |  ⋮    ⋮   …  ⋮   ⋮ |   | a_m^1 a_m^2 … a_m^k |   |      ⋮          ⋮     …      ⋮     |
            | x_n1 x_n2 … x_nm 1 |   |  θ_1   θ_2  …  θ_k  |   | σ(x_n|e_1) σ(x_n|e_2) … σ(x_n|e_k) |

where σ(x_i|e_l) = ∑_j a_j^l*x_ij + θ_l.
"""
@inline function pad_data(M::Matrix{<:Real})::Matrix{Float64}
  n, m = size(M)
  S = Matrix{Float64}(undef, n, m+1)
  @inbounds S[1:n,1:m] .= M
  fill!(view(S, :, m+1), 1)
  return S
end
export pad_data

"""Computes and sets sum node weights according to projections."""
function compute!(P::Vector{Projection}, D::Matrix{<:Real};
    S::Vector{Float64} = Vector{Float64}(undef, length(P)),
    M::Matrix{Float64} = Matrix{Float64}(undef, size(D, 1), length(P)),
    A::Matrix{Float64} = extract_a(P),
    W::Matrix{Float64} = Matrix{Float64}(undef, size(D, 1), length(P)))
  n, m = size(D)
  k = length(P)

  # S: k    , Vector of sums ∑_i w_i^p.
  # M: n × k, Matrix of computed sigmas.
  # W: n × k, Matrix of weights

  # M ← D⋅A to get ∑_i a_i⋅x_i + θ.
  mul!(M, D, A)
  # Apply σ to each element in M to get σ(∑_i a_i⋅x_i + θ).
  σ!(M)
  # Special case for the root: w_i = ∑_i σ(x_i|r)/n, where r are the proj. parameters for the root.
  S[1] = sum(view(M, :, 1))/n
  W[:,1] .= view(M, :, 1)/n
  P[1].w[] .= (1-S[1], S[1])
  # Compute view(W, :, 1:k-1) .* view(M, :, 2:k) ./ S
  for j ∈ 2:k
    @simd for i ∈ 1:n
      @inbounds W[i,j] = W[i,j-1] * M[i,j] / S[j-1]
    end
    S[j] = sum(view(W, :, j))
    P[j].w[] .= (1-S[j], S[j])
  end
  return nothing
end
export compute!

function prune!(r::Node, D::AbstractMatrix{<:Real}, n::Int; em_steps::Integer = 30,
    decay::Real = 0.975, batch::Integer = 500, validation::AbstractMatrix{<:Real} = D)
  L = SEM(r)
  I = shuffle!(collect(1:size(D, 1)))
  m = length(I)
  η = decay
  while L.steps < em_steps
    sid = rand(1:(m-batch))
    B = view(D, I[sid:(sid+batch-1)], :)
    update(L, B, η; verbose, validation)
    η *= decay
  end
  # TODO
  return nothing
end

function mini_em(C::Node, D::AbstractMatrix{<:Real}; steps::Integer = 10, decay::Real = 0.975,
    batch_size::Integer = 500, validation::AbstractMatrix{<:Real} = D, smoothing::Float64 = 0.05,
    learngaussians::Bool = false, minimumvariance::Float64 = 0.5)::Float64
  n = size(D, 1)
  L = SEM(C; gauss = learngaussians)
  indices = shuffle!(collect(1:n))
  η = decay
  while L.steps < steps
    sid = rand(1:(length(indices)-batch_size))
    batch = view(D, indices[sid:(sid+batch_size-1)], :)
    η *= decay
    update(L, batch, η, smoothing, learngaussians, minimumvariance; verbose = true, validation)
  end
  return pNLL(L.circ.V, L.circ.C, L.circ.L, validation)
end

function ensemble(D::AbstractMatrix{<:Real}, k::Integer; em_steps::Integer = 10, batch_size::Integer = 500,
    max_diff::Real = 0.1, bin::Bool = true, decay::Real = 0.975, validation::AbstractMatrix{<:Real} = D,
    strategy::Symbol = :em, smoothing::Float64 = 0.05, learngaussians::Bool = false,
    minimumvariance::Float64 = 0.5, learnvtree::Bool = true, kwargs...)::Node
  n, m = size(D)
  i = 1
  K = Vector{Node}(undef, k)
  LL = Vector{Float64}(undef, k)
  I = bin ? [(Indicator(u, 0), Indicator(u, 1)) for u ∈ 1:m] : Tuple{Indicator, Indicator}[]
  println("Learning initial components...")
  if bin D_df = DataFrame(D, :auto) end
  best = Threads.Atomic{Float64}(Inf)
  threshold = 1.0+max_diff
  @qthreads for i ∈ 1:k
    ll = 0
    # while (ll == 0) || (ll/best[] > threshold)
      println("  Retry ", i, "...\n  Learning vtree ", i, "...")
      if learnvtree
        V = bin ? learn_vtree(D_df; alg = rand((:bottomup, :topdown))) : learn_vtree_cont(D; alg = :topdown)
      else
        V = Vtree(m, :random)
      end
      println("  Learning circuit ", i, "...")
      K[i], _ = learn_rpp(D, V; bin, I, kwargs...)
      # ll = NLL(K[i], validation)
      # Threads.atomic_min!(best, ll)
    # end
    println("  Found suitable candidate for ", i)
  end
  if strategy != :stack
    # best = Inf
    for i ∈ 1:k
      nll = mini_em(K[i], D; steps = em_steps, decay, batch_size, validation, smoothing,
                    learngaussians, minimumvariance)
      LL[i] = nll
      # if nll < best best = nll end
    end
  end
  # R = findall(x -> x/best > threshold, LL)
  # println("Discarding worst components...")
  # while !isempty(R)
    # @qthreads for i ∈ R K[i], _ = learn_rpp(D, Vtree(m, :random); bin, I, kwargs...) end
    # for i ∈ R
      # nll = mini_em(K[i], D; steps = em_steps, decay, batch_size, validation)
      # LL[i] = nll
      # if nll < best best = nll end
    # end
    # R = findall(x -> x/best > threshold, LL)
    # println("  Candidates: ", LL)
    # println("  Retries: ", R)
  # end
  w = rand(k)
  S = Sum(K, w/sum(w))
  println("Computing strategy...")
  if strategy == :em learn_mix_em!(S, D; steps = 100)
  elseif strategy == :stack learn_mix_stack!(S, D; v = 10, steps = 10, validation)
  elseif strategy == :llw learn_mix_llw!(S, D)
  elseif strategy == :fullem
    mini_em(S, D; batch_size, validation, smoothing, learngaussians, minimumvariance, steps = 100)
  end
  return S
end
export ensemble

function learn_mix_stack!(P::Node, D::AbstractMatrix{<:Real}; v::Int = 10, steps::Int = 10,
    validation::AbstractMatrix{<:Real} = D)
  N, K = size(D, 1), length(P.children)
  N == 1 && return
  F = kfold(N, v)
  LL = Matrix{Float64}(undef, N, K)
  for j ∈ 1:v
    I, J = F[j]
    T, R = view(D, I, :), view(D, J, :)
    for i ∈ 1:K
      LL[I,i] .= mini_em(P.children[i], R; steps, validation)
    end
    println("Stacking fold ", j, '/', v, '.')
  end
  learn_mix_em!(P, D; steps, reuse = LL)
  for i ∈ 1:K
    mini_em(P.children[i], D; steps, validation)
  end
  return nothing
end
export learn_mix_stack!

function learn_mix_em!(P::Node, D::AbstractMatrix{<:Real}; steps::Int = 100, reuse::Union{AbstractMatrix{<:Real}, Nothing} = nothing)
  N, K = size(D, 1), length(P.children)
  ln_N = log(N)
  W = Matrix{Float64}(undef, N, K)
  N_k = Vector{Float64}(undef, K)
  ll = Vector{Float64}(undef, N)
  if isnothing(reuse)
    println("Pre-computing component log-likelihoods...")
    LL = Matrix{Float64}(undef, N, K)
    for i ∈ 1:K logpdf!(view(LL, :, i), P.children[i], D) end
  else LL = reuse end
  L_w = Vector{Float64}(undef, K)
  for j ∈ 1:steps
    L_w .= log.(P.weights)
    Threads.@threads for i ∈ 1:K
      W[:,i] .= LL[:,i] .+ L_w[i]
    end
    Threads.@threads for i ∈ 1:N
      W[i,:] .-= logsumexp(W[i,:])
    end
    Threads.@threads for i ∈ 1:K
      N_k[i] = logsumexp(W[:,i])
    end
    L_w = N_k .- ln_N
    P.weights .= exp.(L_w)
    Threads.@threads for i ∈ 1:K
      W[:,i] .= LL[:,i] .+ L_w[i]
    end
    Threads.@threads for i ∈ 1:N
      ll[i] = logsumexp(W[i,:])
    end
    println("EM Iteration ", j, "/", steps, ". Log likelihood ", sum(ll)/N)
  end
end
export learn_mix_em!

function learn_mix_llw!(E::Node, D::AbstractMatrix{<:Real})
    n = length(E.children)
    LL = Vector{Float64}(undef, n)
    for i ∈ 1:n @inbounds LL[i] = avgll(E.children[i], D) end
    W = exp.(LL .- maximum(LL))
    E.weights .= W ./ sum(W)
end
export learn_mix_llw!

"Bayesian Model Combination (BMC)."
mutable struct BMC
  E::Vector{Node}
  W::Vector{Float64}
  n::Int

  "Constructs a BMC with q*t combinations, each with n models."
  function BMC(n::Int, D::AbstractMatrix{<:Real}, q::Int, t::Int;
      α::Union{Vector{Float64}, Nothing} = nothing, reuse::Union{Vector{Node}, Nothing} = nothing,
      validation::AbstractMatrix{<:Real} = D, bin::Bool = true, smoothing::Float64 = 0.05,
      learngaussians::Bool = false, minimumvariance::Float64 = 0.5, kwargs...)::ModelComb
    if isnothing(α) α = ones(n) end
    K = q*t
    M = K*n
    D_df = DataFrame(D, :auto)
    m = size(D, 2)
    I = bin ? [(Indicator(u, 0), Indicator(u, 1)) for u ∈ 1:m] : Tuple{Indicator, Indicator}[]
    if isnothing(reuse)
      circs = Vector{Node}(undef, M)
      @qthreads for i ∈ 1:M
        circs[i], _ = learn_rpp(D, learn_vtree(D_df; alg = rand((:bottomup, :topdown))); bin,
                                I, kwargs...)
      end
      for i ∈ 1:M mini_em(circs[i], D; steps = 6, validation, smoothing, learngaussians, minimumvariance) end
    else circs = reuse end
    E = Vector{Node}(undef, K)
    dirichlet = Dirichlet(α)
    LL = Vector{Float64}(undef, K)
    W = Vector{Vector{Float64}}(undef, q)
    e = 1
    for i ∈ 1:t
      i_max, max_ll = -1, -Inf
      for j ∈ 1:q
        W[j] = rand(dirichlet)
        E[e] = Sum(circs[(e-1)*n+1:e*n], W[j])
        ll = avgll(E[e], validation)
        # Assume a uniform prior on the ensembles so that max p(e|D) = max p(D|e).
        if ll > max_ll i_max, max_ll = j, ll end
        LL[e] = ll
        println("BMC iteration ", e, '/', K, ": ", ll)
        e += 1
      end
      α .+= W[i_max]
    end
    LL .= exp.(LL .- maximum(LL))
    LL .= LL ./ sum(LL)
    return new(E, log.(LL), K)
  end
end
export BMC

function avgll(B::BMC, D::AbstractMatrix{<:Real})::Float64
  n, m = size(D, 1), B.n
  LL = Matrix{Float64}(undef, n, m)
  for i ∈ 1:B.n
    logpdf!(view(LL, :, i), B.E[i], D)
    @inbounds LL[:,i] .+= B.W[i]
  end
  S = Vector{Float64}(undef, n)
  Threads.@threads for i ∈ 1:n
    @inbounds S[i] = logsumexp(view(LL, i, :))
  end
  return sum(S)/n
end
@inline NLL(B::BMC, D::AbstractMatrix{<:Real})::Float64 = -avgll(B, D)
@inline Base.size(B::BMC)::Tuple{Int, Int, Int} = reduce(.+, size.(B.E))
