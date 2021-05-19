using LinearAlgebra
using Statistics

"""
`randunit(d::Int)`

Returns a uniformly distributed random unit vector.
"""
function randunit(d::Int)::Vector{Float64}
  u = randn(d)
  u .= u ./ norm(u)
  return u
end

"""
`furthest_from(D::Matrix{Float64}, x::Int)::Int`

Returns the farthest index from `x` using distance matrix `D`.
"""
function farthest_from(D::AbstractMatrix{<:Real}, x::Int)::Int
  n = size(D)[1]
  i_max, v_max = -1, -Inf
  for i in 1:n
    v = D[x, i]
    if v > v_max
      i_max, v_max = i, v
    end
  end
  return i_max
end

"""
`max_rule(S::AbstractMatrix{<:Real}, D::Matrix{Float64})`

Returns a random RPTree-Max rule given dataset `S` and distance matrix `D`.
"""
function max_rule(S::AbstractMatrix{<:Real}, D::AbstractMatrix{Float64})::Function
  n, m = size(S)
  v = randunit(m)
  x = rand(1:n)
  y = farthest_from(D, x)
  δ = ((rand() * 2 - 1) * 2 * norm(S[x,:] - S[y,:])) / m
  Z = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    Z[i] = dot(S[i,:], v)
  end
  μ = median(Z) + δ
  return i::AbstractVector{<:Real} -> dot(i, v) <= μ
end

"""
`mean_rule(S::AbstractMatrix{<:Real}, D::Matrix{Float64}, c::Float64)`

Returns a random RPTree-Mean rule given dataset `S`, distance matrix `D` and distance constant `c`.
"""
function mean_rule(S::AbstractMatrix{<:Real}, D::AbstractMatrix{Float64}, c::Float64)::Function
  Δ, Δ_A = maximum(D), mean(D)
  n, m = size(S)
  Z = Vector{Float64}(undef, n)
  if Δ * Δ <= c * (Δ_A * Δ_A)
    v = randunit(m)
    @inbounds for i in 1:n
      Z[i] = dot(S[i, :], v)
    end
    μ = median(Z)
    return x::AbstractVector{<:Real} -> dot(x, v) <= μ
  end
  m = vec(mean(S; dims = 1))
  @inbounds for i in 1:n
    Z[i] = norm(S[i, :] - m)
  end
  μ = median(Z)
  return x::AbstractVector{<:Real} -> norm(x - m) <= μ
end

"""
`learn_projections(S; c, n_projs, t_proj)`

Learns a Random Projection Circuit by sampling `n_projs` projections of type `t_proj` (either
`:mean` or `:max`), with constant `c` when `t_proj = :mean`.
"""
function learn_projections(S::AbstractMatrix{<:Real}; c::Float64 = 1.0, n_projs::Int = 3,
    t_proj::Symbol = :max, max_height::Int = 10, min_examples = 5)::Circuit
  n = size(S)[1]
  C = Vector{Node}()
  D = Matrix{Float64}(undef, n, n)
  Threads.@threads for i ∈ 1:n
    for j ∈ 1:n
      @inbounds D[i, j] = i == j ? 0 : norm(S[i, :] - S[j, :])
    end
  end
  if t_proj == :mean
    learn_projections!(C, S, D, n_projs, max_height, min_examples, (x, y) -> mean_rule(x, y, c))
  else
    learn_projections!(C, S, D, n_projs, max_height, min_examples, max_rule)
  end
  return Circuit(C; as_ref = true)
end
export learn_projections

function learn_projections!(
  C::Vector{Node},
  S::AbstractMatrix{<:Real},
  D::Matrix{Float64},
  n_projs::Int,
  max_height::Int,
  min_examples::Int,
  t_rule::Function,
)
  n_count = 1
  n_height = 0
  n = size(S)[2]
  Q = Tuple{AbstractMatrix{<:Real}, AbstractMatrix{Float64}, Node}[(S, D, Sum(n_projs))]
  while !isempty(Q)
    data, dists, Σ = popfirst!(Q)
    m = size(data)[1]
    n_height += 1
    push!(C, Σ)
    Ch = Σ.children
    @simd for i ∈ 1:n_projs @inbounds Σ.weights[i] = rand() end
    s = sum(Σ.weights)
    @simd for i ∈ 1:n_projs @inbounds Σ.weights[i] /= s end
    for i ∈ 1:n_projs
      R = t_rule(data, dists)
      λ = 0
      I, J = Vector{Int}(), Vector{Int}()
      for (j, x) ∈ enumerate(eachrow(data))
        if R(x)
          λ += 1
          push!(I, j)
        else push!(J, j) end
      end
      n_count += 1
      factorize_pos_sub = (n_height > max_height) || (length(I) < min_examples)
      factorize_neg_sub = (n_height > max_height) || (length(J) < min_examples)
      pos_sub = factorize_pos_sub ? Product(n) : Sum(n_projs)
      neg_sub = factorize_neg_sub ? Product(n) : Sum(n_projs)
      pos_data = view(data, I, :)
      neg_data = view(data, J, :)
      Ch[i] = n_count
      # println(n_count, ", ", n_count+1, ", ", n_count+2, " ?= ", length(C)+1, ", ", length(C)+2, ", ", length(C)+3)
      P = Projection(n_count + 1, n_count + 2, λ / m, R)
      append!(C, (P, pos_sub, neg_sub))
      n_count += 2
      if factorize_pos_sub
        μ = mean(pos_data; dims = 1)
        σ = std(pos_data; dims = 1)
        # println("m, n = ", m, ", ", n, "\n  I, J = ", length(I), ", ", length(J))
        for j ∈ 1:n
          push!(C, Gaussian(j, μ[j], σ[j]))
          n_count += 1
          pos_sub.children[j] = n_count
        end
      else
        pos_dists = view(dists, I, I)
        push!(Q, (pos_data, pos_dists, pos_sub))
      end
      if factorize_neg_sub
        μ = mean(neg_data; dims = 1)
        σ = std(neg_data; dims = 1)
        # println("-: ", size(neg_data), "+: ", size(pos_data))
        # println("μ: ", μ, ", σ: ", σ)
        for j ∈ 1:n
          push!(C, Gaussian(j, μ[j], σ[j]))
          n_count += 1
          neg_sub.children[j] = n_count
        end
      else
        neg_dists = view(dists, J, J)
        push!(Q, (neg_data, neg_dists, neg_sub))
      end
    end
    n_count += 1
  end
end
