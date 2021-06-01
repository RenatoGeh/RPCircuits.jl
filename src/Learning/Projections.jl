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
  n = size(D, 1)
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
`select(f::Function, S::AbstractMatrix{<:Real}`

Selects rows which `f` returns true, returning rows which were selected and not selected.
"""
function select(f::Function, S::AbstractMatrix{<:Real})
  I, J = Vector{Int}(), Vector{Int}()
  n = size(S, 1)
  @inbounds for i ∈ 1:n
    if f(S[i,:]) push!(I, i)
    else push!(J, i) end
  end
  return view(S, I, :), view(S, J, :)
end

"""
`avgdiam(S::AbstractMatrix{<:Real}, μ::Vector{<:Real})`

Computes the average diameter Δ_A of a dataset given the mean of its instances.
"""
function avgdiam(S::AbstractMatrix{<:Real}, μ::Vector{<:Real})::Float64
  n = size(S, 1)
  Δ_A = 0
  Threads.@threads for i ∈ 1:n
    d = norm(S[i,:] - μ)
    Δ_A += d*d
  end
  return 2*Δ_A/n
end
@inline avgdiam(S::AbstractMatrix{<:Real})::Float64 = avgdiam(S, vec(mean(S; dims = 1)))

"""
`max_rule(S::AbstractMatrix{<:Real}, D::Matrix{Float64})`

Returns a random RPTree-Max rule given dataset `S` and distance matrix `D`.
"""
function max_rule(S::AbstractMatrix{<:Real}, D::AbstractMatrix{Float64}, r::Float64)::Function
  n, m = size(S)
  v = randunit(m)
  x = rand(1:n)
  y = farthest_from(D, x)
  δ = ((rand() * 2 - 1) * r * norm(S[x, :] - S[y, :])) / m
  Z = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    Z[i] = dot(S[i, :], v)
  end
  μ = median(Z) + δ
  return i::AbstractVector{<:Real} -> dot(i, v) <= μ
end
function max_rule(S::AbstractMatrix{<:Real}, r::Float64, trials::Int)::Function
  n, m = size(S)
  best_diff, best_split = Inf, nothing
  for j ∈ 1:trials
    v = randunit(m)
    x, y = rand(1:n), rand(1:n-1)
    if x == y y = n end
    δ = ((rand() * 2 - 1) * r * norm(S[x, :] - S[y, :])) / m
    Z = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
      Z[i] = dot(S[i, :], v)
    end
    μ = median(Z) + δ
    f = i::AbstractVector{<:Real} -> dot(i, v) <= μ
    P, Q = select(f, S)
    d_P, d_Q = avgdiam(P), avgdiam(Q)
    d = abs(d_P*d_P-d_Q*d_Q)
    if d < best_diff best_diff, best_split = d, f end
  end
  return best_split
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
`sid_rule(S::AbstractMatrix{<:Real}, D::Matrix{Float64}, c::Float64)`

Returns a random RPTree rule from the squared interpoint distance given datase `S`, distance matrix
`D` and distance constant `c`.
"""
function sid_rule(S::AbstractMatrix{<:Real}, D::AbstractMatrix{Float64}, c::Float64)::Function
  Δ, Δ_A = maximum(D), mean(D)
  n, m = size(S)
  if Δ*Δ <= c*(Δ_A*Δ_A)
    v = randunit(m)
    a = sort!([dot(v, x) for x ∈ eachrow(S)])
    μ_1, μ_2 = a[1], sum(a[2:end])
    i_min, c_min = -1, Inf
    for i ∈ 1:n-1
      δ_1, δ_2 = μ_1/i, μ_2/(n-i)
      p, q = a[1:i] .- δ_1, a[i+1:n] .- δ_2
      c = sum(p .* p) + sum(q .* q)
      if c < c_min
        i_min, c_min = i, c
      end
      μ_1 += a[i+1]
      μ_2 -= a[i+1]
    end
    θ = (a[i_min]+a[i_min+1])/2
    return x::AbstractVector{<:Real} -> dot(v, x) <= θ
  end
  me = vec(mean(S; dims = 1))
  Z = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    Z[i] = norm(S[i, :] - me)
  end
  μ = median(Z)
  return x::AbstractVector{<:Real} -> norm(x - me) <= μ
end
function sid_rule(S::AbstractMatrix{<:Real}, c::Float64, trials::Int)::Function
  n, m = size(S)
  x, y = rand(1:n), rand(1:n-1)
  if x == y y = n end
  Δ = norm(S[x,:] - S[y,:]); Δ *= Δ
  me = vec(mean(S; dims = 1))
  Δ_A = avgdiam(S, me)
  if Δ <= c*Δ_A
    best_diff, best_split = Inf, nothing
    for j ∈ 1:trials
      v = randunit(m)
      a = sort!([dot(v, x) for x ∈ eachrow(S)])
      s_μ_1, s_μ_2 = a[1]*a[1], sum(a[2:end] .* a[2:end])
      μ_1, μ_2 = a[1], sum(a[2:end])
      i_min, c_min = -1, Inf
      for i ∈ 1:n-1
        δ_1, δ_2 = μ_1/i, μ_2/(n-i)
        p, q = s_μ_1-μ_1*δ_1, s_μ_2-μ_2*δ_2
        c = p + q
        if c < c_min
          i_min, c_min = i, c
        end
        a_i = a[i+1]
        μ_1 += a_i
        μ_2 -= a_i
        s_μ_1 += a_i*a_i
        s_μ_2 -= a_i*a_i
      end
      θ = (a[i_min]+a[i_min+1])/2
      f = x::AbstractVector{<:Real} -> dot(v, x) <= θ
      P, Q = select(f, S)
      d_P, d_Q = avgdiam(P), avgdiam(Q)
      d = abs(d_P*d_P-d_Q*d_Q)
      if d < best_diff best_diff, best_split = d, f end
    end
    return best_split
  end
  Z = Vector{Float64}(undef, n)
  @inbounds for i ∈ 1:n
    Z[i] = norm(S[i, :] - me)
  end
  μ = median(Z)
  return x::AbstractVector{<:Real} -> norm(x - me) <= μ
end

"""
`learn_projections(S; c, n_projs, t_proj)`

Learns a Random Projection Circuit by sampling `n_projs` projections of type `t_proj` (either
`:mean` or `:max`), with constant `c` when `t_proj = :mean`.
"""
function learn_projections(
  S::AbstractMatrix{<:Real};
  c::Float64 = 1.0,
  r::Float64 = 2.0,
  n_projs::Int = 3,
  t_proj::Symbol = :max,
  max_height::Int = -1,
  min_examples::Int = 30,
  binarize::Bool = false,
  single_mix::Bool = false,
  no_dist::Bool = true,
  trials::Int = 5,
)::Circuit
  n, m = size(S)
  if max_height < 0 max_height = floor(Int, sqrt(m)) end
  C = Vector{Node}()
  if !no_dist
    upper = Matrix{Float64}(undef, n, n)
    Threads.@threads for i in 1:n
      @inbounds upper[i,i] = 0
      for j in i+1:n
        @inbounds upper[i, j] = norm(S[i, :] - S[j, :])
      end
    end
    D = Symmetric(upper, :U)
  else
    D = nothing
  end
  learn_func = single_mix ? learn_only_projections! : learn_projections!
  if t_proj == :mean
    learn_func(C, S, D, n_projs, max_height, min_examples, binarize, (x, y) -> mean_rule(x, y, c))
  elseif t_proj == :max
    learn_func(C, S, D, n_projs, max_height, min_examples, binarize, (x, y) -> max_rule(x, r, trials))
  else
    learn_func(C, S, D, n_projs, max_height, min_examples, binarize, (x, y) -> sid_rule(x, c, trials))
  end
  return Circuit(C; as_ref = true)
end
export learn_projections

function learn_only_projections!(
  C::Vector{Node},
  S::AbstractMatrix{<:Real},
  D::Union{AbstractMatrix{Float64}, Nothing},
  n_projs::Int,
  max_height::Int,
  min_examples::Int,
  binarize::Bool,
  t_rule::Function,
)
  n_count = 1
  n_height = 0
  n = size(S, 2)
  root = Sum(collect(2:n_projs+1), fill(1/n_projs, n_projs))
  push!(C, root)
  Q = Vector{Tuple{AbstractMatrix{<:Real}, Union{AbstractMatrix{Float64}, Nothing}, Node}}()
  for i ∈ 1:n_projs
    s = Sum(2)
    push!(C, s)
    push!(Q, (S, D, s))
  end
  n_count += n_projs
  while !isempty(Q)
    data, dists, Σ = popfirst!(Q)
    m = size(data, 1)
    n_height += 1
    R = t_rule(data, dists)
    λ = 0
    I, J = Vector{Int}(), Vector{Int}()
    for (j, x) in enumerate(eachrow(data))
      if R(x)
        λ += 1
        push!(I, j)
      else
        push!(J, j)
      end
    end
    λ /= m
    factorize_pos_sub = (n_height > max_height) || (length(I) < min_examples)
    factorize_neg_sub = (n_height > max_height) || (length(J) < min_examples)
    pos = factorize_pos_sub ? Product(n) : Sum(2)
    neg = factorize_neg_sub ? Product(n) : Sum(2)
    pos_data = view(data, I, :)
    neg_data = view(data, J, :)
    Σ.children[1], Σ.children[2] = n_count+1, n_count+2
    Σ.weights[1], Σ.weights[2] = λ, 1.0-λ
    n_count += 2
    append!(C, (pos, neg))
    if factorize_pos_sub
      if binarize
        θ = vec(sum(pos_data; dims = 1)) / length(I)
      else
        μ = mean(pos_data; dims = 1)
        σ = std(pos_data; dims = 1)
      end
      for j in 1:n
        n_count += 1
        pos.children[j] = n_count
        if binarize
          ⊥ = Indicator(j, 0)
          ⊤ = Indicator(j, 1)
          B = Sum([n_count+1, n_count+2], [1-θ[j], θ[j]])
          append!(C, (B, ⊥, ⊤))
          n_count += 2
        else
          push!(C, Gaussian(j, μ[j], isnan(σ[j]) || σ[j] == 0 ? 0.05 : σ[j]))
        end
      end
    else
      pos_dists = isnothing(dists) ? nothing : view(dists, I, I)
      push!(Q, (pos_data, pos_dists, pos))
    end
    if factorize_neg_sub
      if binarize
        θ = vec(sum(neg_data; dims = 1)) / length(J)
      else
        μ = mean(neg_data; dims = 1)
        σ = std(neg_data; dims = 1)
      end
      for j in 1:n
        n_count += 1
        neg.children[j] = n_count
        if binarize
          ⊥ = Indicator(j, 0)
          ⊤ = Indicator(j, 1)
          B = Sum([n_count+1, n_count+2], [1-θ[j], θ[j]])
          append!(C, (B, ⊥, ⊤))
          n_count += 2
        else
          push!(C, Gaussian(j, μ[j], isnan(σ[j]) || σ[j] == 0 ? 0.05 : σ[j]))
        end
      end
    else
      neg_dists = isnothing(dists) ? nothing : view(dists, J, J)
      push!(Q, (neg_data, neg_dists, neg))
    end
  end
  n_count += 1
end

function learn_projections!(
  C::Vector{Node},
  S::AbstractMatrix{<:Real},
  D::Union{AbstractMatrix{Float64}, Nothing},
  n_projs::Int,
  max_height::Int,
  min_examples::Int,
  binarize::Bool,
  t_rule::Function,
)
  c_weight = 1.0/n_projs
  n_count = 1
  n_height = 0
  n = size(S, 2)
  Q = Tuple{AbstractMatrix{<:Real}, Union{AbstractMatrix{Float64}, Nothing}, Node}[(S, D, Sum(n_projs))]
  while !isempty(Q)
    data, dists, Σ = popfirst!(Q)
    m = size(data, 1)
    n_height += 1
    push!(C, Σ)
    Ch = Σ.children
    Σ.weights .= fill(c_weight, n_projs)
    for i in 1:n_projs
      R = t_rule(data, dists)
      λ = 0
      I, J = Vector{Int}(), Vector{Int}()
      r_I, r_J = nothing, nothing
      same_I, same_J = true, true
      for (j, x) in enumerate(eachrow(data))
        if R(x)
          λ += 1
          push!(I, j)
          if isnothing(r_I) r_I = x
          elseif same_I same_I = (r_I == x) end
        else
          push!(J, j)
          if isnothing(r_J) r_J = x
          elseif same_J same_J = (r_J == x) end
        end
      end
      λ /= m
      n_count += 1
      factorize_pos_sub = (n_height > max_height) || (length(I) < min_examples) || same_I
      factorize_neg_sub = (n_height > max_height) || (length(J) < min_examples) || same_J
      pos = factorize_pos_sub ? Product(n) : Sum(n_projs)
      neg = factorize_neg_sub ? Product(n) : Sum(n_projs)
      pos_data = view(data, I, :)
      neg_data = view(data, J, :)
      Ch[i] = n_count
      # println(n_count, ", ", n_count+1, ", ", n_count+2, " ?= ", length(C)+1, ", ", length(C)+2, ", ", length(C)+3)
      P = Sum([n_count + 1, n_count + 2], [λ, 1.0-λ])
      append!(C, (P, pos, neg))
      n_count += 2
      if factorize_pos_sub
        if binarize
          θ = vec(sum(pos_data; dims = 1)) / length(I)
        else
          μ = mean(pos_data; dims = 1)
          σ = std(pos_data; dims = 1)
        end
        # println("m, n = ", m, ", ", n, "\n  I, J = ", length(I), ", ", length(J))
        for j in 1:n
          n_count += 1
          pos.children[j] = n_count
          if binarize
            ⊥ = Indicator(j, 0)
            ⊤ = Indicator(j, 1)
            B = Sum([n_count+1, n_count+2], [1-θ[j], θ[j]])
            append!(C, (B, ⊥, ⊤))
            n_count += 2
          else
            push!(C, Gaussian(j, μ[j], isnan(σ[j]) || σ[j] == 0 ? 0.05 : σ[j]))
          end
        end
      else
        pos_dists = isnothing(dists) ? nothing : view(dists, I, I)
        push!(Q, (pos_data, pos_dists, pos))
      end
      if factorize_neg_sub
        if binarize
          θ = vec(sum(neg_data; dims = 1)) / length(J)
        else
          μ = mean(neg_data; dims = 1)
          σ = std(neg_data; dims = 1)
        end
        # println("-: ", size(neg_data), "+: ", size(pos_data))
        # println("μ: ", μ, ", σ: ", σ)
        for j in 1:n
          n_count += 1
          neg.children[j] = n_count
          if binarize
            ⊥ = Indicator(j, 0)
            ⊤ = Indicator(j, 1)
            B = Sum([n_count+1, n_count+2], [1-θ[j], θ[j]])
            append!(C, (B, ⊥, ⊤))
            n_count += 2
          else
            push!(C, Gaussian(j, μ[j], isnan(σ[j]) || σ[j] == 0 ? 0.05 : σ[j]))
          end
        end
      else
        neg_dists = isnothing(dists) ? nothing : view(dists, J, J)
        push!(Q, (neg_data, neg_dists, neg))
      end
    end
    n_count += 1
  end
end
