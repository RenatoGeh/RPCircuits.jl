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
  for i ∈ 1:n Δ_A += norm(S[i,:] - μ) end
  return 2*Δ_A/n
end
@inline avgdiam(S::AbstractMatrix{<:Real})::Float64 = avgdiam(S, vec(mean(S; dims = 1)))

function avgdiam_p(S::AbstractMatrix{<:Real}, μ::Vector{<:Real})::Float64
  n = size(S, 1)
  Δ_A = Vector{Float64}(undef, n)
  Threads.@threads for i ∈ 1:n @inbounds Δ_A[i] = norm(S[i,:] - μ) end
  return 2*sum(Δ_A)/n
end
@inline avgdiam_p(S::AbstractMatrix{<:Real})::Float64 = avgdiam_p(S, vec(mean(S; dims = 1)))

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
function max_rule(S::AbstractMatrix{<:Real}, r::Float64, trials::Int)::Union{Function, Nothing}
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
function max_rulep(S::AbstractMatrix{<:Real}, r::Float64 = 2, trials::Int = 10)::Union{Tuple{Vector{Float64}, Float64, Function}, Tuple{Nothing, Nothing, Nothing}}
  n, m = size(S)
  best_diff, best_f, a, θ = Inf, nothing, nothing, nothing
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
    if d < best_diff best_diff, best_f, a, θ = d, f, v, μ end
  end
  return a, θ, best_f
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
function sid_rule(S::AbstractMatrix{<:Real}, c::Float64, trials::Int)::Union{Function, Nothing}
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
      if isnothing(best_split) || (d < best_diff) best_diff, best_split = d, f end
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
function sid_rulep(S::AbstractMatrix{<:Real}, c::Float64, trials::Int = 10)::Union{Tuple{Vector{Float64}, Float64, Function}, Nothing}
  n, m = size(S)
  x, y = rand(1:n), rand(1:n-1)
  if x == y y = n end
  Δ = norm(S[x,:] - S[y,:]); Δ *= Δ
  me = vec(mean(S; dims = 1))
  Δ_A = avgdiam(S, me)
  if Δ <= c*Δ_A
    best_diff, best_f, best_a, best_θ = Inf, nothing, nothing, nothing
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
      if d <= best_diff best_diff, best_f, best_a, best_θ = d, f, v, θ end
    end
    return best_a, best_θ, best_f
  end
  # This is the case when most (if not all) of the datapoints are clustered into a single point.
  # Can't model this case only with hyperplanes. Return a nothing so that we can turn this into a
  # leaf node in the probabilistic circuit.
  return nothing, nothing, nothing
end

"""
`learn_projections(S; c, n_projs, t_proj)`

Learns a Random Projection Circuit by sampling `n_projs` projections of type `t_proj` (either
`:mean` or `:max`), with constant `c` when `t_proj = :mean`.
"""
function learn_projections(
  S::AbstractMatrix{<:Real};
  c::Real = 1.0,
  r::Real = 2.0,
  split::Symbol = :max,
  trials::Integer = 10,
  n_projs::Integer = 3,
  t_proj::Symbol = :max,
  max_height::Integer = typemax(Int),
  min_examples::Integer = 30,
  t_mix::Symbol = :all,
)::Node
  n, m = size(S)
  if max_height < 0 max_height = floor(Int, sqrt(n)) end
  f = split == :max ? (x -> max_rulep(x, r, trials)) : (x -> sid_rulep(x, c, trials))
  return t_mix == :all ? learn_projections_all!(S, n_projs, f, 0, max_height, min_examples) : learn_projections_single!(S, f, 0, max_height, min_examples)
end
export learn_projections

function learn_projections_all!(D::AbstractMatrix{<:Real}, n_projs::Integer, f::Function,
    height::Integer, max_height::Integer, min_examples::Integer)::Node
  n, m = size(D)
  if (height > max_height) || (n < min_examples)
    @label ff
    μ = mean(D; dims = 1)
    σ = std(D; dims = 1, mean = μ)
    return Product([Gaussian(i, μ[i], !(σ[i] > 0.03) ? 1e-3 : σ[i]^2) for i ∈ 1:m])
  end
  Ch = Vector{Node}(undef, n_projs)
  for i ∈ 1:n_projs
    a, _, g = f(D)
    if isnothing(a) @goto ff end
    A, B = select(g, D)
    k = size(A, 1)/n; w = [k, 1-k]
    Ch[i] = Sum([learn_projections_all!(A, n_projs, f, height + 1, max_height, min_examples),
                 learn_projections_all!(B, n_projs, f, height + 1, max_height, min_examples)], w)
  end
  return Sum(Ch, fill(1/n_projs, n_projs))
end

function learn_projections_single!(D::AbstractMatrix{<:Real}, f::Function, height::Integer,
    max_height::Integer, min_examples::Integer)::Node
  n, m = size(D)
  if (height > max_height) || (n < min_examples)
    @label ff
    μ = mean(D; dims = 1)
    σ = std(D; dims = 1, mean = μ)
    return Product([Gaussian(i, μ[i], !(σ[i] > 0.03) ? 1e-3 : σ[i]^2) for i ∈ 1:m])
  end
  a, _, g = f(D)
  if isnothing(a) @goto ff end
  A, B = select(g, D)
  k = size(A, 1)/n; w = [k, 1-k]
  return Sum([learn_projections_single!(A, f, height + 1, max_height, min_examples),
              learn_projections_single!(B, f, height + 1, max_height, min_examples)], w)
end
