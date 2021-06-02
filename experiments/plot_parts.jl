using Pkg; Pkg.activate("..")
using RPCircuits
using Plots
using LinearAlgebra
using Statistics
using NPZ
using Random

"Returns the orthogonal unit vector of another vector."
function ortho(x::Vector{<:Real})::Vector{<:Real}
  y = zeros(2)
  y[1], y[2] = x[2], -x[1]
  return y ./ norm(y)
end

function randunit(d::Int)::Vector{Float64}
  u = randn(d)
  u .= u ./ norm(u)
  return u
end

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

function select(f::Function, S::AbstractMatrix{<:Real})
  I, J = Vector{Int}(), Vector{Int}()
  n = size(S, 1)
  @inbounds for i ∈ 1:n
    if f(S[i,:]) push!(I, i)
    else push!(J, i) end
  end
  return view(S, I, :), view(S, J, :)
end

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

function max_rule(S::AbstractMatrix{<:Real}, r::Float64, trials::Int; v_d = nothing)
  n, m = size(S)
  best_diff, best_split = Inf, nothing
  best_μ, best_v = nothing, nothing
  for j ∈ 1:trials
    v = isnothing(v_d) ? randunit(m) : v_d
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
    d = abs(avgdiam(P)-avgdiam(Q))
    if d < best_diff best_diff, best_split, best_μ, best_v = d, f, μ, v end
  end
  return best_split, best_μ, best_v
end

function sid_rule(S::AbstractMatrix{<:Real}, c::Float64, trials::Int; v_d = nothing)
  n, m = size(S)
  x, y = rand(1:n), rand(1:n-1)
  if x == y y = n end
  Δ = norm(S[x,:] - S[y,:]); Δ *= Δ
  me = vec(mean(S; dims = 1))
  Δ_A = avgdiam(S, me)
  if Δ <= c*Δ_A
    best_diff, best_split = Inf, nothing
    best_μ, best_v = nothing, nothing
    for j ∈ 1:trials
      v = isnothing(v_d) ? randunit(m) : v_d
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
      d = abs(avgdiam(P)-avgdiam(Q))
      if d < best_diff best_diff, best_split, best_μ, best_v = d, f, θ, v end
    end
    return best_split, best_μ, best_v
  end
  Z = Vector{Float64}(undef, n)
  @inbounds for i ∈ 1:n
    Z[i] = norm(S[i, :] - me)
  end
  μ = median(Z)
  return x::AbstractVector{<:Real} -> norm(x - me) <= μ, μ, [-Inf, -Inf]
end

function median_rule(S::AbstractMatrix{<:Real}; v_d = nothing)
  n, m = size(S)
  v = isnothing(v_d) ? RPCircuits.randunit(m) : v_d
  Z = Vector{Float64}(undef, n)
  @inbounds for i ∈ 1:n
    Z[i] = dot(S[i,:], v)
  end
  μ = median(Z)
  return x -> dot(x, v) <= μ, μ, v
end

function save_data(D::AbstractMatrix{<:Real}, path::String)
  out = open(path, "w")
  n, m = size(D)
  for i ∈ 1:n
    for j ∈ 1:m
      write(out, "$(D[i,j]) ")
    end
    write(out, '\n')
  end
  close(out)
end

function plot_parts(D::AbstractMatrix{<:Real}, F::Vector, pre::String; level::Int = 1,
    Z = nothing, max_depth::Int = 2)
  if level > max_depth return end
  f = F[level]
  g, μ, v = f(D)
  X, Y = RPCircuits.select(g, D)
  scatter(X[:,1], X[:,2]; legend = false);
  scatter!(Y[:,1], Y[:,2]; legend = false);
  !isnothing(Z) && scatter!(Z[:,1], Z[:,2]; legend = false, seriescolor = :gray74)
  savefig("/tmp/$(pre)_$(level).pdf");
  Z = isnothing(Z) ? Y : vcat(Y, Z)
  save_data(X, "/tmp/$(pre)_blue_$(level).data")
  save_data(Y, "/tmp/$(pre)_red_$(level).data")
  save_data(Z, "/tmp/$(pre)_gray_$(level).data")
  open("/tmp/$(pre)_$(level).params", "w") do out write(out, "$(μ)\n$(v[1]) $(v[2])\n") end
  plot_parts(X, F, pre; level = level + 1, Z)
end

function plot_full_parts(D::AbstractMatrix{<:Real}, F::Vector, pre::String; max_depth::Int = 2)
  params = Vector()
  parts = Vector{AbstractMatrix{<:Real}}()
  function split(S::AbstractMatrix{<:Real}, level::Int)
    f = F[level]
    println(level)
    g, μ, v = f(S)
    X, Y = RPCircuits.select(g, S)
    push!(params, (μ, v))
    if level > max_depth-1
      append!(parts, (X, Y))
    else split(X, level + 1); split(Y, level + 2) end
  end
  split(D, 1)
  for (i, X) ∈ enumerate(parts)
    save_data(X, "/tmp/$(pre)_$(i)_full_parts.data")
  end
  out = open("/tmp/$(pre)_full.params", "w")
  for (μ, v) ∈ params
    write(out, "$(μ)\n$(v[1]) $(v[2])\n")
  end
  close(out)
  return parts
end

function plot_leaves(D::AbstractMatrix{<:Real}, F::Vector, pre::String; min_examples::Int = 100)
  P = Vector{AbstractMatrix{<:Real}}()
  function split(S::AbstractMatrix{<:Real}, l::Int)
    n = size(S, 1)
    if n <= min_examples push!(P, S); return end
    X, Y = RPCircuits.select((F[l])(S)[1], S)
    split(X, l+1)
    split(Y, l+1)
  end
  split(D, 1)
  scatter(legend = false, palette = :glasbey_bw_n256)
  for S ∈ P
    scatter!(S[:,1], S[:,2])
  end
  savefig("/tmp/$(pre)_leaves.pdf")
  out = open("/tmp/$(pre)_leaves.data", "w")
  for (i, S) ∈ enumerate(P)
    n = size(S, 1)
    for j ∈ 1:n
      write(out, "$(i) $(S[j,1]) $(S[j,2])\n")
    end
  end
  close(out)
end

function run_parts()
  Random.seed!(1)
  M = npzread("sin_rot.npy")
  a = [1.5, 1.0]
  a = a ./ norm(a)
  V = append!([ortho(a), a], [randunit(size(M, 2)) for i in 1:10])
  plot_parts(M, [x -> median_rule(x; v_d = v) for v in V], "median")
  plot_parts(M, [x -> max_rule(x, 1.0, 100; v_d = v) for v in V], "max")
  plot_parts(M, [x -> sid_rule(x, 10.0, 100; v_d = v) for v in V], "sid")
end

function run_leaves()
  Random.seed!(2)
  M = npzread("sin_rot.npy")
  a = [1.5, 1.0]
  a = a ./ norm(a)
  V = append!([ortho(a), a], [randunit(size(M, 2)) for i in 1:50])
  plot_leaves(M, [x -> median_rule(x; v_d = v) for v in V], "median")
  plot_leaves(M, [x -> max_rule(x, 0.5, 10; v_d = v) for v in V], "max")
  plot_leaves(M, [x -> sid_rule(x, 10.0, 10; v_d = v) for v in V], "sid")
end

function run_full_parts()
  Random.seed!(1)
  M = npzread("sin_rot.npy")
  a = [1.5, 1.0]
  a = a ./ norm(a)
  V = append!([ortho(a), a], [randunit(size(M, 2)) for i in 1:10])
  println(V)
  plot_full_parts(M, [x -> median_rule(x; v_d = v) for v in V], "median")
  plot_full_parts(M, [x -> max_rule(x, 1.0, 100; v_d = v) for v in V], "max")
  plot_full_parts(M, [x -> sid_rule(x, 10.0, 100; v_d = v) for v in V], "sid")
end

