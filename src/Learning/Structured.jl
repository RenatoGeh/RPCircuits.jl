function learn_structured(S::Matrix{<:Real}; n_projs::Int = 3, max_height::Int = -1,
    min_examples::Int = 30, binarize::Bool = false, t_proj::Symbol = :max, trials::Int = 5,
    dense_leaves::Bool = false, r::Real = 2.0, c::Real = 1.0)::Circuit
  m = size(S, 1)
  if max_height < 0 max_height = floor(Int, sqrt(m)) end
  D = DataFrame(S, :auto)
  Sc = propertynames(D)
  C = Vector{Node}()
  rule = t_proj == :max ? (x -> max_rule(x, r, trials)) : (x -> sid_rule(x, c, trials))
  learn_mix_structured(C, D, Sc, n_projs, rule, max_height, min_examples, binarize, dense_leaves)
  return Circuit(C; as_ref = true)
end
export learn_structured

function partition_by(R::Union{Function, Nothing},
    M::AbstractMatrix{<:Real})::Tuple{Float64, Vector{Int}, Vector{Int}, Bool, Bool}
  if isnothing(R)
    n = size(M, 1)
    λ, k = 0.5, n ÷ 2
    I, J = collect(1:k), collect(k:n)
    return λ, I, J, true, true
  end
  λ = 0
  I, J = Vector{Int}(), Vector{Int}()
  r_I, r_J = nothing, nothing
  same_I, same_J = true, true
  m = size(M, 1)
  for (j, x) ∈ enumerate(eachrow(M))
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
  return λ/m, I, J, same_I, same_J
end

function factorize_sub(binarize::Bool, M::AbstractMatrix{<:Real}, I::Vector{Int}, Z::Vector{Symbol},
    V::Dict{Symbol, Int}, C::Vector{Node}, n_count::Int, P::Product)::Int
  n = size(M, 2)
  if binarize θ = vec(sum(M; dims = 1)) / length(I)
  else μ, σ = mean(M; dims = 1), std(M; dims = 1) end
  for j ∈ 1:n
    n_count += 1
    P.children[j] = n_count
    v = V[Z[j]]
    if binarize
      ⊥, ⊤ = Indicator(v, 0), Indicator(v, 1)
      B = Sum([n_count+1, n_count+2], [1-θ[j], θ[j]])
      append!(C, (B, ⊥, ⊤))
      n_count += 2
    else push!(C, Gaussian(v, μ[j], isnan(σ[j]) || σ[j] == 0 ? 0.05 : σ[j]*σ[j])) end
  end
  return n_count
end

function random_split(D::AbstractDataFrame, Sc::Vector{Symbol}, M::AbstractMatrix{<:Real};
    how::Symbol = :balanced)::Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol},
                                    AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}}
  n = length(Sc)
  k = how == :balanced ? n÷2 : rand(1:n-1)
  I, k = shuffle(1:n), n÷2
  I_1, I_2 = I[1:k], I[k+1:end]
  S_1, S_2 = view(D, :, I_1), view(D, :, I_2)
  A_1, A_2 = view(M, :, I_1), view(M, :, I_2)
  Z_1, Z_2 = Sc[I_1], Sc[I_2]
  return S_1, A_1, Z_1, S_2, A_2, Z_2
end

function factorize_random_split(S::AbstractDataFrame, Z::Vector{Symbol}, M::AbstractMatrix{<:Real},
    n_count::Int, binarize::Bool, C::Vector{Node}, pa::Product, n_projs::Int, V::Dict{Symbol, Int},
    Q::Vector{Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}, Sum}})::Int
  S_1, A_1, Z_1, S_2, A_2, Z_2 = random_split(S, Z, M)
  n_count += 1
  pa.children[1] = n_count
  if length(Z_1) == 1
    v = V[first(Z_1)]
    if binarize
      θ = mean(A_1)
      ⊥, ⊤ = Indicator(v, 0), Indicator(v, 1)
      B = Sum([n_count+1, n_count+2], [1-θ, θ])
      append!(C, (B, ⊥, ⊤))
      n_count += 2
    else
      μ, σ = mean(A_1), std(A_1)
      push!(C, Gaussian(v, μ, isnan(σ) || σ == 0 ? 0.05 : σ*σ))
    end
  else
    s = Sum(n_projs)
    push!(C, s)
    push!(Q, (S_1, A_1, Z_1, s))
  end
  n_count += 1
  pa.children[2] = n_count
  if length(Z_2) == 1
    v = V[first(Z_2)]
    if binarize
      θ = mean(A_2)
      ⊥, ⊤ = Indicator(v, 0), Indicator(v, 1)
      B = Sum([n_count+1, n_count+2], [1-θ, θ])
      append!(C, (B, ⊥, ⊤))
      n_count += 2
    else
      μ, σ = mean(A_2), std(A_2)
      push!(C, Gaussian(v, μ, isnan(σ) || σ == 0 ? 0.05 : σ*σ))
    end
  else
    s = Sum(n_projs)
    push!(C, s)
    push!(Q, (S_2, A_2, Z_2, s))
  end
  return n_count
end

function learn_mix_structured(C::Vector{Node}, D::DataFrame, Sc::Vector{Symbol}, n_projs::Int,
    t_rule::Function, max_height::Int, min_examples::Int, binarize::Bool, dense_leaves::Bool)
  c_weight = 1.0/n_projs
  push!(C, Sum(n_projs))
  Q = Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}, Sum}[(view(D, :, :), Matrix(D), Sc, first(C))]
  n_height, n_count = 0, 1
  V = Dict{Symbol, Int}(x => i for (i, x) ∈ enumerate(Sc))
  while !isempty(Q)
    S, M, Z, Σ = popfirst!(Q)
    n = ncol(S)
    K = Σ.children # components
    Σ.weights .= fill(c_weight, n_projs)
    n_height += 1
    for i ∈ 1:n_projs
      λ, I, J, same_I, same_J = partition_by(t_rule(M), M)
      factorize_pos_sub = (n_height > max_height) || (length(I) < min_examples) || same_I
      factorize_neg_sub = (n_height > max_height) || (length(J) < min_examples) || same_J
      pos_data, pos_mat = view(S, I, :), view(M, I, :)
      neg_data, neg_mat = view(S, J, :), view(M, J, :)
      n_count += 1
      K[i] = n_count
      P = Sum([0, 0], [λ, 1.0-λ])
      push!(C, P)
      if dense_leaves
        P.children[1] = n_count+1
        if factorize_pos_sub
          dense = sample_dense(pos_mat, collect(1:n), 1, 3, 2, 2, 2; offset = n_count,
                               binary = binarize, V = x -> V[Z[x]])
          n_count += length(dense)
          append!(C, dense)
          pos = first(dense)
        else
          pos = Product(2)
          n_count += 1
          push!(C, pos)
          n_count = factorize_random_split(pos_data, Z, pos_mat, n_count, binarize, C, pos, n_projs, V, Q)
        end
        P.children[2] = n_count+1
        if factorize_neg_sub
          dense = sample_dense(neg_mat, collect(1:n), 1, 3, 2, 2, 2; offset = n_count,
                               binary = binarize, V = x -> V[Z[x]])
          n_count += length(dense)
          append!(C, dense)
          neg = first(dense)
        else
          neg = Product(2)
          n_count += 1
          push!(C, neg)
          n_count = factorize_random_split(neg_data, Z, neg_mat, n_count, binarize, C, neg, n_projs, V, Q)
        end
      else
        pos = factorize_pos_sub ? Product(n) : Product(2)
        neg = factorize_neg_sub ? Product(n) : Product(2)
        n_count += 2
        append!(C, (pos, neg))
        if factorize_pos_sub n_count = factorize_sub(binarize, pos_mat, I, Z, V, C, n_count, pos)
        else n_count = factorize_random_split(pos_data, Z, pos_mat, n_count, binarize, C, pos, n_projs, V, Q) end
        if factorize_neg_sub n_count = factorize_sub(binarize, neg_mat, J, Z, V, C, n_count, neg)
        else n_count = factorize_random_split(neg_data, Z, neg_mat, n_count, binarize, C, neg, n_projs, V, Q) end
      end
    end
  end
end
