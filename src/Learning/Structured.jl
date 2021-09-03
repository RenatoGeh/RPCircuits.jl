using LogicCircuits: Vtree, variables

function learn_structured(S::Matrix{<:Real}; n_projs::Int = 3, max_height::Int = -1,
    min_examples::Int = 30, binarize::Bool = false, t_proj::Symbol = :max, trials::Int = 5,
    dense_leaves::Bool = false, r::Real = 2.0, c::Real = 1.0, split_t::Symbol = :random,
    vtree::Union{Vtree, Nothing} = nothing, residuals::Bool = false, n_comps::Int = 0, p_res::Real = 0.25)::Circuit
  m = size(S, 1)
  if max_height < 0 max_height = floor(Int, sqrt(m)) end
  D = DataFrame(S, :auto)
  Sc = propertynames(D)
  rule = t_proj == :max ? (x -> max_rule(x, r, trials)) : (x -> sid_rule(x, c, trials))
  if residuals
    if !isnothing(vtree) K = Dict{Vtree, Vector{Sum}}()
    else K = Dict{BitSet, Vector{Sum}}() end
  else K = nothing end
  if n_comps > 1
    root = Sum(n_comps)
    root.weights .= fill(1.0/n_comps, n_comps)
    for i ∈ 1:n_comps
      M.children[i] = learn_mix_structured(C, D, Sc, n_projs, rule, max_height, min_examples,
                                           binarize, dense_leaves, vtree, split_t; save_layers = K)
    end
  else
    root = learn_mix_structured(C, D, Sc, n_projs, rule, max_height, min_examples, binarize,
                                dense_leaves, vtree, split_t; save_layers = K)
  end
  if residuals
    for v ∈ values(K)
      n = length(v)
      l = floor(Int, n_projs*n*p_res)
      for i ∈ 1:l
        a, b = rand(1:n), rand(1:n-1)
        if a == b b = n end
        push!(v[a].children, rand(v[b].children))
        u = length(v[a].children)
        v[a].weights = fill(1.0/u, u)
      end
    end
  end
  return root
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

function factorize_sub(binarize::Bool, M::AbstractMatrix{<:Real}, I::Vector{Int},
    Z::Vector{Symbol}, V::Dict{Union{Symbol, Int}, Union{Int, Symbol}};
    ⊥::Union{Vector{Indicator}, Nothing} = nothing, ⊤::Union{Vector{Indicator}, Nothing} = nothing)::Product
  n = size(M, 2)
  if binarize θ = vec(sum(M; dims = 1)) / length(I)
  else μ, σ = mean(M; dims = 1), std(M; dims = 1) end
  P = Product(n)
  for j ∈ 1:n
    v = V[Z[j]]
    if binarize P.children[j] = Sum([⊥[v], ⊤[v]], [1-θ[j], θ[j]])
    else P.children[j] = Gaussian(v, μ[j], isnan(σ[j]) || σ[j] == 0 ? 0.05 : σ[j]*σ[j]) end
  end
  return P
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

function projection_split(D::AbstractDataFrame, Sc::Vector{Symbol}, M::AbstractMatrix{<:Real},
    rule::Function)::Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol},
                           AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}}
  T = M'
  f = rule(T)
  P = propertynames(D)
  I, J = Vector{Int}(), Vector{Int}()
  for (j, x) ∈ enumerate(eachrow(T))
    if f(x) push!(I, j)
    else push!(J, j) end
  end
  Z_1, Z_2 = P[I], P[J]
  S_1, S_2 = view(D, :, Z_1), view(D, :, Z_2)
  A_1, A_2 = view(M, :, I), view(M, :, J)
  return S_1, A_1, Z_1, S_2, A_2, Z_2
end

function vtree_split(D::AbstractDataFrame, Sc::Vector{Symbol}, M::AbstractMatrix{<:Real}, vtree::Vtree,
    V::Dict{Union{Symbol, Int}, Union{Int, Symbol}})::Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol},
                                                            AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}}
  rev_indices = Dict{Symbol, Int}(x => i for (i, x) ∈ enumerate(propertynames(D)))
  L, R = variables(vtree.left), variables(vtree.right)
  a, b = length(L), length(R)
  I, J = Vector{Int}(undef, a), Vector{Int}(undef, b)
  Z_1, Z_2 = Vector{Symbol}(undef, a), Vector{Symbol}(undef, b)
  for (i, v) ∈ enumerate(L) x = V[v]; I[i], Z_1[i] = rev_indices[x], x end
  for (i, v) ∈ enumerate(R) x = V[v]; J[i], Z_2[i] = rev_indices[x], x end
  S_1, S_2 = view(D, :, Z_1), view(D, :, Z_2)
  A_1, A_2 = view(M, :, I), view(M, :, J)
  return S_1, A_1, Z_1, S_2, A_2, Z_2
end

function factorize_random_split(S::AbstractDataFrame, Z::Vector{Symbol}, M::AbstractMatrix{<:Real},
    binarize::Bool, n_projs::Int, V::Dict{Union{Symbol, Int}, Union{Int, Symbol}},
    Q::Vector{Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}, Sum, Union{Nothing, Vtree}, Int}},
    split_t::Symbol, rule::Function, n_height::Int; vtree::Union{Nothing, Vtree} = nothing,
    ⊥::Union{Vector{Indicator}, Nothing} = nothing, ⊤::Union{Vector{Indicator}, Nothing} = nothing)::Product
  if split_t == :random
    S_1, A_1, Z_1, S_2, A_2, Z_2 = random_split(S, Z, M)
  elseif split_t == :vtree
    @assert !isnothing(vtree)
    S_1, A_1, Z_1, S_2, A_2, Z_2 = vtree_split(S, Z, M, vtree, V)
  else
    @assert !isnothing(rule)
    S_1, A_1, Z_1, S_2, A_2, Z_2 = projection_split(S, Z, M, rule)
  end
  pa = Product(2)
  if length(Z_1) == 1
    v = V[first(Z_1)]
    if binarize
      θ = mean(A_1)
      c = Sum([⊥[v], ⊤[v]], [1-θ, θ])
    else
      μ, σ = mean(A_1), std(A_1)
      c = Gaussian(v, μ, isnan(σ) || σ == 0 ? 0.05 : σ*σ)
    end
  else
    c = Sum(n_projs)
    push!(Q, (S_1, A_1, Z_1, c, (isnothing(vtree) ? nothing : vtree.left), n_height))
  end
  pa.children[1] = c
  if length(Z_2) == 1
    v = V[first(Z_2)]
    if binarize
      θ = mean(A_2)
      c = Sum([⊥[v], ⊤[v]], [1-θ, θ])
    else
      μ, σ = mean(A_2), std(A_2)
      c = Gaussian(v, μ, isnan(σ) || σ == 0 ? 0.05 : σ*σ)
    end
  else
    c = Sum(n_projs)
    push!(Q, (S_2, A_2, Z_2, c, (isnothing(vtree) ? nothing : vtree.right), n_height))
  end
  pa.children[2] = c
  return pa
end

function learn_mix_structured(D::DataFrame, Sc::Vector{Symbol}, n_projs::Int, t_rule::Function,
    max_height::Int, min_examples::Int, binarize::Bool, dense_leaves::Bool, vtree::Union{Nothing, Vtree},
    split_t::Symbol; save_layers::Union{Dict{Vtree, Vector{Sum}}, Dict{BitSet, Vector{Sum}}, Nothing} = nothing)::Node
  c_weight = 1.0/n_projs
  root = Sum(n_projs)
  Q = Tuple{AbstractDataFrame, AbstractMatrix{<:Real}, Vector{Symbol}, Sum, Union{Nothing, Vtree}, Int}[(view(D, :, :), Matrix(D), Sc, C[length(C)], vtree, 0)]
  V = Dict{Union{Symbol, Int}, Union{Int, Symbol}}()
  for (i, x) ∈ enumerate(Sc) V[i], V[x] = x, i end
  ⊥, ⊤ = binarize ? indicators(1:length(Sc)) : (nothing, nothing)
  while !isempty(Q)
    S, M, Z, Σ, utree, n_height = popfirst!(Q)
    if !isnothing(save_layers)
      if !isnothing(utree)
        if !haskey(save_layers, utree) save_layers[utree] = Sum[Σ]
        else push!(save_layers[utree], Σ) end
      else
        sc = BitSet(V[x] for x in Z)
        if !haskey(save_layers, sc) save_layers[sc] = Sum[Σ]
        else push!(save_layers[sc], Σ) end
      end
    end
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
      if dense_leaves
        if factorize_pos_sub
          pos = sample_dense(pos_mat, collect(1:n), 1, 3, 2, 2, 2; binary = binarize, V = x -> V[Z[x]])
        else
          pos = factorize_random_split(pos_data, Z, pos_mat, n_count, binarize, C, n_projs, V, Q,
                                       split_t, t_rule, n_height; vtree = utree)
        end
        if factorize_neg_sub
          neg = sample_dense(neg_mat, collect(1:n), 1, 3, 2, 2, 2; offset = n_count,
                               binary = binarize, V = x -> V[Z[x]])
        else
          neg = factorize_random_split(neg_data, Z, neg_mat, n_count, binarize, C, n_projs, V, Q,
                                       split_t, t_rule, n_height; vtree = utree)
        end
      else
        if factorize_pos_sub pos = factorize_sub!(binarize, pos_mat, I, Z, V, C, n_count; ⊥, ⊤)
        else pos = factorize_random_split(pos_data, Z, pos_mat, n_count, binarize, C, n_projs, V,
                                        Q, split_t, t_rule, n_height; vtree = utree) end
        if factorize_neg_sub neg = factorize_sub(binarize, neg_mat, J, Z, V, C, n_count; ⊥, ⊤)
        else neg = factorize_random_split(neg_data, Z, neg_mat, n_count, binarize, C, n_projs, V,
                                        Q, split_t, t_rule, n_height; vtree = utree) end
      end
      K[i] = Sum([pos, neg], [λ, 1.0-λ])
    end
  end
  return root
end
