using LogicCircuits: Vtree, PlainVtreeLeafNode, variable, variables
using LinearAlgebra

BLAS.set_num_threads(Threads.nthreads())

@inline isleaf(v::Vtree)::Bool = isa(v, PlainVtreeLeafNode)

function learn_rpp!(S::SubArray{<:Real, 2}, V::Vtree, f::Function, pa_id::Int,
    P::Vector{Projection}, bin::Bool, min_examples::Int)::Node
  n, m = size(S)
  # Single variable. Return univariate distribution.
  if isleaf(V)
    u = variable(V)
    Z = view(S, :, u)
    if bin
      w = sum(Z)/n
      return Sum([Indicator(u, 0), Indicator(u, 1)], [1.0-w, w])
    end
    μ = mean(Z)
    s = std(Z; mean = μ)
    # This deals with NaNs, Infs and 0.
    if !(s > 0.2) s = 0.04 else s *= s end
    return Gaussian(u, μ, s)
  # Small dataset. Return fully factorized circuit.
  elseif n < min_examples
    @label ff
    ch = Vector{Node}(undef, m)
    U = S.indices[2]
    if bin
      W = sum(S; dims = 1) / n
      for i ∈ 1:m
        u, w = U[i], W[i]
        ch[i] = Sum([Indicator(u, 0), Indicator(u, 1)], [1.0-w, w])
      end
    else
      μ = mean(S; dims = 1)
      s = std(S; dims = 1, mean = μ)
      map!(x -> !(x > 0.2) ? 0.04 : x*x, s, s)
      for i ∈ 1:m ch[i] = Gaussian(U[i], μ, s) end
    end
    return Product(ch)
  end
  # Create projection and subsequent sum node.
  a, θ, g = f(S)
  if isnothing(a) @goto ff end
  # Negatives (A) and positives (B).
  A, B = select(g, S)
  # Initially give weights as the data proportion.
  k = size(A, 1)/n; w = [k, 1-k]
  # Be sure to pass w as a pointer (actually a reference) to the sum's weights so we can
  # efficiently compute during parameter learning.
  proj = Projection(pa_id, Ref(w), a, θ)
  push!(P, proj)
  # IDs are the indeces of the parent projections.
  id = length(P)
  # Scopes for subs and primes.
  # Sc_sub, Sc_prime = variables(V.left), variables(V.right)
  # U = S.indices[2]
  # I, J = findall(∈(Sc_sub), U), findall(∈(Sc_prime), U)
  # Negatives, each on the subs and primes.
  neg = Product([learn_rpp!(A#=view(A, :, I)=#, V.left, f, id, P, bin, min_examples),
                 learn_rpp!(A#=view(A, :, J)=#, V.right, f, id, P, bin, min_examples)])
  # Positives, each on the subs and primes.
  pos = Product([learn_rpp!(B#=view(B, :, I)=#, V.left, f, id, P, bin, min_examples),
                 learn_rpp!(B#=view(B, :, J)=#, V.right, f, id, P, bin, min_examples)])
  # Finally, return the resulting sum node, with same weights as the ones given to the projection.
  return Sum([neg, pos], w)
end

"""Learns a PC by projections and returns the root and a vector with all projections."""
function learn_rpp(D::Matrix{<:Real}, V::Vtree; split::Symbol = :max, c::Real = 1.0, r::Real = 2.0,
    trials::Int = 5, bin::Bool = true, min_examples::Int = 30)::Tuple{Node, Vector{Projection}}
  f = split == :max ? (x -> max_rulep(x, r, trials)) : (x -> sid_rulep(x, c, trials))
  P = Vector{Projection}()
  r = learn_rpp!(view(D, :, :), V, f, -1, P, bin, min_examples)
  return r, P
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
