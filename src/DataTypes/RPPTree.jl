mutable struct Projection
  pa::Int
  w::Ref{Vector{Float64}}
  a::Vector{Float64} # ∑ aᵢ⋅xᵢ+θ
  θ::Float64
end
export Projection

@inline σ(x::Float64)::Float64 = 1.0/(1.0+exp(-x))
@inline function σ!(M::AbstractMatrix{<:Real})
  m = size(M, 2)
  # Apply σ to each column.
  Threads.@threads for i ∈ 1:m
    V = view(M, :, i)
    map!(σ, V, V)
  end
  return nothing
end
export σ, σ!
