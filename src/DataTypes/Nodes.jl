"""
Node Data Structures

Implement a labeled sparse matrix.
"""
abstract type Node end

"""
Sum node data type
"""
struct Sum <: Node
  children::Vector{UInt}
  weights::Vector{Float64}
  # Sum() = new(Vector{UInt}(),Vector{Float64}())
  # Sum(children::Vector{<:Integer},weights::Vector{Float64}) = new(children,weights)
end
export Sum

"""
Product node data type
"""
struct Product <: Node
  children::Vector{UInt}
  #Product() = new(Vector{UInt}())
  #Product(children::Vector{<:Integer}) = new(children)
  #Product(children) = new(children)
end
export Product

"""
Abstract leaf node type
"""
abstract type Leaf <: Node end

"""
Indicator Function Node. Tolerance sets a maximum discrepancy when evaluating the node at a given
value. Its default value is 1e-6.
"""
struct Indicator <: Leaf
  scope::UInt
  value::Float64
  tolerance::Float64
  Indicator(scope::Integer, value::Float64) = new(scope, value, 1e-6)
  Indicator(scope::Integer, value::Integer) = new(scope, Float64(value), 1e-6)
end
export Indicator

function (n::Indicator)(x::AbstractVector{<:Real})::Float64
  return isnan(x[n.scope]) ? 1.0 : n.value ≈ x[n.scope] ? 1.0 : 0.0
end

"""
Univariate Categorical Distribution Node
"""
struct Categorical <: Leaf
  scope::UInt
  values::Vector{Float64}
end
export Categorical

function (n::Categorical)(x::AbstractVector{<:Real})::Float64
  return isnan(x[n.scope]) ? 1.0 : n.values[Int(x[n.scope])]
end

"""
Univariate Gaussian Distribution Node
"""
mutable struct Gaussian <: Leaf
  scope::UInt
  mean::Float64
  variance::Float64
end
export Gaussian

function (n::Gaussian)(x::AbstractVector{<:Real})::Float64
  return isnan(x[n.scope]) ? 1.0 :
         exp(-(x[n.scope] - n.mean)^2 / (2 * n.variance)) / sqrt(2 * π * n.variance)
end

"""
Is this a leaf node?
"""
@inline isleaf(n::Node) = isa(n, Leaf)
export isleaf

"""
Is this a sum node?
"""
@inline issum(n::Node) = isa(n, Sum)
export issum

"""
Is this a product node?
"""
@inline isprod(n::Node) = isa(n, Product)
export isprod

"""
    logpdf(node, value)

Evaluates leaf `node` at the given `value` in log domain.
"""
@inline logpdf(n::Indicator, value::Integer) =
  isnan(value) ? 0.0 : value == Int(n.value) ? 0.0 : -Inf
@inline logpdf(n::Indicator, value::Float64) =
  isnan(value) ? 0.0 : abs(value - n.value) < n.tolerance ? 0.0 : -Inf
@inline logpdf(n::Categorical, value::Integer) = log(n.values[value])
@inline logpdf(n::Categorical, value::Float64) = isnan(value) ? 0.0 : logpdf(n, Int(value))
@inline logpdf(n::Gaussian, value::Float64)::Float64 =
  isnan(value) ? 0.0 : (-(value - n.mean)^2 / (2 * n.variance)) - log(2 * π * n.variance) / 2
export logpdf

"""
    maximum(node)

Returns the maximum value of the distribution
"""
@inline Base.maximum(n::Indicator) = 1.0
@inline Base.maximum(n::Categorical) = Base.maximum(n.values)
@inline Base.maximum(n::Gaussian) = 1 / sqrt(2 * π * n.variance)

"""
    argmax(node)

Returns the value at which the distribution is maximum
"""
@inline Base.argmax(n::Indicator) = n.value
@inline Base.argmax(n::Categorical) = Base.argmax(n.values)
@inline Base.argmax(n::Gaussian) = n.mean

"""
    scope(node)

Returns the scope of a leaf node
"""
scope(n::Leaf) = n.scope
export scope
