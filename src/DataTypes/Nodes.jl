"""
Node Data Structures

Implement a labeled sparse matrix.
"""
abstract type Node end
abstract type Inner <: Node end
export Node, Inner

"""
Sum node data type
"""
mutable struct Sum <: Inner
  children::Vector{Node}
  weights::Vector{Float64}
  Sum(ch::AbstractVector{<:Node}, w::AbstractVector{Float64}) = new(ch, w)
  Sum(n::Int) = new(Vector{Node}(undef, n), Vector{Float64}(undef, n))
end
export Sum

"""
Product node data type
"""
struct Product <: Inner
  children::Vector{Node}
  Product(ch::AbstractVector{<:Node}) = new(ch)
  Product(ch::Tuple) = new([c for c ∈ ch])
  Product(n::Int) = new(Vector{Node}(undef, n))
end
export Product

"""
Abstract leaf node type
"""
abstract type Leaf <: Node end
export Leaf

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

@inline Base.copy(i::Indicator)::Indicator = Indicator(i.scope, i.value)

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

@inline Base.copy(c::Categorical)::Categorical = Categorical(c.scope, c.values)

struct Bernoulli <: Leaf
  scope::UInt
  p::Float64
end
export Bernoulli

function (n::Bernoulli)(x::AbstractVector{<:Real})::Float64
  return isnan(x[n.scope]) ? 1.0 : Int(x[n.scope]) == 1 ? n.p : 1-n.p
end

@inline Base.copy(b::Bernoulli)::Bernoulli = Bernoulli(b.scope, b.p)

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

@inline Base.copy(g::Gaussian)::Gaussian = Gaussian(g.scope, g.mean, g.variance)

"""
Is this an inner node?
"""
@inline isinner(n::Node) = isa(n, Inner)
export isinner

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
@inline logpdf(n::Bernoulli, value::Integer) = value == 1 ? log(n.p) : log(1-n.p)
@inline logpdf(n::Bernoulli, value::Float64) = isnan(value) ? 0.0 : logpdf(n, Int(value))
export logpdf

"""
    maximum(node)

Returns the maximum value of the distribution
"""
@inline Base.maximum(n::Indicator) = 1.0
@inline Base.maximum(n::Categorical) = Base.maximum(n.values)
@inline Base.maximum(n::Gaussian) = 1 / sqrt(2 * π * n.variance)
@inline Base.maximum(n::Bernoulli) = n.p < 0.5 ? 1-n.p : n.p

"""
    argmax(node)

Returns the value at which the distribution is maximum
"""
@inline Base.argmax(n::Indicator) = n.value
@inline Base.argmax(n::Categorical) = Base.argmax(n.values)
@inline Base.argmax(n::Gaussian) = n.mean
@inline Base.argmax(n::Bernoulli) = n.p < 0.5 ? 0 : 1

"""
    scope(node)

Returns the scope of a leaf node
"""
@inline scope(n::Leaf) = n.scope
export scope

