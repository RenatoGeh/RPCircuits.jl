using DataFrames

const Data = Union{AbstractMatrix{<:Real}, AbstractDataFrame}
const Example = Union{AbstractVector{<:Real}, DataFrameRow}

"""
    layers(r::Node)::Vector{Vector{Node}}

Returns list of node layers. Each node in a layer is a function of nodes in previous layers. This allows parallelization when computing with the circuit.
"""
function layers(r::Node)::Vector{Vector{Node}}
  L = Vector{Vector{Node}}()
  Q = Tuple{Node, Int}[(r, 1)]
  V = Set{Node}()
  push!(V, r)
  while !isempty(Q)
    u, l = popfirst!(Q)
    if length(L) < l push!(L, Node[u])
    else push!(L[l], u) end
    if isinner(u)
      nl = l + 1
      for c ∈ u.children
        if c ∉ V
          push!(Q, (c, nl))
          push!(V, c)
        end
      end
    end
  end
  return L
  # c = nodes(r)
  # layer = Dict{Node, Int}(n => 0 for n ∈ c) # will contain the layer of each node
  # layer[r] = 1 # root node is first layer
  # for i in 1:length(c)
    # n = c[i]
    # # travesrse nodes in topological order
    # if !isleaf(n)
      # for j in n.children
        # # child j cannot be in same layer as i, for all i < j
        # layer[j] = max(layer[j], layer[n] + 1)
      # end
    # end
  # end
  # # get number of layers
  # nlayers = maximum(values(layer))
  # # obtain layers (this is quadratic runtime -- can probably be improved to n log n)
  # thelayers = Vector()
  # @inbounds for l in 1:nlayers
    # # collect all nodes in layer l
    # thislayer = filter(i -> (layer[i] == l), c)
    # push!(thelayers, thislayer)
  # end
  # return thelayers
end
export layers

"""
    nodes(r::Node; f::Union{Nothing, Function} = nothing)::Vector{Node}

Collects the list of nodes in `r`, traversing graph in topological order and storing nodes that
return `true` when function `f` is applied to them. If no function is given, stores every node.
"""
function nodes(r::Node; f::Union{Nothing, Function} = nothing)::Vector{Node}
  N = Node[]
  V = Set{Node}()
  Q = Node[r]
  while !isempty(Q)
    u = popfirst!(Q)
    if isnothing(f) || f(u) push!(N, u) end
    if isleaf(u) continue end
    for c ∈ u.children
      if c ∉ V push!(V, c); push!(Q, c) end
    end
  end
  return N
end
export nodes

function Base.foreach(f::Function, r::Node)
  V = Set{Node}()
  Q = Node[r]
  i = 0
  while !isempty(Q)
    u = popfirst!(Q)
    f(i, u)
    i += 1
    if isleaf(u) continue end
    for c ∈ u.children
      if c ∉ V
        push!(V, c)
        push!(Q, c)
      end
    end
  end
  return nothing
end

"""
Select nodes by topology
"""
@inline leaves(r::Node) = nodes(r; f = Base.Fix2(isa, Leaf))
@inline sums(r::Node) = nodes(r; f = Base.Fix2(isa, Sum))
@inline products(r::Node) = nodes(r; f = Base.Fix2(isa, Product))
@inline projections(r::Node) = nodes(r; f = Base.Fix2(isa, Projections))
@inline root(r::Node) = r

#TODO #variables(c::Circuit) = collect(1:c._numvars)
@inline children(r::Inner) = r.children
export leaves, sums, products, root, projections

"""
Return vector of weights associate to outgoing edges of (sum) node n.
"""
@inline weights(r::Sum) = c.weights
export weights

"""
    nparams(r::Node)

Computes the number of parameters in the circuit rooted at `r`.
"""
function nparams(r::Node)::Int
  n = 0
  Q = Node[r]
  V = Set{Node}()
  while !isempty(Q)
    u = popfirst!(Q)
    if issum(u) n += length(u.weights)
    elseif isa(u, Categorical) n += length(u.values)
    elseif isa(u, Gaussian) n += 2 end
    if isleaf(u) continue end
    for c ∈ u.children
      if c ∉ V
        push!(Q, c)
        push!(V, c)
      end
    end
  end
  return n
end
export nparams

"""
    vardims(c::Node)

Returns a dictionary mapping each variable index to its dimension (no. of values).
Assigns dimension = -1 for continuous variables.
"""
function vardims(r::Node)::Dict{Int, Int}
  N = leaves(r)
  vdims = Dict{Int, Int}()
  for n ∈ N
    if isa(n, Indicator)
      dim = get(vdims, n.scope, 0)
      vdims[n.scope] = max(dim, convert(Int, n.value))
    elseif isa(n, Categorical)
      vdims[n.scope] = length(n.values)
    elseif isa(n, Gaussian)
      vdims[n.scope] = -1
    end
  end
  return vdims
end
export vardims

"""
    scope(r::Node)::BitSet

Returns the scope of circuit rooted at `r`, given by the scope of its root node.
"""
scope(r::Node)::BitSet = union!(BitSet(), scope.(leaves(r))...)
export scope

"""
    scopes(r::Node)

Returns a dictionary of scopes for every node in the circuit rooted at `r`.
"""
function scopes(r::Node)::Dict{Node, BitSet}
  sclist = Dict{Node, BitSet}()
  N = nodes(r)
  for node in Iterators.reverse(N)
    if isleaf(node)
      sclist[node] = BitSet(node.scope)
    elseif issum(node) # assume completeness
      sclist[node] = copy(sclist[node.children[1]])
    else
      sc = BitSet()
      for c ∈ node.children union!(sc, sclist[c]) end
      sclist[node] = sc
    end
  end
  return sclist
end
export scopes

"""
    project(c::Circuit,query::AbstractSet,evidence::AbstractVector)

Returns the projection of a _normalized_ `c` onto the scope of `query` by removing marginalized subcircuits of marginalized variables and reducing subcircuits with fixed `evidence`.
Marginalized variables are the ones that are not in `query` and are assigned `NaN` in `evidence`.
The projected circuit assigns the same values to configurations that agree on evidence and marginalized variables w.r.t. to `evidence`.
The scope of the generated circuit contains query and evidence variables, but not marginalized variables.
"""
function project(r::Node, query::AbstractSet, evidence::AbstractVector)
  nodes = Dict{UInt, Node}()
  # evaluate circuit to collect node values
  vals = Dict{Node, Float64}()
  logpdf!(vals, c, evidence)
  # println(exp(vals[1]))
  # collect marginalized variables
  marginalized = Set(Base.filter(i -> (isnan(evidence[i]) && (i ∉ query)), 1:length(evidence)))
  # collect evidence variables
  evidvars = Set(Base.filter(i -> (!isnan(evidence[i]) && (i ∉ query)), 1:length(evidence)))
  # println(query)
  # println(evidvars)
  # println(marginalized)
  nscopes = scopes(c)
  newid = length(c) + 1 # unused id for new node
  stack = UInt[1]
  cache = Dict{UInt, UInt}() # cache indicator nodes created for enconding evidence
  while !isempty(stack)
    n = pop!(stack)
    node = c[n]
    if isprod(node)
      children = UInt[]
      for ch in node.children
        if !isempty(nscopes[ch] ∩ query)
          # subcircuit contains query variables, keep it
          push!(children, ch)
          push!(stack, ch)
        else # Replace node with subcircuit of equal value
          e_in_node = (evidvars ∩ nscopes[ch])
          if !isempty(e_in_node) # if there are evidence variables in node's scope
            # replace it with equivalent fragment
            if !haskey(nodes, ch) # only if we haven't already done this
              e = first(e_in_node)
              # e = Base.sort!(collect(e_in_node))[1]
              if !haskey(cache, e) # create indicator nodes
                nodes[newid] = Indicator(e, evidence[e])
                nodes[newid+1] = Indicator(e, evidence[e] + 1) # arbitrary different value
                cache[e] = newid
                newid += 2
              end
              nodes[ch] = Sum([cache[e], cache[e] + 1], [exp(vals[ch]), 1.0 - exp(vals[ch])])
            end
            push!(children, ch)
          end
        end
      end
      # TODO: Eliminate product nodes with single child
      # if length(children) == 1
      #     nodes[n] = c[children[1]]
      # else
      # nodes[n] = Product(children)
      # end
      nodes[n] = Product(children)
    else
      if issum(node)
        append!(stack, node.children)
      end
      nodes[n] = deepcopy(node)
    end
  end
  # Reassign indices so that the become contiguous
  # Sorted list of remaining node ids -- position in list gives new index
  nodeid = Base.sort!(collect(keys(nodes)))
  idmap = Dict{UInt, UInt}()
  for (newid, oldid) in enumerate(nodeid)
    idmap[oldid] = newid
  end
  # Now remap ids of children nodes
  for node in values(nodes)
    if !isleaf(node)
      # if length(node.children) < 2
      #     println(node)
      # end
      for (i, ch) in enumerate(node.children)
        node.children[i] = idmap[ch]
      end
    end
  end
  # println(idmap)
  c = Circuit([nodes[i] for i in nodeid])
  # println(c)
  sort!(c) # ensure nodes are topologically sorted (with ties broken by bfs-order)
  return c
end
export project

# Alternative implementation that maintains scopes of nodes
function project2(r::Node, query::AbstractSet, evidence::AbstractVector)
  nodes = Dict{UInt, Node}()
  # evaluate circuit to collect node values
  vals = Array{Float64}(undef, length(c))
  RPCircuits.logpdf!(vals, c, evidence)
  # println(exp(vals[1]))
  # collect marginalized variables
  marginalized = Set(Base.filter(i -> (isnan(evidence[i]) && (i ∉ query)), 1:length(evidence)))
  # collect evidence variables
  evidvars = Set(Base.filter(i -> (!isnan(evidence[i]) && (i ∉ query)), 1:length(evidence)))
  # println(query)
  # println(evidvars)
  # println(marginalized)
  nscopes = scopes(c)
  newid = length(c) + 1 # unused id for new node
  stack = UInt[1]
  cache = Dict{UInt, UInt}() # cache indicator nodes created for enconding evidence
  while !isempty(stack)
    n = pop!(stack)
    node = c[n]
    if isprod(node)
      children = UInt[]
      for ch in node.children
        if !isempty(nscopes[ch] ∩ query)
          # subcircuit contains query variables, keep it
          push!(children, ch)
          push!(stack, ch)
        else # Replace node with subcircuit of equal value
          e_in_node = (evidvars ∩ nscopes[ch])
          if !isempty(e_in_node) # if there are evidence variables in node's scope
            # replace it with equivalent fragment
            if !haskey(nodes, ch) # only if we haven't already done this
              nodes[ch] = Sum([newid, newid + 1], [exp(vals[ch]), 1.0 - exp(vals[ch])])
              nodes[newid] = lpn = Product([])
              nodes[newid+1] = rpn = Product([])
              newid += 2
              for e in e_in_node
                if !haskey(cache, e) # create indicator nodes
                  nodes[newid] = Indicator(e, evidence[e])
                  nodes[newid+1] = Indicator(e, evidence[e] + 1) # arbitrary different value
                  cache[e] = newid
                  newid += 2
                end
                push!(lpn.children, cache[e])
                push!(rpn.children, cache[e] + 1)
              end
            end
            push!(children, ch)
          end
        end
      end
      nodes[n] = Product(children)
    else
      if issum(node)
        append!(stack, node.children)
      end
      nodes[n] = deepcopy(node)
    end
  end
  # Reassign indices so that the become contiguous
  # Sorted list of remaining node ids -- position in list gives new index
  nodeid = Base.sort!(collect(keys(nodes)))
  idmap = Dict{UInt, UInt}()
  for (newid, oldid) in enumerate(nodeid)
    idmap[oldid] = newid
  end
  # Now remap ids of children nodes
  for node in values(nodes)
    if !isleaf(node)
      # if length(node.children) < 2
      #     println(node)
      # end
      for (i, ch) in enumerate(node.children)
        node.children[i] = idmap[ch]
      end
    end
  end
  # println(idmap)
  c = Circuit([nodes[i] for i in nodeid])
  # println(c)
  sort!(c) # ensure nodes are topologically sorted (with ties broken by bfs-order)
  return c
end

"""
Modifies circuit so that each node has at most two children. Assume circuit is normalized.
"""
function binarize!(r::Node)
  # TODO: refactor away from Circuit
  stack = UInt[1]
  newid = length(c) + 1
  while !isempty(stack)
    n = pop!(stack)
    node = c[n]
    if !isleaf(node)
      if length(node.children) > 2
        leftchild = node.children[1]
        if isprod(node)
          # add new product node
          newnode = Product(node.children[2:end])
        else
          # add new sum node
          w = node.weights[1]
          newnode = Sum(node.children[2:end], node.weights[2:end] ./ (1 - w))
          empty!(node.weights)
          push!(node.weights, w)
          push!(node.weights, 1 - w)
        end
        push!(c, newnode)
        empty!(node.children)
        push!(node.children, leftchild)
        push!(node.children, newid)
        newid += 1
      end
      append!(stack, node.children)
    end
  end
  # relabel node ids
  sort!(c)
  return nothing
end
export binarize!

"""
Verifies smoothness.
"""
function issmooth(r::Node; φ::Dict{Node, BitSet} = Dict{Node, BitSet}())::Bool
  assign = isempty(φ)
  N = nodes(r)
  for n ∈ Iterators.reverse(N)
    if assign && haskey(φ, n) continue end
    if assign && isleaf(n) φ[n] = BitSet(n.scope)
    elseif !isleaf(n)
      ch = getindex.(Ref(φ), n.children)
      if issum(n)
        Sc = first(ch)
        for i ∈ 2:length(ch)
          if ch[i] != S return false end
        end
        φ[n] = S
      else assign && (φ[n] = reduce(∪, ch)) end
    end
  end
  return true
end
export issmooth

"""
Verifies decomposability.
"""
function isdecomposable(r::Node; φ::Dict{Node, BitSet} = Dict{Node, BitSet}())::Bool
  assign = isempty(φ)
  N = nodes(r)
  for n ∈ Iterators.reverse(N)
    if assign && haskey(φ, n) continue end
    if assign && isleaf(n) φ[n] = BitSet(n.scope)
    elseif !isleaf(n)
      ch = getindex.(Ref(φ), n.children)
      if isprod(n)
        Sc = first(ch)
        for i ∈ 2:length(ch)
          if !isempty(ch[i] ∩ Sc) return false end
          union!(Sc, ch[i])
        end
        φ[j] = Sc
      end
      assign && (φ[j] = reduce(∪, ch))
    end
  end
  return true
end
export isdecomposable

"""
Verifies validity.
"""
function Base.isvalid(r::Node)::Bool
  φ = scope(r)
  N = nodes(r)
  for n ∈ Iterators.reverse(N)
    if haskey(φ, n) continue end
    if isleaf(n) φ[n] = BitSet(n.scope)
    elseif !isleaf(n)
      ch = getindex.(Ref(φ), n.children)
      Sc = first(ch)
      if issum(n)
        for i ∈ 2:length(ch)
          if ch[i] != S return false end
        end
        φ[n] = S
      else
        for i ∈ 2:length(ch)
          if !isempty(ch[i] ∩ Sc) return false end
          union!(Sc, ch[i])
        end
        φ[j] = Sc
      end
    end
  end
  return true
end
export isvalid

@inline Base.:(*)(w::Float64, n::Node)::Tuple{Float64, Node} = (w, n)
@inline Base.:(*)(n::Node, w::Float64)::Tuple{Float64, Node} = (w, n)
@inline function Base.:(+)(Ch::Tuple{Float64, Node}...)::Sum
  n = length(Ch)
  s = Sum(n)
  for i ∈ 1:n
    w, c = Ch[i]
    s.weights[i] = w
    s.children[i] = c
  end
  return s
end
@inline Base.:(*)(x::Node, y::Node)::Product = Product([x, y])
@inline function Base.:(*)(p::Node, q::Product)::Product push!(q.children, p); return q end
@inline Base.:(*)(p::Product, q::Node)::Product = q*p

@inline function Base.size(r::Node)::Tuple{Int, Int, Int}
  s, p, l = 0, 0, 0
  function f(i::Int, n::Node)
    if issum(n) s += 1
    elseif isprod(n) p += 1
    else l += 1 end
    return nothing
  end
  foreach(f, r)
  return s, p, l
end

@inline function Base.length(r::Node)::Int
  n = 0; foreach((_, x) -> n += 1, r)
  return n
end
