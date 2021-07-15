using DataFrames

const Data = Union{AbstractMatrix{<:Real}, AbstractDataFrame}
const Example = Union{AbstractVector{<:Real}, DataFrameRow}

const Circuit = AbstractVector{<:Node}
export Circuit

"""
Implements a Circuit.

Assumes nodes are numbered topologically so that `nodes[1]` is the root (output) of the circuit and
nodes[end] is a leaf.

# Arguments

  - `nodes`: vector of nodes sorted in topological order (use `sort!(c)` after creating the circuit
    if this is not the case).

# Examples

```@example
c = Circuit([
  Sum([2, 3, 4], [0.2, 0.5, 0.3]),
  Product([5, 7]),
  Product([5, 8]),
  Product([6, 8]),
  Categorical(1, [0.6, 0.4]),
  Categorical(1, [0.1, 0.9]),
  Categorical(2, [0.3, 0.7]),
  Categorical(2, [0.8, 0.2]),
])
```
"""
@inline Circuit(N::AbstractVector{<:Node}; as_ref::Bool = false)::Circuit = as_ref ? N : copy(N)

"""
sort!(c::Circuit)

Sort nodes in topological order (with ties broken by breadth-first order) and modify node ids
accordingly.

Returns the permutation applied.
"""
function Base.sort!(c::Circuit)
  # First compute the number of parents for each node
  pa = zeros(length(c))
  for (i, n) in enumerate(c)
    if !isleaf(n)
      @inbounds for j in n.children
        pa[j] += 1
      end
    end
  end
  @assert count(isequal(0), pa) == 1 "Circuit has more than one parentless node"
  root = findfirst(isequal(0), pa) # root is the single parentless node
  # Kanh's algorithm: collect node ids in topological BFS order
  open = Vector{Int}()
  # visited = Set{Int}()
  closed = Vector{Int}() # topo bfs order
  push!(open, root) # enqueue root node
  while !isempty(open)
    n = popfirst!(open) # dequeue node
    # push!(visited, n)
    push!(closed, n)
    if !isleaf(c[n])
      # append!(open, ch for ch in c[n].children if !in(ch, visited) && !in(ch, open))
      @inbounds for j in c[n].children
        pa[j] -= 1
        if pa[j] == 0
          push!(open, j)
        end
      end
    end
  end
  @assert length(closed) == length(c)
  inverse = similar(closed) # inverse mapping
  @inbounds for i in 1:length(closed)
    inverse[closed[i]] = i
  end
  # permute nodes according to closed
  permute!(c, closed) # is this faster than c .= c[closed]?
  # now fix ids of children
  @inbounds for i in 1:length(c)
    if !isleaf(c[i])
      for (j, ch) in enumerate(c[i].children)
        c[i].children[j] = inverse[ch]
      end
    end
  end
  return closed
end

"""
layers(c::Circuit)

Returns list of node layers. Each node in a layer is a function of nodes in previous layers. This allows parallelization when computing with the circuit.
Assume nodes are topologically sorted (e.g. by calling `sort!(c)`).
"""
function layers(c::Circuit)
  layer = zeros(length(c)) # will contain the layer of each node
  layer[1] = 1 # root node is first layer
  for i in 1:length(c)
    # travesrse nodes in topological order
    if !isleaf(c[i])
      @inbounds for j in c[i].children
        # child j cannot be in same layer as i, for all i < j
        layer[j] = max(layer[j], layer[i] + 1)
      end
    end
  end
  # get number of layers
  nlayers = maximum(layer)
  # obtain layers (this is quadratic runtime -- can probably be improved to n log n)
  thelayers = Vector()
  @inbounds for l in 1:nlayers
    # collect all nodes in layer l
    thislayer = filter(i -> (layer[i] == l), 1:length(c))
    push!(thelayers, thislayer)
  end
  return thelayers
end
export layers

"""
    nodes(c::Circuit)

Collects the list of nodes in `c`.
"""
@inline nodes(c::Circuit) = c
export nodes

"""
Select nodes by topology
"""
@inline leaves(c::Circuit) = filter(n -> isa(n, Leaf), c)
@inline sums(c::Circuit) = filter(n -> isa(n, Sum), c)
@inline products(c::Circuit) = filter(n -> isa(n, Product), c)
@inline projections(c::Circuit) = filter(n -> isa(n, Projection), c)
@inline root(c::Circuit) = @inbounds c[1]
#TODO #variables(c::Circuit) = collect(1:c._numvars)
@inline children(c::Circuit, n) = @inbounds c[n].children
export leaves, sums, products, root, projections

"""
Return vector of weights associate to outgoing edges of (sum) node n.
"""
@inline weights(c::Circuit, n) = @inbounds c[n].weights
export weights

"""
    nparams(c::Circuit)

Computes the number of parameters in the circuit `c`.
"""
function nparams(c::Circuit)
  numparams = 0
  for i in 1:length(c)
    if issum(c[i])
      numparams += length(children(c, i))
    elseif isa(c[i], Categorical)
      numparams += length(c[i].values)
    elseif isa(c[i], Gaussian)
      numparams += 2
    end
  end
  return numparams
end
export nparams

"""
    vardims(c::Circuit)

Returns a dictionary mapping each variable index to its dimension (no. of values).
Assigns dimension = -1 for continuous variables.
"""
function vardims(c::Circuit)
  vdims = Dict{Int, Int}()
  for node in leaves(c)
    if isa(node, Indicator)
      dim = get(vdims, node.scope, 0)
      vdims[node.scope] = max(dim, convert(Int, node.value))
    elseif isa(node, Categorical)
      vdims[node.scope] = length(node.values)
    elseif isa(node, Gaussian)
      vdims[node.scope] = -1
    end
  end
  return vdims
end
export vardims

"""
    scope(c)

Returns the scope of circuit `c`, given by the scope of its root node.
"""
scope(c::Circuit)::AbstractVector = unique(collect(map(n -> scope(n), leaves(c))))
export scope

"""
    scopes(c::Circuit)

Returns an array of scopes for every node in the `c` (ordered by their index).
"""
function scopes(c::Circuit)
  sclist = Array{Array{Int}}(undef, length(c))
  for i in length(c):-1:1
    node = c[i]
    if isleaf(node)
      sclist[i] = Int[node.scope]
    elseif issum(node) # assume completeness
      sclist[i] = copy(sclist[node.children[1]])
    else # can probably be done more efficiently
      sclist[i] = Base.reduce(union, map(j -> sclist[j], node.children))
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
function project(c::Circuit, query::AbstractSet, evidence::AbstractVector)
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
function project2(c::Circuit, query::AbstractSet, evidence::AbstractVector)
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
    subcircuit(c::Circuit, node)

Returns the subcircuit of `c` rooted at given `node`.
"""
function subcircuit(c::Circuit, node::Integer)
  # Collect nodes in subcircuit
  nodes = Dict{UInt, Node}()
  stack = UInt[node]
  while !isempty(stack)
    n = pop!(stack)
    node = c[n]
    nodes[n] = deepcopy(node)
    if !isleaf(node)
      append!(stack, node.children)
    end
  end
  # println(nodes)
  # Reassign indices so that the become contiguous
  # Sorted list of remaining node ids -- position in list gives new index
  nodeid = Base.sort!(collect(keys(nodes)))
  idmap = Dict{UInt, UInt}()
  for (newid, oldid) in enumerate(nodeid)
    idmap[oldid] = newid
  end
  # println(idmap)
  # Now remap ids of children nodes
  for node in values(nodes)
    if !isleaf(node)
      for (i, ch) in enumerate(node.children)
        node.children[i] = idmap[ch]
      end
    end
  end
  return c = Circuit([nodes[i] for i in nodeid])
end
export subcircuit

"""
Modifies circuit so that each node has at most two children. Assume circuit is normalized.
"""
function binarize!(c::Circuit)
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

function scope_dict(C::Circuit)::Dict{UInt, BitSet}
  φ = Dict{UInt, BitSet}()
  for (i, n) ∈ Iterators.reverse(enumerate(C))
    j = UInt(i)
    if haskey(φ, j) continue end
    φ[j] = isleaf(n) ? BitSet(scope(n)) : reduce(∪, getindex.(Ref(φ), n.children))
  end
  return φ
end

"""
Verifies smoothness.
"""
function issmooth(C::Circuit; φ::Dict{UInt, BitSet} = Dict{UInt, BitSet}())::Bool
  assign = isempty(φ)
  for (i, n) ∈ Iterators.reverse(enumerate(C))
    j = UInt(i)
    if assign && haskey(φ, j) continue end
    if assign && isleaf(n) φ[j] = BitSet(scope(n))
    elseif !isleaf(n)
      ch = getindex.(Ref(φ), n.children)
      if issum(n) && !allequal(ch) return false end
      assign && (φ[j] = reduce(∪, ch))
    end
  end
  return true
end
export issmooth

"""
Verifies decomposability.
"""
function isdecomposable(C::Circuit; φ::Dict{UInt, BitSet} = Dict{UInt, BitSet}())::Bool
  assign = isempty(φ)
  for (i, n) ∈ Iterators.reverse(enumerate(C))
    j = UInt(i)
    if assign && haskey(φ, j) continue end
    if assign && isleaf(n)
      φ[j] = BitSet(scope(n))
    elseif !isleaf(n)
      ch = getindex.(Ref(φ), n.children)
      if isprod(n)
        Sc = BitSet(first(ch))
        for i ∈ 2:length(ch)
          if !isempty(ch[i] ∩ Sc) return false end
          union!(Sc, ch[i])
        end
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
function Base.isvalid(C::Circuit)::Bool
  φ = scope_dict(C)
  return issmooth(C; φ = φ) && isdecomposable(C; φ = φ)
end
export isvalid
