# I/O functions

"""
Prints out Node content
"""
function Base.show(io::IO, n::Sum)
  print(io, "+")
  for i in 1:length(n.children)
    print(io, " $(n.children[i]) $(n.weights[i])")
  end
end

function Base.show(io::IO, n::Product)
  print(io, "*")
  for i in 1:length(n.children)
    print(io, " $(n.children[i])")
  end
end

Base.show(io::IO, p::Projection) =print(io, "⟂ $(p.pos) $(p.λ) $(p.neg) $(1-p.λ)")

function Base.show(io::IO, n::Indicator)
  return print(io, "indicator $(n.scope) $(n.value)")
end

function Base.show(io::IO, n::Categorical)
  print(io, "categorical $(n.scope)")
  for v in n.values
    print(io, " $v")
  end
end

function Base.show(io::IO, n::Gaussian)
  return print(io, "gaussian $(n.scope) $(n.mean) $(n.variance)")
end

"""
summary(io::IO, circ::Circuit)

Print out information about the network `circ` to stream `io`
"""
function Base.summary(io::IO, circ::Circuit)
  len = length(circ)
  lensum, lenprod, lenproj, lenleaves = 0, 0, 0, 0
  for i ∈ 1:len
    if issum(circ[i]) lensum += 1
    elseif isprod(circ[i]) lenprod += 1
    elseif isproj(circ[i]) lenproj += 1
    else lenleaves += 1 end
  end
  lenvars = length(scope(circ))
  return print(
    io,
    "Circuit with ",
    len,
    (len == 1 ? " node" : " nodes"),
    " (",
    lensum,
    (lensum == 1 ? " sum" : " sums"),
    ", ",
    lenprod,
    (lenprod == 1 ? " product" : " products"),
    ", ",
    lenproj,
    (lenproj == 1 ? " projection" : " projections"),
    ", ",
    lenleaves,
    (lenleaves == 1 ? " leaf" : " leaves"),
    ") and ",
    lenvars,
    (lenvars == 1 ? " variable" : " variables"),
  )
  # print(io, "Circuit with $(length(circ)) nodes ($(length(sums(circ))) sums, $(length(products(circ))) products, $(length(leaves(circ))) leaves), $(nparams(circ)) parameters, and $(length(scope(circ))) variables")
  # #println(io, summary(circ))
  # println(io, "Circuit with:")
  # println(io, "│\t$(length(circ)) nodes: $(length(sums(circ))) sums, $(length(products(circ))) products, $(length(leaves(circ))) leaves")
  # println(io, "│\t$(nparams(circ)) parameters")
  # println(io, "╰\t$(length(scope(circ))) variables")
  #println(io, "\tdepth = $(length(layers(circ)))")
end
# function Base.summary(circ::Circuit)
#     io = IOBuffer()
#     print(io, "Circuit with $(length(circ)) nodes ($(length(sums(circ))) sums, $(length(products(circ))) products, $(length(leaves(circ))) leaves), $(nparams(circ)) parameters, and $(length(scope(circ))) variables.")
#     String(take!(io))
# end

"""
    show(io::IO, circ::Circuit)

Print the nodes of the network `circ` to stream `io`
"""
function Base.show(io::IO, circ::Circuit)
  println(io, "Circuit(IOBuffer(\"\"\"# ", summary(circ))
  for (i, node) in enumerate(circ)
    println(io, i, " ", node)
  end
  return print(io, "\"\"\"))")
  # print(io, summary(circ))
end

function Base.show(io::IO, ::MIME"text/plain", circ::Circuit)
  # recur_io = IOContext(io)
  recur_io = IOContext(io, :SHOWN_SET => circ)
  limit::Bool = get(io, :limit, false)
  if !haskey(io, :compact)
    recur_io = IOContext(recur_io, :compact => true)
  end
  summary(io, circ)
  print(io, ":")
  # print(io, "\n  1: ", circ[1])
  # println(io, "Circuit with:")
  # println(io, "│\t$(length(circ)) nodes: $(length(sums(circ))) sums, $(length(products(circ))) products, $(length(leaves(circ))) leaves")
  # println(io, "│\t$(nparams(circ)) parameters")
  # println(io, "╰\t$(length(scope(circ))) variables")
  if limit
    sz = displaysize(io)
    rows, cols = sz[1] - 3, sz[2]
    rows < 4 && (print(io, " …"); return)
    cols -= 5 # Subtract the width of prefix "  " and separator " : "
    cols < 12 && (cols = 12) # Minimum widths of 2 for id, 4 for value
    rows -= 1 # Subtract the summary

    # determine max id width to align the output, caching the strings
    ks = Vector{String}(undef, min(rows, length(circ)))
    vs = Vector{String}(undef, min(rows, length(circ)))
    keylen = 0
    vallen = 0
    for (i, n) in enumerate(circ)
      i > rows && break
      ks[i] = sprint(show, i, context = recur_io, sizehint = 0)
      vs[i] = sprint(show, n, context = recur_io, sizehint = 0)
      keylen = clamp(length(ks[i]), keylen, cols)
      vallen = clamp(length(vs[i]), vallen, cols)
    end
    if keylen > max(div(cols, 2), cols - vallen)
      keylen = max(cld(cols, 3), cols - vallen)
    end
  else
    rows = cols = typemax(Int)
  end
  for (i, node) in enumerate(circ)
    print(io, "\n  ")
    if i == rows < length(circ)
      print(io, rpad("⋮", keylen), " : ⋮")
      # print(io, rpad("⋮", 2))
      break
    end
    if limit
      key = rpad(Base._truncate_at_width_or_chars(ks[i], keylen, "\r\n"), keylen)
    else
      key = sprint(show, i, context = recur_io, sizehint = 0)
    end
    print(recur_io, key)
    print(io, " : ")
    if limit
      val = Base._truncate_at_width_or_chars(vs[i], cols - keylen, "\r\n")
      print(io, val)
    else
      show(recur_io, n)
    end
  end
end

"""
    Circuit(filename::AbstractString; offset=0)::Circuit
    Circuit(io::IO=stdin; offset=0)::Circuit

Reads network from file. Assume 1-based indexing for node ids and values at indicator nodes. Set offset = 1 if these values are 0-based instead.
"""
function Circuit(filename::String; offset::Integer = 0)
  circ = open(filename) do file
    return circ = Circuit(file, offset = offset)
  end
  return circ
end

function Circuit(io::IO = stdin; offset::Integer = 0)
  # create dictionary of node_id => node (so they can be read in any order)
  nodes = Dict{UInt, Node}()
  # read and create nodes
  for line in eachline(io)
    # remove line break
    line = strip(line)
    # remove comments
    i = findfirst(isequal('#'), line)
    if !isnothing(i)
      line = line[1:i-1]
    end
    if length(line) > 0
      fields = split(line)
      if tryparse(Int, fields[1]) !== nothing
        nodeid = parse(Int, fields[1]) + offset
        nodetype = fields[2][1]
      else
        nodeid = parse(Int, fields[2]) + offset
        nodetype = fields[1][1]
      end
      if nodetype == '+'
        node = Sum(
          [parse(Int, ch) + offset for ch in fields[3:2:end]],
          [parse(Float64, w) for w in fields[4:2:end]],
        )
      elseif nodetype == '*'
        node = Product([parse(Int, id) + offset for id in fields[3:end]])
      elseif nodetype == 'c'
        varid = parse(Int, fields[3]) + offset
        node = Categorical(varid, [parse(Float64, value) for value in fields[4:end]])
      elseif nodetype == 'i' || nodetype == 'l'
        varid = parse(Int, fields[3]) + offset
        value = parse(Float64, fields[4]) + offset
        node = Indicator(varid, value)
      elseif nodetype == 'g'
        # TODO: read Gaussian leaves
        error("Reading of gaussian nodes is not implemented!")
      end
      nodes[nodeid] = node
    end
  end
  nodelist = Vector{Node}(undef, length(nodes))
  for (id, node) in nodes
    nodelist[id] = node
  end
  circ = Circuit(nodelist)
  sort!(circ) # ensure nodes are topologically sorted (with ties broken by bfs-order)
  return circ
end

"""
    save(circ,filename,[offset=0])

Writes network circ to file. Offset adds constant to node instances (useful for translating to 0 starting indexes).
"""
function save(circ::Circuit, filename::String, offset = 0)
  open(filename, "w") do io
    for i in 1:length(circ)
      println(io, "$i $(circ[i])")
    end
  end
end
export save

"""
    todot(io, son)

Prints out network structure in graphviz format
"""
function todot(io::IO, circ::Circuit)
  println(io, "digraph S {")
  for i in 1:length(circ)
    if isa(circ[i], Sum)
      if i == 1
        println(
          io,
          "n$i [shape=circle,rank=source,style=filled,color=\"#fed434\",label=\"+\",margin=0.05];",
        )
      else
        println(io, "n$i [shape=circle,style=filled,color=\"#fed434\",label=\"+\",margin=0.05];")
      end
    elseif isa(circ[i], Product)
      println(io, "n$i [shape=circle,style=filled,color=\"#b0db51\",label=\"×\",margin=0.05];")
    elseif isa(circ[i], LeafNode)
      println(
        io,
        "n$i [shape=circle,rank=sink,style=filled,color=\"#02a1d8\",label=\"X$(circ[i].scope)\",margin=0.05];",
      )
    end
    if !isa(circ[i], LeafNode)
      print(io, "n$i -> { ")
      for j in children(circ, i)
        print(io, "n$j; ")
      end
      println(io, "};")
    end
  end
  return println(io, "}")
end
export todot
