function exppc(C::Node, L::Node)::Tuple{Float64, Dict{Node, Vector{Float64}}}
  E = Dict{Node, Vector{Float64}}()
  V = Set{Node}()
  function passdown(c::Node, l::Node)::Float64
    if c ∉ V E[c] = Vector{Float64}() end
    push!(V, c)
    if c isa Categorical
      if l isa Indicator
        p = c.values[Int(l.value)+1]
      else
        p = 1
      end
    else
      n, m = length(c.children), length(l.children)
      if c isa Product
        p = prod(passdown(c.children[i], l.children[i]) for i ∈ 1:n)
      elseif c isa Sum
        p = sum(c.weights[i]*passdown(c.children[i], l.children[j]) for i ∈ 1:n, j ∈ 1:m)
      end
    end
    push!(E[c], p)
    return p
  end
  return passdown(C, L), E
end
export exppc
