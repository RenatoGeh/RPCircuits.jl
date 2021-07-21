"""
Vtree Data Structures
"""
abstract type Vtree end
export Vtree

"""
Vtree leaf node
"""
struct VtreeLeaf <: Vtree
  var::UInt
end
export VtreeLeaf

"""
Vtree inner node
"""
mutable struct VtreeInner <: Vtree
  left::Vtree
  right::Vtree
  var::BitSet
end
export VtreeInner

function scope(V::Vtree)::BitSet
  passdown(L::VtreeLeaf)::BitSet = BitSet(L.var)
  function passdown(I::VtreeInner)::BitSet
    if isempty(I.var) I.var = passdown(I.left) ∪ passdown(I.right) end
    return I.var
  end
  return passdown(V)
end
export scope

Vtree(v::Integer)::VtreeLeaf = VtreeLeaf(v)
function Vtree(left::Vtree, right::Vtree; compute_scope::Bool = true)::VtreeInner
  V = VtreeInner(left, right, BitSet())
  if compute_scope scope(V) end
  return V
end

function Vtree(n::Integer, how::Symbol; scope::AbstractArray{<:Integer} = shuffle!(collect(1:n)))::Vtree
  if n == 1 return Vtree(scope[1]) end
  function passdown_right(S::AbstractArray{<:Integer})::VtreeInner
    L, R = S[1], length(S) == 2 ? S[2] : (@view S[2:end])
    return Vtree(passdown_right(L), passdown_right(R); compute_scope = false)
  end
  function passdown_left(S::AbstractArray{<:Integer})::VtreeInner
    L, R = length(S) == 2 ? S[2] : (@view S[2:end]), S[1]
    return Vtree(passdown_left(L), passdown_left(R); compute_scope = false)
  end
  function passdown(S::AbstractArray{<:Integer})::VtreeInner
    k = rand(1:length(S)-1)
    L, R = k == 1 ? S[1] : (@view S[1:k]), k == length(S)-1 ? S[end] : (@view S[k+1:end])
    return Vtree(passdown(L), passdown(R))
  end
  passdown(v::Integer)::VtreeLeaf = Vtree(v)
  passdown_left(v::Integer)::VtreeLeaf = Vtree(v)
  passdown_right(v::Integer)::VtreeLeaf = Vtree(v)
  if how == :left return passdown_left(scope)
  elseif how == :right return passdown_right(scope) end
  return passdown(scope)
end

@inline isleaf(v::Vtree)::Bool = v isa VtreeLeaf
Base.show(io::IO, v::Vtree) = print(io, isleaf(v) ? "Vtree($(v.var))" : "Vtree($(scope(v.left)), $(scope(v.right)))")

function Base.collect(V::Vtree)::Vector{Vtree}
  C = Vector{Vtree}()
  function passdown(V::Vtree)
    push!(C, V)
    if V isa VtreeInner passdown(V.left); passdown(V.right) end
  end
  passdown(V)
  return C
end
export collect

function leaves(V::Vtree)::Vector{VtreeLeaf}
  C = Vector{VtreeLeaf}()
  function passdown(V::Vtree)
    if V isa VtreeLeaf push!(C, V)
    else passdown(V.left); passdown(V.right) end
  end
  passdown(V)
  return C
end
export leaves
