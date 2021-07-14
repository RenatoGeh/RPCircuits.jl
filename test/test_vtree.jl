@testset "Vtree" begin
  n = 5
  Sc = collect(1:n)
  V = Vtree(n, :left; scope = Sc)
  L, P = V, nothing
  for i ∈ 1:n-1
    @test L.right isa VtreeLeaf
    @test L.right.var == Sc[i]
    P = L
    L = L.left
  end
  @test P.left isa VtreeLeaf
  @test P.left.var == Sc[end]
  V = Vtree(n, :right; scope = Sc)
  L, P = V, nothing
  for i ∈ 1:n-1
    @test L.left isa VtreeLeaf
    @test L.left.var == Sc[i]
    P = L
    L = L.right
  end
  @test P.right isa VtreeLeaf
  @test P.right.var == Sc[end]
  V = Vtree(n, :random; scope = Sc)
  L = leaves(V)
  @test sort!(map(x -> x.var, L)) == Sc
end
