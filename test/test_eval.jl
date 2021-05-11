@testset "Evaluation of large tree-shaped circuits" begin
  using DataFrames
  using CSV

  C = Circuit(zoo_spn("nltcs"))

  _, _, T_df = twenty_datasets("nltcs")
  T = Matrix{Float64}(T_df) .+ 1

  r = -19582.020235794218
  @test isapprox(logpdf(C, T), r; atol = 100)
  @test isapprox(plogpdf(C, T), r; atol = 100)
end
