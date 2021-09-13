@testset "Evaluation of large tree-shaped circuits" begin
  using DataFrames
  using CSV

  C = Circuit(zoo_spn("nltcs"))

  _, _, T = twenty_datasets("nltcs"; as_df = false)

  r = -19582.020235794218
  @test isapprox(logpdf(C, T), r; atol = 100)
  @test isapprox(plogpdf(C, T), r; atol = 100)
end
