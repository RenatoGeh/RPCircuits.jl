using DataFrames
using CSV

@testset "Evaluation of large tree-shaped circuits" begin
  C = Circuit(normpath("$(@__DIR__)/../assets/nltcs.spn"))

  T_df = DataFrame(CSV.File(normpath("$(@__DIR__)/../assets/nltcs.test.csv")))
  # println("Testset has $(size(testset,1)) instances, $(size(testset,2)) columns.")
  T = convert(Matrix, T_df) .+ 1
  @test logpdf(C, T) ≈ -19582.020235794218 #-21281.999990461
  @test plogpdf(C, T) ≈ -19582.020235794218 #-21281.999990461
  # @btime logpdf($C,$test)

  # C = Circuit(normpath("$(@__DIR__)/../assets/nltcs.learnspn.em.spn"))
  # @test logpdf(C,test) ≈ -22284.141587948394
  # @btime logpdf($C,$test)
end
