using DataFrames
using CSV

@testset "Evaluation of large tree-shaped circuits" begin
  C = Circuit(normpath("$(@__DIR__)/../assets/nltcs.spn"))

  testset = DataFrame(CSV.File(normpath("$(@__DIR__)/../assets/nltcs.test.csv")))
  # println("Testset has $(size(testset,1)) instances, $(size(testset,2)) columns.")
  test = convert(Matrix, testset) .+ 1
  @test logpdf(C, test) ≈ -19582.020235794218 #-21281.999990461
  @test plogpdf(C, test) ≈ -19582.020235794218 #-21281.999990461
  # @btime logpdf($C,$test)

  # C = Circuit(normpath("$(@__DIR__)/../assets/nltcs.learnspn.em.spn"))
  # @test logpdf(C,test) ≈ -22284.141587948394
  # @btime logpdf($C,$test)
end
