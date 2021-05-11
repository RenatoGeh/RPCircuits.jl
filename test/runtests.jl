using Test

using RPCircuits

@testset "Creation, evaluation, sampling" begin
  include("test_discrete.jl")
  include("test_continuous.jl")
  include("test_eval.jl")
end

