using Test

using RPCircuits

include("setup.jl")

@testset "Creation, evaluation, sampling" begin
  include("test_discrete.jl")
  include("test_continuous.jl")
  include("test_eval.jl")
end

# include("test_vtree.jl")
include("test_queries.jl")

