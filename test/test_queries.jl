@testset "Structural queries" begin
  # Smooth and decomposable.
  C = [simple_circuit(), hmm(), selspn(), psdd(), gaussian_circuit()]

  # Decomposable but not smooth.
  D = [0.2*(Indicator(2, 0)*Indicator(3, 0))+0.5*(Indicator(4, 0)*Indicator(5, 0))+0.3*Indicator(1, 0),
       Gaussian(1, 0, 1)*Gaussian(2, 0, 1)*(0.2*Gaussian(4, 0, 1)+0.8*Gaussian(4, 0, 1))*Gaussian(3, 0, 1),
       0.5*Categorical(1, [0.3, 0.7])+0.5*Categorical(2, [0.5, 0.5])]

  # Smooth but not decomposable.
  S = [0.2*(Gaussian(1, 0, 1)*Categorical(1, [0.1, 0.9]))+0.5*(Indicator(1, 1)*Gaussian(1, 0, 1))+0.3*Indicator(1, 0),
       Gaussian(1, 0, 1)*Gaussian(3, 0, 1)*(0.2*Gaussian(3, 0, 1)+0.8*Gaussian(3, 0, 1))*Gaussian(3, 0, 1),
       Categorical(1, [0.3, 0.7])*Categorical(1, [0.5, 0.5])]

  @testset "Smoothness" begin
    for c ∈ C @test issmooth(c) end
    for c ∈ D @test !issmooth(c) end
    for c ∈ S @test issmooth(c) end
  end

  @testset "Decomposability" begin
    for c ∈ C @test isdecomposable(c) end
    for c ∈ D @test isdecomposable(c) end
    for c ∈ S @test !isdecomposable(c) end
  end

  @testset "Validity" begin
    for c ∈ C @test isvalid(c) end
    for c ∈ D @test !isvalid(c) end
    for c ∈ S @test !isvalid(c) end
  end
end
