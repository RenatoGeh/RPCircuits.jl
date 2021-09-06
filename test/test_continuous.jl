@testset "Defining and evaluating Gaussian circuits" begin
  G = gaussian_circuit()

  @test length(G) == 8
  @test size(G) == (1, 3, 4)

  @testset "Evaluation" begin
    res = G([11.0, -4.0])
    @test res ≈ 0.008137858167261642
    @test logpdf(G, [11.0, -4.0]) ≈ -4.811228258006023
  end

  @testset "Sampling" begin
    #N = 500000
    N = 1000
    data = rand(G, N)

    max = -Inf
    amax = 0
    for n in 1:N
      v = logpdf(G, view(data, n, :))  # G(data[n,:])
      if v > max
        max = v
        amax = n
      end
    end
    a, b = 11.0, -4.0
    ref = logpdf(G, [a, b]) #G([a,b])
    # println("max ln S($(data[amax,:])) = $max ≈ ln S($a,$b) = $ref")
  end
end
