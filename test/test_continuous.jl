@testset "Defining and evaluating Gaussian circuits" begin
  G = Circuit([
    Sum([2, 3, 4], [4 / 20, 9 / 20, 7 / 20]),# 1 Sum P(A, B) 0.2,0.5,0.3
    Product([5, 7]),    # 2 Prod P(A, B)=P(A)*P(B)
    Product([5, 8]),    # 3 Prod P(A,B)=P(A)*P(B)
    Product([6, 8]),    # 4 Prod P(A,B)=P(A)*P(B)
    Gaussian(1, 2, 18), # 5 Normal(A, mean=2, var=18)
    Gaussian(1, 11, 8), # 6 Normal(A, mean=11, var=8)
    Gaussian(2, 3, 10), # 7 Normal(B, mean=3, var=10)
    Gaussian(2, -4, 7), # 8 Normal(B, mean=-4, var=7)
  ])

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
