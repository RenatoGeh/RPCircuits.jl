@testset "Structural queries" begin
  # Smooth and decomposable.
  C = [Circuit([
    Sum([2, 3, 4], [0.2, 0.5, 0.3]), # 1
    Product([5, 7]),                 # 2
    Product([5, 8]),                 # 3
    Product([6, 8]),                 # 4
    Categorical(1, [0.6, 0.4]),      # 5
    Categorical(1, [0.1, 0.9]),      # 6
    Categorical(2, [0.3, 0.7]),      # 7
    Categorical(2, [0.8, 0.2]),      # 8
  ]), Circuit([
    Sum([2, 3], [0.3, 0.7]),      # 1
    Product([4, 5]),              # 2
    Product([6, 7]),              # 3
    Categorical(1, [0.3, 0.7]),   # 4 (D1)
    Sum([8, 9], [0.5, 0.5]),      # 5
    Sum([8, 9], [0.2, 0.8]),      # 6
    Categorical(1, [0.8, 0.2]),   # 7 (D2)
    Product([10, 11]),            # 8
    Product([12, 13]),            # 9
    Categorical(2, [0.4, 0.6]),   # 10 (D3)
    Sum([14, 15], [0.6, 0.4]),    # 11
    Sum([14, 15], [0.4, 0.6]),    # 12
    Categorical(2, [0.25, 0.75]), # 13 (D4)
    Categorical(3, [0.9, 0.1]),   # 14 (D5)
    Categorical(3, [0.42, 0.58]), # 15 (D6)
  ]), Circuit([
    Sum([2, 3], [0.4, 0.6]),     # 1
    Product([4, 5, 6]),          # 2
    Product([7, 8, 9]),          # 3
    Indicator(1, 2.0),           # 4
    Categorical(2, [0.3, 0.7]),  # 5
    Categorical(3, [0.4, 0.6]),  # 6
    Categorical(2, [0.8, 0.2]),  # 7
    Categorical(3, [0.9, 0.1]),  # 8
    Indicator(1, 1.0),           # 9
  ]), Circuit([
    Product([2, 3]),                    # 1 (10)
    Sum(
      [7, 6, 5, 4],
      exp.([-1.6094379124341003, -1.2039728043259361, -0.916290731874155, -2.3025850929940455]),
    ),     # 2 (9)
    Sum(
      [11, 10, 9, 8],
      exp.([-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -0.35667494393873245]),
    ), # 3 (8)
    Product([14, 12]), # 4
    Product([14, 13]), # 5
    Product([15, 12]), # 6
    Product([15, 13]), # 7
    Product([18, 16]), # 8
    Product([18, 17]), # 9
    Product([19, 16]), # 10
    Product([19, 17]), # 11
    Indicator(4, 1.0), # 12
    Indicator(4, 2.0), # 13
    Indicator(3, 1.0), # 14
    Indicator(3, 2.0), # 15
    Indicator(2, 1.0), # 16
    Indicator(2, 2.0), # 17
    Indicator(1, 1.0), # 18
    Indicator(1, 2.0), # 19
  ]), Circuit([
    Sum([2, 3, 4], [4 / 20, 9 / 20, 7 / 20]),# 1 Sum P(A, B) 0.2,0.5,0.3
    Product([5, 7]),    # 2 Prod P(A, B)=P(A)*P(B)
    Product([5, 8]),    # 3 Prod P(A,B)=P(A)*P(B)
    Product([6, 8]),    # 4 Prod P(A,B)=P(A)*P(B)
    Gaussian(1, 2, 18), # 5 Normal(A, mean=2, var=18)
    Gaussian(1, 11, 8), # 6 Normal(A, mean=11, var=8)
    Gaussian(2, 3, 10), # 7 Normal(B, mean=3, var=10)
    Gaussian(2, -4, 7), # 8 Normal(B, mean=-4, var=7)
  ])]

  # Decomposable but not smooth.
  D = [Circuit([Sum([2, 3, 4], [0.2, 0.5, 0.3]), Product([5, 6]), Product([7, 8]), Indicator(1, 0.0),
                Indicator(2, 0.0), Indicator(3, 0.0), Indicator(4, 0.0), Indicator(5, 0.0)]),
       Circuit([Product([2, 3, 4, 5]), Gaussian(1, 0, 1), Gaussian(2, 0, 1), Sum([6, 7], [0.2, 0.8]),
                Gaussian(3, 0, 1), Gaussian(4, 0, 1), Gaussian(5, 0, 1)]),
       Circuit([Sum([2, 3], [0.5, 0.5]), Categorical(1, [0.3, 0.7]), Categorical(2, [0.5, 0.5])])]

  # Smooth but not decomposable.
  S = [Circuit([Sum([2, 3, 4], [0.2, 0.5, 0.3]), Product([5, 6]), Product([7, 8]), Indicator(1, 0.0),
                Gaussian(1, 0.0, 1.0), Categorical(1, [0.1, 0.9]), Indicator(1, 1.0), Gaussian(1, 0.0, 1.0)]),
       Circuit([Product([2, 3, 4, 5]), Gaussian(1, 0, 1), Gaussian(3, 0, 1), Sum([6, 7], [0.2, 0.8]),
                Gaussian(3, 0, 1), Gaussian(3, 0, 1), Gaussian(3, 0, 1)]),
       Circuit([Product([2, 3]), Categorical(1, [0.3, 0.7]), Categorical(1, [0.5, 0.5])])]

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
