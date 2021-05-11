@testset "Simple DAG circuit" begin
  C = Circuit([
    Sum([2, 3, 4], [0.2, 0.5, 0.3]), # 1
    Product([5, 7]),                 # 2
    Product([5, 8]),                 # 3
    Product([6, 8]),                 # 4
    Categorical(1, [0.6, 0.4]),      # 5
    Categorical(1, [0.1, 0.9]),      # 6
    Categorical(2, [0.3, 0.7]),      # 7
    Categorical(2, [0.8, 0.2]),      # 8
  ])

  @test length(C) == 8
  @test ncircuits(C) == 3
  @test nparams(C) == 11
  sc = scope(C)
  # println("Scope: ", map(string,sc))
  @test (length(sc) == 2) && (1 in sc) && (2 in sc)

  @testset "Evaluate node scopes" begin
    scl = scopes(C)
    @test length(scl) == 8
    @test (length(scl[1]) == 2) && (1 in scl[1]) && (2 in scl[1])
    @test (length(scl[2]) == 2) && (1 in scl[2]) && (2 in scl[2])
    @test (length(scl[3]) == 2) && (1 in scl[3]) && (2 in scl[3])
    @test (length(scl[4]) == 2) && (1 in scl[4]) && (2 in scl[4])
    @test (length(scl[5]) == 1) && (1 in scl[5])
    @test (length(scl[6]) == 1) && (1 in scl[6])
    @test (length(scl[7]) == 1) && (2 in scl[7])
    @test (length(scl[8]) == 1) && (2 in scl[8])
  end

  results = [0.3, 0.15, 0.4, 0.15]
  @testset "Evaluation at $a,$b" for a in 1:2, b in 1:2
    @test C(a, b) ≈ results[2*(a-1)+b]
    @test logpdf(C, [a, b]) ≈ log(results[2*(a-1)+b])
    @test plogpdf(C, [a, b]) ≈ log(results[2*(a-1)+b])
  end

  @testset "Marginalization" begin
    @test C(1, NaN) ≈ 0.45
    @test C(NaN, 2) ≈ 0.3
    @test logpdf(C, [NaN, 2]) ≈ log(0.3)
    @test plogpdf(C, [NaN, 2]) ≈ log(0.3)
  end

  @testset "Sampling" begin
    x = rand(C)
    @test length(x) == 2
    #N = 10000
    N = 1000
    data = rand(C, N)
    counts = [0 0; 0 0]
    for i in 1:N
      counts[Int(data[i, 1]), Int(data[i, 2])] += 1
    end

    @testset "Verifing empirical estimates at $a,$b" for a in 1:2, b in 1:2
      ref = C([a, b])
      # println("S($a,$b) = $ref ≈ ", counts[a,b]/N)
      @test ref ≈ counts[a, b] / N atol = 0.1
    end
  end
end # end of DAG circuit testset

@testset "DAG circuit encoding HMM" begin
  HMM = Circuit([
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
  ])

  @test length(HMM) == 15
  @test ncircuits(HMM) == 8
  @test length(scope(HMM)) == 3

  @testset "Evaluate node scopes" begin
    scl = scopes(HMM)
    @test length(scl) == 15
    @test (length(scl[1]) == 3) && (1 in scl[1]) && (2 in scl[1]) && (3 in scl[1])
    @test (length(scl[5]) == 2) && (2 in scl[5]) && (3 in scl[5])
    @test (length(scl[6]) == 2) && (2 in scl[6]) && (3 in scl[6])
    @test (length(scl[8]) == 2) && (2 in scl[8]) && (3 in scl[8])
    @test (length(scl[9]) == 2) && (2 in scl[9]) && (3 in scl[9])
  end

  results = [
    0.11989139999999997,
    0.06615860000000003,
    0.29298060000000004,
    0.1709694,
    0.0708666,
    0.03658340000000001,
    0.1561014,
    0.08644860000000001,
  ]
  @testset "Evaluating HMM at $a,$b,$c" for a in 1:2, b in 1:2, c in 1:2
    v = HMM([a, b, c])
    @test v ≈ results[4*(a-1)+2*(b-1)+c]
    @test plogpdf(HMM, [a, b, c]) ≈ log(results[4*(a-1)+2*(b-1)+c])
  end

  # println("HMM() ≈ $(HMM([NaN,NaN,NaN]))")
  @test logpdf(HMM, [NaN, NaN, NaN]) ≈ 0.0
  @test plogpdf(HMM, [NaN, NaN, NaN]) ≈ 0.0
  # println("HMM(X1=1) ≈ $(HMM([1,NaN,NaN]))")
  @test logpdf(HMM, [1, NaN, NaN]) ≈ log(0.65)
  @test plogpdf(HMM, [1, NaN, NaN]) ≈ log(0.65)
  x = rand(HMM)
  @test length(x) == 3
  x[3] = NaN
  # project/prune network

  prunedHMM = project(HMM, Set([1]), x)
  @testset "Projection" for a in 1:2
    x[1] = a
    @test HMM(x) ≈ prunedHMM(x)
  end

  x = [1.0, 1.0, 2.0]
  prunedHMM = project(HMM, Set([1]), x)
  @testset "Projection" for a in 1:2
    x[1] = a
    @test HMM(x) ≈ prunedHMM(x)
  end
end # end of HMM testset

@testset "Selective SPN" begin
  selSPN = Circuit([
    Sum([2, 3], [0.4, 0.6]),     # 1
    Product([4, 5, 6]),          # 2
    Product([7, 8, 9]),          # 3
    Indicator(1, 2.0),           # 4
    Categorical(2, [0.3, 0.7]),  # 5
    Categorical(3, [0.4, 0.6]),  # 6
    Categorical(2, [0.8, 0.2]),  # 7
    Categorical(3, [0.9, 0.1]),  # 8
    Indicator(1, 1.0),           # 9
  ])

  @test length(selSPN) == 9
  @test ncircuits(selSPN) == 2
  @test length(scope(selSPN)) == 3

  results = [0.432, 0.048, 0.108, 0.012, 0.048, 0.072, 0.112, 0.168]
  @testset "Evaluating Sel SPN at $a,$b,$c" for a in 1:2, b in 1:2, c in 1:2
    v = selSPN(Float64[a, b, c])
    # println("SEL($a,$b,$c) = $v")
    @test v ≈ results[4*(a-1)+2*(b-1)+c]
    @test plogpdf(selSPN, [a, b, c]) ≈ log(results[4*(a-1)+2*(b-1)+c])
  end

  # println("- Selective SPN")
  value = selSPN([1, NaN, NaN])
  # println("S2(A=1) = $value")
  @test value ≈ 0.6
  @test logpdf(selSPN, [2, NaN, NaN]) ≈ log(0.4)
  @test plogpdf(selSPN, [2, NaN, NaN]) ≈ log(0.4)
  # println()
  x = rand(selSPN)
  @test length(x) == 3
end # end of selective SPN testset

@testset "Circuit encoding PSDD" begin
  # taken from https://github.com/UCLA-StarAI/Circuit-Model-Zoo/blob/master/psdds/little_4var.psdd
  PSDD = Circuit([
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
  ])

  @test length(PSDD) == 19
  @test ncircuits(PSDD) == 16
  @test length(scope(PSDD)) == 4

  results = [
    0.07,
    0.27999999999999997,
    0.20999999999999996,
    0.14,
    0.010000000000000005,
    0.04000000000000001,
    0.029999999999999995,
    0.02000000000000001,
    0.010000000000000005,
    0.04000000000000001,
    0.029999999999999995,
    0.02000000000000001,
    0.010000000000000005,
    0.04000000000000001,
    0.029999999999999995,
    0.02000000000000001,
  ]
  @testset "Evaluating PSDD at $a,$b,$c,$d" for a in 1:2, b in 1:2, c in 1:2, d in 1:2
    v = PSDD([a, b, c, d])
    # println("PSDD($a,$b,$c,$d) = $v")
    @test v ≈ results[8*(a-1)+4*(b-1)+2*(c-1)+d]
    @test plogpdf(PSDD, [a, b, c, d]) ≈ log(results[8*(a-1)+4*(b-1)+2*(c-1)+d])
  end

  @testset "Sampling" begin
    x = rand(PSDD)
    @test length(x) == 4
  end
end # end of PSDD test
