using Pkg; Pkg.activate("..")
using RPCircuits

R, V, T = twenty_datasets("accidents"; as_df = false);
C = learn_projections(R; t_proj = :sid, binarize = true)
learner = SQUAREM(C)
initialize(learner)

while !converged(learner) || learner.steps < 10
  score, α = update(learner, R)
  testnll = NLL(C, V)
  trainnll = NLL(C, R)
  println("It: $(learner.steps) \t NLL: $trainnll \t held-out NLL: $testnll \t α: $α")
end

println(logpdf(C, T))
