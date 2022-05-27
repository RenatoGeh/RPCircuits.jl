RPCircuits.jl
=============

A Julia package for efficiently building and using [Probabilistic Circuits][probcirc20]. Particularly,
*RPCircuits.jl* implements:

* Sparse and dense data structures for representing probabilistic circuits
* Multithread inference routines for likelihood and marginal computation   
* EM and gradient ascent parameter learning
* Linear-time structure learning for probabilistic circuits

**In particular**, this repository reproduces the results in "[Fast And Accurate
Learning of Probabilistic Circuits by Random Projections][link_article]" and in Chapter 5 of "[Scalable
Learning of Probabilistic Circuits][link_msc]".

[link_msc]: https://www.teses.usp.br/teses/disponiveis/45/45134/tde-23052022-122922/en.php
[link_article]: https://www.ime.usp.br/~renatolg/docs/geh21b_paper.pdf
[probcirc20]: http://starai.cs.ucla.edu/papers/ProbCirc20.pdf

## Quick Tutorial

The package is currently not registered in Julia's general registry. 
To use it, you need to clone this repository locally then either install it with 
```bash
julia -e 'using Pkg; Pkg.add("/path/to/RPCircuits")'
```

Alternatively, you can use the package without installing by mannually activating its environment. 
See [Installation](#installation) for more information.

As usual, to use the package add the following line to your Julia program:

```julia
using RPCircuits
```

### Manually building circuits

We begin by creating a simple circuit representing the polynomial
$$ f(x,y,\bar{x},\bar{y}) = (xy + x\bar{y} + \bar{x}y + \bar{x}\bar{y})/4 $$
over four indicator functions for 0/1-valued variables $X$ and $Y$.

In `RPCircuits`, variables are represented by integers `1,2,...` and their values as floats `0.0, 1.0` (irrespective of the variable being categorical or numerical). To create the above indicator functions, we can use:
```julia
julia> x, y, x̄, ȳ = Indicator(1, 1.0), Indicator(2, 1.0), Indicator(1, 0.0), Indicator(2, 0.0)
(indicator 1 1.0, indicator 2 1.0, indicator 1 0.0, indicator 2 0.0)
```
The `Indicator` function takes 2 positional arguments and a keyword argument. The first two
arguments are: the `index` (essentially the identifier) of the variable, and the `value` such that
the node outputs `true` when the variable is set to `value`. The keyword argument `tolerance` sets
a maximum discrepancy when evaluating the indicator at a given value (its default value is
`1e-6`.).

To evaluate any node for a given configuration of its input, we use function-like call. For example, 
to evaluate the indicators on various inputs, we use:
```julia
julia> x(1,1), y(0,0), x̄(0,1), ȳ(1,0)
(1.0, 0.0, 1.0, 1.0)
```
Note that the syntax above takes the corresponding value as defined in the creation of the `Indicator` node. 
For instance, the call `ȳ(1,0)` looks at the `2`nd argument (`0` in this case) and whether 
it matches the indicator `value` (`0.0` in this case). 

Next, we create four product nodes to represent the terms in the polynomial $f$:
```julia
julia> P1, P2, P3, P4 = Product([x,y]), Product([x,ȳ]), Product([x̄,y]), Product([x̄,ȳ])
(* 1 2, * 1 2, * 1 2, * 1 2)
```
The `Product` function takes a vector `v = [v1,..., vn]` of nodes as inputs and returns the respective product
node.

To finish building our representation of $f$, we create a weighted sum of the four product nodes with uniform weights :
```julia
julia> f = Sum([P1,P2,P3,P4], [0.25,0.25,0.25,0.25])
Circuit with 9 nodes (1 sum, 4 products, 4 leaves) and 2 variables:
  1 : + 1 0.25 2 0.25 3 0.25 4 0.25
  2 : * 1 2
  3 : * 1 2
  4 : indicator 1 0.0
  5 : * 1 2
  6 : indicator 2 0.0
  7 : * 1 2
  8 : indicator 2 1.0
  9 : indicator 1 1.0
```

The `Sum` functions takes two arguments: a vector of nodes (i.e., the `children`), and a vector of
the respective `weights`.

### Learning the parameters of circuits

One of the main purposes of this package is learning of Probabilistic Circuits from data. 
Parameter learning takes an initial probabilistic circuit and a dataset estimates the weight by maximum likelihood. 
We illustrate parameter learning with the simple circuit structure we built, but with different weights.
To modify the weights, simply change the `weights` vector of the root (sum) node:
```julia
f.weights .= [0.4, 0.3, 0.2, 0.1]
```
We will use that circuit as ground truth (the targer distribution) and use it to generate a data sample of size `N=1000`.
```julia
using Random

Random.seed!(42) # Locking seed

# Sample N samples from circuit f
N = 1_000
D = rand(f, N)
```

We now build a new circuit `g` whose structure is the same as `f` but whose weights are set
to uniform. Ideally, when learning from the dataset `D`, we want `g`'s weights to approximate that of `f`'s.
```julia
# Let's copy the same structure as f, but set its weights to a uniform
g = copy(f)
g.weights .= 0.25*ones(4)
```

Now that we have our distribution `g`, we are ready to fit it to the data `D`. We'll do this by
[Expectation-Maximization][em-spns], maximizing the log-likelihood (or equivalently minimizing the
negative log-likelihood, here denoted by `NLL`).

[em-spns]: https://ipa.iwr.uni-heidelberg.de/ipabib/Papers/Desana2016.pdf
```julia
println("Target model NLL = ", NLL(f, D))
println("Initial circuit NLL = ", NLL(g, D))
L = SEM(g) # Learner with EM algorithm
for i = 1:50
    update(L, D) # Iterations of the EM algorithm
end
println("Final circuit NLL = ", NLL(g, D))
println("          weights = ", round.(g.weights, digits = 2))
```
```julia
Target model NLL = 1.2905776805822866
Initial circuit NLL = 1.3862943611198644
Final circuit NLL = 1.2891629841331291
          weights = [0.38, 0.32, 0.19, 0.11]
```

## Installation

If you're not familiar with Julia's [REPL][repl_doc] or [Pkg][pkg_doc], we highly recommend having
a look at the linked Julia docs. To install all dependencies,

1. Install [Julia][julialang] version 1.7.2

2. Clone this repository
   ```
   git clone https://github.com/RenatoGeh/RPCircuits.jl
   ```

3. Start the [Julia REPL][repl_doc] using the command `julia`. Next, switch to package mode by
   entering the command `]`. Your screen should look like

   ```julia
   (@v1.7) pkg>
   ```

4. Within [Pkg mode][pkg_doc], activate the RPCircuits environment via 
   ```julia
   activate /path/to/RPCircuits
   ```
   and install all dependencies with
   ```julia
   instantiate
   ```

5. You may then locally install `RPCircuits` with
   ```julia
   add /path/to/RPCircuits
   ```

If step 5 is not taken, you will need to activate the package enviroment every time you start a new Julia session or 
run a program from shell. To activate the RPCircuits environment from a Julia program add the following to the beggining

```julia
using Pkg
Pkg.activate("/path/to/RPCircuits")
```

## Troubleshooting

If you have any problems installing the [`BlossomV`][blossomv] package dependency, try installing
the latest version of [`gcc`][gcc] **and** `g++` (more info [here][blossomv_build]). Once
`g++` has been installed, build `BlossomV`, through `build BlossomV` within Julia's `Pkg` mode.
Similarly, if you have any problems installing `HDF5`, `MAT` or `MLDatasets`, try building each
package one at a time.

**Note:** If you use an Arch based Linux distribution, you may want to install `base devel`. In
case you run into any other problems `Arpack`, `GaussianMixtures`, `HDF5`, we highly recommend
switching to `julia-bin` via AUR and reinstalling `RPCircuits`.

[julialang]: https://julialang.org/
[repl_doc]: https://docs.julialang.org/en/v1/stdlib/REPL/
[pkg_doc]: https://pkgdocs.julialang.org/v1
[blossomv]: https://github.com/mlewe/BlossomV.jl
[blossomv_build]: https://github.com/mlewe/BlossomV.jl#building
[gcc]: https://gcc.gnu.org/

## How to cite

To acknowledge this package, please cite:
```
@mastersthesis{geh22a,
  author = {Renato Lui Geh},
  title  = {Scalable Learning of Probabilistic Circuits},
  school = {University of S{\~{a}}o Paulo},
  type   = {Master's in Computer Science dissertation},
  year   = {2022},
  month  = {April},
  doi    = {10.11606/D.45.2022.tde-23052022-122922},
  url    = {https://doi.org/10.11606/D.45.2022.tde-23052022-122922}
}

@inproceedings{geh2021fast,
   title={Fast And Accurate Learning of Probabilistic Circuits by Random Projections},
   author={Renato Geh and Denis Mau{\'a}},
   booktitle={The 4th Workshop on Tractable Probabilistic Modeling},
   year={2021},
   url={https://www.ime.usp.br/~renatolg/docs/geh21b_paper.pdf}
}
```

