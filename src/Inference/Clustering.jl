function memStep(circ::CCircuit, nvars::Int, x::AbstractVector{<:Real})::Tuple{Vector{Float64}, Vector{Float64}}
	N = circ.C
	nlayers = circ.L
	V = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, length(N))
	@inbounds for l in length(nlayers):-1:1
		Threads.@threads for i in nlayers[l]
			n = N[i]
			if isprod(n)
        nums = zeros(nvars)
        dems = zeros(nvars)
        for c in n.children
          (nn, dd) = V[c]
          nums += nn
          dems += dd
        end
        V[i] = (nums, dems)
			elseif issum(n)
        nums_children = zeros(Float64, nvars, length(n.children))
        dems_children = zeros(Float64, nvars, length(n.children))
        for (j, c) in enumerate(n.children)
          log_w = log(n.weights[j])
          (nn, dd) = V[c]
          (nn, dd) = (nn .+ log_w, dd .+ log_w)
          for k in 1:nvars
            nums_children[k, j] = nn[k]
            dems_children[k, j] = dd[k]
          end
        end
        nums = zeros(nvars)
        dems = zeros(nvars)
        for k in 1:nvars
          nums[k] = logsumexp(nums_children[k, :])
          dems[k] = logsumexp(dems_children[k, :])
        end
        V[i] = (nums, dems)
			else
        @assert n isa Gaussian
        lval = logpdf(n, x[n.scope])
        nums = fill(lval, nvars)
        dems = fill(lval, nvars)
        stdev = sqrt(n.variance)
        nums[n.scope] += log(n.mean) - 2 * log(stdev)
        dems[n.scope] -= 2 * log(stdev)
        V[i] = (nums, dems)
			end
		end
	end
	return V[first(first(nlayers))]
end

"""
Performs Modal EM in PC starting from point `x_0`. Returns the convergence
point after `max_iterations` iterations.

Constraints:
- Only works with Gaussian circuits.
- Gaussian means must be positive.
"""
function modalEM(
	root::Node,
	x_0::AbstractVector{<:Real},
  max_iterations::Int = 16,
)::AbstractVector{<:Real}
  circ = compile(CCircuit, root)
  x_r = copy(x_0)
  for _ in 1:max_iterations
    (nums, dems) = memStep(circ, length(x_0), x_r)
    x_r = exp.(nums - dems)
  end
  return x_r
end
export modalEM

"""
Simple logsumexp.

Source: https://github.com/probcomp/LogSumExp.jl/blob/main/src/LogSumExp.jl
"""
function logsumexp(a::AbstractArray{<:Real}; dims::Union{Integer, Dims, Colon}=:)
  m = maximum(a; dims=dims)
  return m + log.(sum(exp.(a .- m); dims=dims))
end
