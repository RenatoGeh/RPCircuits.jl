struct MAPResult
	lval::Float64
	evidence::Dict{UInt,Real}
end

function cmaxproduct(
	circ::CCircuit,
	x::AbstractVector{<:Real},
)::MAPResult
	N = circ.C
	nlayers = circ.L
	V = Vector{MAPResult}(undef, length(N))
	@inbounds for l in length(nlayers):-1:1
		Threads.@threads for i in nlayers[l]
			n = N[i]
			if isprod(n)
				lval = 0.0
				ev = Dict{UInt,Real}()
				for c in n.children
					lval += V[c].lval
					ev = merge(ev, V[c].evidence)
				end
				lval = isfinite(lval) ? lval : -Inf
				V[i] = MAPResult(lval, ev)
			elseif issum(n)
				res = MAPResult(n.weights[1] + V[n.children[1]].lval, V[n.children[1]].evidence)
        for (j, c) in enumerate(n.children)
					if n.weights[j] + V[c].lval > res.lval
						res = MAPResult(n.weights[j] + V[c].lval, V[c].evidence)
					end
				end
				V[i] = res
			else
				value = isnan(x[n.scope]) ? argmax(n) : x[n.scope]
				lval = logpdf(n, value)
				ev = Dict(n.scope => value)
				V[i] = MAPResult(lval, ev)
			end
		end
	end
	return V[first(first(nlayers))]
end
export cmaxproduct

"""
Computes MPE using Max-Product algorithm.

`x` is a vector that contains existing assignments for the variables.

This method does not support marginalized RVs at the moment. Given a variable
υ, make `x[υ] = NaN` to query an assignment for it.
"""
function maxproduct(
	root::Node,
	x::AbstractVector{<:Real},
)::MAPResult
	circ = compile(CCircuit, root)
	return cmaxproduct(circ, x)
end
export maxproduct
