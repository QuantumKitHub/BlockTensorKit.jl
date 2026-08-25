# Sparsity
# --------
nonzero_pairs(t::AbstractBlockTensorMap) = nonzero_pairs(parent(t))
nonzero_keys(t::AbstractBlockTensorMap) = nonzero_keys(parent(t))
nonzero_values(t::AbstractBlockTensorMap) = nonzero_values(parent(t))
nonzero_length(t::AbstractBlockTensorMap) = nonzero_length(parent(t))

nonzero_values(A::AbstractArray) = values(A)
nonzero_keys(A::AbstractArray) = eachindex(IndexCartesian(), A)
nonzero_pairs(A::AbstractArray) = pairs(IndexCartesian(), A)
nonzero_length(A::AbstractArray) = length(A)

issparse(t::AbstractTensorMap) = false
issparse(t::TensorKit.AdjointTensorMap) = issparse(parent(t))

function sparse(t::AbstractBlockTensorMap)
    issparse(t) && return t
    return SparseBlockTensorMap(t)
end

"""
    dropzeros!(t::AbstractBlockTensorMap)

Remove the tensor entries of a blocktensor that have norm 0. Only applicable to sparse blocktensors.
"""
function dropzeros!(t::AbstractBlockTensorMap)
    issparse(t) || return t
    for (k, v) in nonzero_pairs(t)
        iszero(norm(v)) && delete!(t, k)
    end
    return t
end

"""
    droptol!(t::AbstractBlockTensorMap, tol=eps(real(scalartype(t)))^(3/4))

Remove the tensor entries of a blocktensor that have norm `≤(tol)`.
"""
function droptol!(t::AbstractBlockTensorMap, tol = eps(real(scalartype(t)))^(3 / 4))
    for (k, v) in nonzero_pairs(t)
        norm(v) ≤ tol && delete!(t, k)
    end
    return t
end
