const AdjointBlockTensorMap{T, S, N₁, N₂, TT <: AbstractBlockTensorMap} =
    AdjointTensorMap{T, S, N₁, N₂, TT}

function permute_adjointindices(t::AbstractTensorMap, I::CartesianIndex)
    return CartesianIndex(
        TT.getindices(I.I, adjointtensorindices(t, ntuple(identity, length(I.I))))
    )
end

function Base.adjoint(t::AbstractBlockTensorMap)
    ATT = Base.promote_op(adjoint, eltype(t))
    # ATT = AdjointTensorMap{T,spacetype(t),numin(t),numout(t),TT}
    if issparse(t)
        tdst = SparseBlockTensorMap{ATT}(undef_blocks, domain(t) ← codomain(t))
        for (I, v) in nonzero_pairs(t)
            J = permute_adjointindices(tdst, I)
            tdst[J] = adjoint(v)
        end
    else
        tdst = BlockTensorMap{ATT}(undef_blocks, domain(t) ← codomain(t))
        for I in eachindex(IndexCartesian(), t)
            J = permute_adjointindices(tdst, I)
            v = t[I]
            tdst[J] = adjoint(v)
        end
    end
    return tdst
end

# help out inference
function Base.promote_op(
        ::typeof(Base.adjoint), ::Type{AbstractTensorMap{T, S, N₁, N₂}}
    ) where {T, S, N₁, N₂}
    AT = Base.promote_op(adjoint, T)
    return AbstractTensorMap{AT, S, N₂, N₁}
end

function nonzero_pairs(t::AdjointBlockTensorMap)
    return (permute_adjointindices(t, I) => v' for (I, v) in nonzero_pairs(t'))
end
function nonzero_keys(t::AdjointBlockTensorMap)
    return (permute_adjointindices(t, I) for I in nonzero_keys(t'))
end
function nonzero_values(t::AdjointBlockTensorMap)
    return (v' for v in nonzero_values(t'))
end
function nonzero_length(t::AdjointBlockTensorMap)
    return nonzero_length(t')
end
Base.haskey(t::AdjointBlockTensorMap, I::CartesianIndex) = haskey(t', permute_adjointindices(t', I))
