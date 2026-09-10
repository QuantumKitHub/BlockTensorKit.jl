# zerovector
# ----------
VI.zerovector!(t::AbstractBlockTensorMap) = (zerovector!(parent(t)); t)
VI.zerovector!(t::SparseBlockTensorMap) = (empty!(t.data); t)

# scale
# -----
function VI.scale!(t::AbstractBlockTensorMap, α::Number)
    foreach(Base.Fix2(scale!, α), nonzero_values(t))
    return t
end

function VI.scale!(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    # entries of ty that are structurally zero in tx have to be zeroed out
    issparse(tx) && zerovector!(ty)
    for (I, v) in nonzero_pairs(tx)
        ty[I] = scale!!(ty[I], v, α)
    end
    return ty
end

# add
# ---
function VI.add(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number, β::Number)
    S = TK.check_spacetype(ty, tx)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))

    # result type defaults to TensorMap if the types don't match to avoid assymmetric
    # implementation via zerovector(ty, T) vs zerovector(tx, T)
    # This would give issues for example with DiagonalTensorMap + TensorMap
    T = VectorInterface.promote_add(ty, tx, α, β)
    tdst = if typeof(ty) === typeof(tx)
        zerovector(ty, T)
    else
        M = TK.promote_storagetype(TK.similarstoragetype(ty, T), TK.similarstoragetype(tx, T))
        if issparse(ty) && issparse(tx)
            sparseblocktensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
        else
            blocktensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
        end
    end

    return add!(scale!(tdst, ty, β), tx, α)
end

function VI.add!(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    isone(β) || scale!(ty, β)
    for (I, v) in nonzero_pairs(tx)
        ty[I] = add!!(ty[I], v, α, One())
    end
    return ty
end

# inner
# -----
function VI.inner(x::AbstractBlockTensorMap, y::AbstractBlockTensorMap)
    space(x) == space(y) || throw(SpaceMismatch("$(space(x)) ≠ $(space(y))"))
    T = VI.promote_inner(x, y)
    # only entries that are nonzero in both contribute
    ks = if issparse(x) && issparse(y)
        intersect(nonzero_keys(x), nonzero_keys(y))
    else
        nonzero_keys(issparse(y) ? y : x)
    end
    return sum(ks; init = zero(T)) do I
        inner(x[I], y[I])
    end
end
